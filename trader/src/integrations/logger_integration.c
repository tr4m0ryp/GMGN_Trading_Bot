/**
 * @file logger_integration.c
 * @brief Implementation of logger integration
 *
 * Provides callbacks for the token_tracker in the logger to notify
 * the trader when new tokens pass filter criteria.
 *
 * Integration modes:
 * 1. Shared library linking (compile logger with trader)
 * 2. IPC via Unix socket or shared memory (separate processes)
 * 3. File-based polling (simplest, reads from token list file)
 *
 * Currently implements file-based polling for simplicity.
 *
 * @date 2025-12-24
 */

/* Disable truncation warnings - we handle string sizes manually */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-truncation"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>

#include "logger_integration.h"
#include "trade_logger.h"

/* File polling configuration */
#define TOKEN_LIST_DIR          "../logger/tokens"
#define TOKEN_LIST_FILE         "passed_tokens.txt"
#define POLL_INTERVAL_MS        500
#define MAX_LINE_LEN            256

/* Internal state for file polling */
static struct {
    logger_integration_t *integration;
    pthread_t poll_thread;
    volatile bool running;
    long last_file_pos;         /* Track file position to read new entries */
    char last_token[64];        /* Last token seen to avoid duplicates */
} g_logger_poll = {0};

/**
 * @brief Parse a token line from the file
 *
 * Format: ADDRESS:SYMBOL
 */
static int parse_token_line(const char *line, char *address, char *symbol) {
    const char *colon = strchr(line, ':');
    if (!colon) return -1;

    size_t addr_len = colon - line;
    if (addr_len >= 64) addr_len = 63;
    strncpy(address, line, addr_len);
    address[addr_len] = '\0';

    const char *sym_start = colon + 1;
    size_t sym_len = strlen(sym_start);
    /* Remove newline */
    while (sym_len > 0 && (sym_start[sym_len-1] == '\n' || sym_start[sym_len-1] == '\r')) {
        sym_len--;
    }
    if (sym_len >= 32) sym_len = 31;
    strncpy(symbol, sym_start, sym_len);
    symbol[sym_len] = '\0';

    return 0;
}

/**
 * @brief File polling thread
 *
 * Monitors the token list file for new entries.
 */
static void *poll_thread_func(void *arg) {
    (void)arg;
    char filepath[512];
    char line[MAX_LINE_LEN];
    char address[64];
    char symbol[32];
    struct stat st;
    FILE *fp;

    snprintf(filepath, sizeof(filepath), "%s/%s", TOKEN_LIST_DIR, TOKEN_LIST_FILE);

    log_info("[LOGGER] Starting token file polling: %s", filepath);

    while (g_logger_poll.running) {
        /* Check if file exists and has new content */
        if (stat(filepath, &st) == 0) {
            /* Only read if file has grown */
            if (st.st_size > g_logger_poll.last_file_pos) {
                fp = fopen(filepath, "r");
                if (fp) {
                    /* Seek to last position */
                    fseek(fp, g_logger_poll.last_file_pos, SEEK_SET);

                    /* Read new lines */
                    while (fgets(line, sizeof(line), fp)) {
                        if (parse_token_line(line, address, symbol) == 0) {
                            /* Avoid duplicate dispatch */
                            if (strcmp(address, g_logger_poll.last_token) != 0) {
                                strncpy(g_logger_poll.last_token, address,
                                        sizeof(g_logger_poll.last_token) - 1);

                                log_info("[LOGGER] New token discovered: %s (%s)",
                                         symbol, address);

                                /* Call callback */
                                if (g_logger_poll.integration &&
                                    g_logger_poll.integration->callback) {
                                    g_logger_poll.integration->callback(
                                        address, symbol,
                                        g_logger_poll.integration->user_data);
                                }
                            }
                        }
                    }

                    g_logger_poll.last_file_pos = ftell(fp);
                    fclose(fp);
                }
            }
        }

        /* Sleep for poll interval */
        usleep(POLL_INTERVAL_MS * 1000);
    }

    log_info("[LOGGER] Token polling stopped");
    return NULL;
}

int logger_integration_init(logger_integration_t *integration) {
    if (!integration) return -1;

    memset(integration, 0, sizeof(logger_integration_t));
    return 0;
}

void logger_integration_set_callback(logger_integration_t *integration,
                                     new_token_callback_t callback,
                                     void *user_data) {
    if (!integration) return;

    integration->callback = callback;
    integration->user_data = user_data;
}

int logger_integration_connect(logger_integration_t *integration) {
    if (!integration) return -1;

    /* Ensure token directory exists */
    struct stat st;
    if (stat(TOKEN_LIST_DIR, &st) != 0) {
        /* Directory doesn't exist - try to create it */
        if (mkdir(TOKEN_LIST_DIR, 0755) != 0) {
            log_warn("[LOGGER] Token directory doesn't exist: %s", TOKEN_LIST_DIR);
            /* Continue anyway - logger_c may create it later */
        }
    }

    /* Initialize polling state */
    g_logger_poll.integration = integration;
    g_logger_poll.running = true;
    g_logger_poll.last_file_pos = 0;
    g_logger_poll.last_token[0] = '\0';

    /* Start polling thread */
    if (pthread_create(&g_logger_poll.poll_thread, NULL, poll_thread_func, NULL) != 0) {
        log_error("[LOGGER] Failed to start polling thread");
        return -1;
    }

    integration->connected = true;
    log_info("[LOGGER] Connected - monitoring for new tokens");

    return 0;
}

void logger_integration_disconnect(logger_integration_t *integration) {
    if (!integration || !integration->connected) return;

    g_logger_poll.running = false;
    pthread_join(g_logger_poll.poll_thread, NULL);

    integration->connected = false;
}

void logger_integration_cleanup(logger_integration_t *integration) {
    if (!integration) return;

    logger_integration_disconnect(integration);
    memset(integration, 0, sizeof(logger_integration_t));
}
