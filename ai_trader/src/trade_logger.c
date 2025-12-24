/**
 * @file trade_logger.c
 * @brief Implementation of structured logging system
 *
 * Creates organized log files by date with separate streams for:
 * - General logs (info, debug, warn)
 * - Trade records (CSV format)
 * - Errors only
 *
 * Dependencies: <stdio.h>, <stdlib.h>, <string.h>, <stdarg.h>,
 *               <time.h>, <sys/stat.h>, <errno.h>, <pthread.h>
 *
 * @date 2025-12-24
 */

/* Disable format-truncation warning for this file - paths are validated */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#include <pthread.h>

#include "trade_logger.h"

/**
 * @brief Safe string copy that avoids truncation warnings
 */
static void safe_strcpy(char *dest, size_t dest_size, const char *src) {
    if (!dest || !src || dest_size == 0) {
        return;
    }
    size_t src_len = strlen(src);
    size_t copy_len = (src_len < dest_size - 1) ? src_len : dest_size - 1;
    memcpy(dest, src, copy_len);
    dest[copy_len] = '\0';
}

/* Increase path buffer for safety */
#define SAFE_PATH_LEN 1024

/* Global logger state */
static struct {
    char base_dir[LOG_MAX_PATH_LEN];
    char current_dir[LOG_MAX_PATH_LEN];
    FILE *fp_main;              /* Main log file */
    FILE *fp_trades;            /* Trades CSV file */
    FILE *fp_errors;            /* Errors log file */
    log_level_t min_level;
    pthread_mutex_t lock;
    time_t last_flush;
    int current_day;            /* Day of month for rotation */
    bool initialized;
} g_logger = {0};

/**
 * @brief Get current timestamp string
 */
static void get_timestamp(char *buffer, size_t len) {
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    strftime(buffer, len, "%Y-%m-%d %H:%M:%S", tm_info);
}

/**
 * @brief Create directory if it doesn't exist
 */
static int ensure_directory(const char *path) {
    struct stat st = {0};
    if (stat(path, &st) == -1) {
        if (mkdir(path, 0755) != 0) {
            return -1;
        }
    }
    return 0;
}

/**
 * @brief Create/update daily log directory and files
 */
static int rotate_log_files(void) {
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    char date_dir[SAFE_PATH_LEN];
    char path[SAFE_PATH_LEN];
    int ret;

    /* Check if we need to rotate */
    if (g_logger.current_day == tm_info->tm_mday && g_logger.fp_main) {
        return 0; /* No rotation needed */
    }

    /* Close existing files */
    if (g_logger.fp_main) {
        fclose(g_logger.fp_main);
        g_logger.fp_main = NULL;
    }
    if (g_logger.fp_trades) {
        fclose(g_logger.fp_trades);
        g_logger.fp_trades = NULL;
    }
    if (g_logger.fp_errors) {
        fclose(g_logger.fp_errors);
        g_logger.fp_errors = NULL;
    }

    /* Create date directory */
    snprintf(date_dir, sizeof(date_dir), "%s/%04d-%02d-%02d",
             g_logger.base_dir,
             tm_info->tm_year + 1900,
             tm_info->tm_mon + 1,
             tm_info->tm_mday);

    if (ensure_directory(date_dir) != 0) {
        fprintf(stderr, "[LOGGER] Cannot create directory: %s\n", date_dir);
        return -1;
    }

    safe_strcpy(g_logger.current_dir, sizeof(g_logger.current_dir), date_dir);

    /* Open main log */
    ret = snprintf(path, sizeof(path), "%s/trader.log", date_dir);
    g_logger.fp_main = fopen(path, "a");
    if (!g_logger.fp_main) {
        fprintf(stderr, "[LOGGER] Cannot open %s: %s\n", path, strerror(errno));
        return -1;
    }
    /* Open trades CSV */
    ret = snprintf(path, sizeof(path), "%s/trades.csv", date_dir);
    (void)ret;  /* Suppress unused warning */
    bool trades_exists = (access(path, F_OK) == 0);
    g_logger.fp_trades = fopen(path, "a");
    if (!g_logger.fp_trades) {
        fprintf(stderr, "[LOGGER] Cannot open %s: %s\n", path, strerror(errno));
        return -1;
    }

    /* Write CSV header if new file */
    if (!trades_exists) {
        fprintf(g_logger.fp_trades,
                "timestamp,address,symbol,action,entry_price,exit_price,"
                "amount_sol,pnl_sol,pnl_pct,confidence,balance,"
                "is_paper,seq_len,error\n");
    }
    /* Open errors log */
    ret = snprintf(path, sizeof(path), "%s/errors.log", date_dir);
    (void)ret;
    g_logger.fp_errors = fopen(path, "a");
    if (!g_logger.fp_errors) {
        fprintf(stderr, "[LOGGER] Cannot open %s: %s\n", path, strerror(errno));
        return -1;
    }

    g_logger.current_day = tm_info->tm_mday;
    return 0;
}

int logger_init(const char *log_dir) {
    if (!log_dir) {
        return -1;
    }

    if (g_logger.initialized) {
        return 0; /* Already initialized */
    }

    memset(&g_logger, 0, sizeof(g_logger));

    /* Create base directory */
    if (ensure_directory(log_dir) != 0) {
        fprintf(stderr, "[LOGGER] Cannot create log directory: %s\n", log_dir);
        return -1;
    }

    safe_strcpy(g_logger.base_dir, sizeof(g_logger.base_dir), log_dir);
    g_logger.min_level = LOG_LEVEL_INFO;

    if (pthread_mutex_init(&g_logger.lock, NULL) != 0) {
        return -1;
    }

    /* Create initial log files */
    if (rotate_log_files() != 0) {
        return -1;
    }

    g_logger.initialized = true;

    /* Log startup */
    log_info("Logger initialized: %s", log_dir);

    return 0;
}

void logger_set_level(log_level_t level) {
    g_logger.min_level = level;
}

/**
 * @brief Internal log function
 */
static void log_message(log_level_t level, const char *fmt, va_list args) {
    char timestamp[32];
    char message[LOG_MAX_MSG_LEN];
    const char *level_str;

    if (!g_logger.initialized || level < g_logger.min_level) {
        return;
    }

    /* Level string */
    switch (level) {
        case LOG_LEVEL_DEBUG: level_str = "DEBUG"; break;
        case LOG_LEVEL_INFO:  level_str = "INFO";  break;
        case LOG_LEVEL_WARN:  level_str = "WARN";  break;
        case LOG_LEVEL_ERROR: level_str = "ERROR"; break;
        default: level_str = "???"; break;
    }

    get_timestamp(timestamp, sizeof(timestamp));
    vsnprintf(message, sizeof(message), fmt, args);

    pthread_mutex_lock(&g_logger.lock);

    /* Check for day rotation */
    rotate_log_files();

    /* Write to main log */
    if (g_logger.fp_main) {
        fprintf(g_logger.fp_main, "[%s] [%s] %s\n", timestamp, level_str, message);
    }

    /* Write errors to error log too */
    if (level == LOG_LEVEL_ERROR && g_logger.fp_errors) {
        fprintf(g_logger.fp_errors, "[%s] %s\n", timestamp, message);
    }

    /* Also print to stdout */
    printf("[%s] [%s] %s\n", timestamp, level_str, message);

    /* Periodic flush */
    time_t now = time(NULL);
    if (now - g_logger.last_flush >= LOG_FLUSH_INTERVAL_SEC) {
        logger_flush();
        g_logger.last_flush = now;
    }

    pthread_mutex_unlock(&g_logger.lock);
}

void log_debug(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_message(LOG_LEVEL_DEBUG, fmt, args);
    va_end(args);
}

void log_info(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_message(LOG_LEVEL_INFO, fmt, args);
    va_end(args);
}

void log_warn(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_message(LOG_LEVEL_WARN, fmt, args);
    va_end(args);
}

void log_error(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_message(LOG_LEVEL_ERROR, fmt, args);
    va_end(args);
}

int log_trade(const trade_record_t *record) {
    char timestamp[32];

    if (!g_logger.initialized || !record) {
        return -1;
    }

    pthread_mutex_lock(&g_logger.lock);

    /* Check for day rotation */
    rotate_log_files();

    if (!g_logger.fp_trades) {
        pthread_mutex_unlock(&g_logger.lock);
        return -1;
    }

    /* Format timestamp */
    struct tm *tm_info = localtime(&record->timestamp);
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", tm_info);

    /* Write CSV row */
    fprintf(g_logger.fp_trades,
            "%s,%s,%s,%s,%.10f,%.10f,%.6f,%.8f,%.4f,%.4f,%.6f,%d,%d,\"%s\"\n",
            timestamp,
            record->token_address,
            record->token_symbol,
            record->action,
            record->entry_price,
            record->exit_price,
            record->amount_sol,
            record->pnl_sol,
            record->pnl_pct,
            record->confidence,
            record->balance_after,
            record->is_paper ? 1 : 0,
            record->sequence_length,
            record->error_msg[0] ? record->error_msg : "");

    /* Also log to main log */
    if (g_logger.fp_main) {
        fprintf(g_logger.fp_main,
                "[%s] [TRADE] %s %s %s @ %.10f, PnL: %.8f SOL (%.2f%%)\n",
                timestamp,
                record->is_paper ? "[PAPER]" : "[LIVE]",
                record->action,
                record->token_symbol,
                record->exit_price,
                record->pnl_sol,
                record->pnl_pct * 100);
    }

    pthread_mutex_unlock(&g_logger.lock);
    return 0;
}

void logger_flush(void) {
    if (!g_logger.initialized) {
        return;
    }

    if (g_logger.fp_main) {
        fflush(g_logger.fp_main);
    }
    if (g_logger.fp_trades) {
        fflush(g_logger.fp_trades);
    }
    if (g_logger.fp_errors) {
        fflush(g_logger.fp_errors);
    }
}

void logger_cleanup(void) {
    if (!g_logger.initialized) {
        return;
    }

    log_info("Logger shutting down");

    pthread_mutex_lock(&g_logger.lock);

    if (g_logger.fp_main) {
        fclose(g_logger.fp_main);
        g_logger.fp_main = NULL;
    }
    if (g_logger.fp_trades) {
        fclose(g_logger.fp_trades);
        g_logger.fp_trades = NULL;
    }
    if (g_logger.fp_errors) {
        fclose(g_logger.fp_errors);
        g_logger.fp_errors = NULL;
    }

    pthread_mutex_unlock(&g_logger.lock);
    pthread_mutex_destroy(&g_logger.lock);

    g_logger.initialized = false;
}

int logger_get_current_dir(char *buffer, size_t buffer_len) {
    if (!buffer || buffer_len < strlen(g_logger.current_dir) + 1) {
        return -1;
    }

    safe_strcpy(buffer, buffer_len, g_logger.current_dir);
    return 0;
}
