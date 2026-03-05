/**
 * @file output_logging.c
 * @brief Log level output functions (error, warning, info, debug)
 *
 * Implements the verbosity-aware log output functions for errors,
 * warnings, informational messages, debug output, and statistics.
 *
 * Dependencies: "output_internal.h"
 *
 * @date 2025-12-20
 */

#include "output_internal.h"

void output_log_filtered(const token_info_t *token, const char *reason) {
    if (g_output_verbosity < OUTPUT_VERBOSE || !token) {
        return;
    }

    char timestamp[32];
    output_get_timestamp(timestamp, sizeof(timestamp));

    if (g_output_use_colors) {
        printf(ANSI_DIM);
    }
    printf("[%s] FILTERED: %s - %s\n", timestamp, token->symbol,
           reason ? reason : "unknown");
    if (g_output_use_colors) {
        printf(ANSI_RESET);
    }
}

void output_print_stats(uint64_t tokens_seen, uint64_t tokens_passed,
                        uint64_t messages_received, uint32_t uptime_seconds) {
    char timestamp[32];
    output_get_timestamp(timestamp, sizeof(timestamp));

    printf("\n[%s] ", timestamp);
    if (g_output_use_colors) {
        printf(ANSI_BOLD);
    }
    printf("Statistics:\n");
    if (g_output_use_colors) {
        printf(ANSI_RESET);
    }

    printf("  Tokens Seen:    %" PRIu64 "\n", tokens_seen);
    printf("  Tokens Passed:  %" PRIu64 "", tokens_passed);

    if (tokens_seen > 0) {
        double pass_rate = (double)tokens_passed / (double)tokens_seen * 100.0;
        printf(" (%.1f%%)", pass_rate);
    }
    printf("\n");

    printf("  Messages:       %" PRIu64 "\n", messages_received);

    uint32_t hours = uptime_seconds / 3600;
    uint32_t minutes = (uptime_seconds % 3600) / 60;
    uint32_t seconds = uptime_seconds % 60;
    printf("  Uptime:         %02u:%02u:%02u\n", hours, minutes, seconds);
}

void output_print_error(int error_code, const char *message) {
    char timestamp[32];
    output_get_timestamp(timestamp, sizeof(timestamp));

    printf("[%s] ", timestamp);
    if (g_output_use_colors) {
        printf(ANSI_BOLD ANSI_RED);
    }
    printf("ERROR");
    if (g_output_use_colors) {
        printf(ANSI_RESET ANSI_RED);
    }
    printf(" (%d): %s\n", error_code, message ? message : "Unknown error");
    if (g_output_use_colors) {
        printf(ANSI_RESET);
    }
}

void output_print_warning(const char *message) {
    if (g_output_verbosity < OUTPUT_NORMAL || !message) {
        return;
    }

    char timestamp[32];
    output_get_timestamp(timestamp, sizeof(timestamp));

    printf("[%s] ", timestamp);
    if (g_output_use_colors) {
        printf(ANSI_YELLOW);
    }
    printf("WARNING: %s\n", message);
    if (g_output_use_colors) {
        printf(ANSI_RESET);
    }
}

void output_print_info(const char *message) {
    if (g_output_verbosity < OUTPUT_NORMAL || !message) {
        return;
    }

    char timestamp[32];
    output_get_timestamp(timestamp, sizeof(timestamp));

    printf("[%s] %s\n", timestamp, message);
}

void output_print_debug(const char *message) {
    if (g_output_verbosity < OUTPUT_VERBOSE || !message) {
        return;
    }

    char timestamp[32];
    output_get_timestamp(timestamp, sizeof(timestamp));

    if (g_output_use_colors) {
        printf(ANSI_DIM);
    }
    printf("[%s] DEBUG: %s\n", timestamp, message);
    if (g_output_use_colors) {
        printf(ANSI_RESET);
    }
}

void output_print_heartbeat(void) {
    if (g_output_verbosity < OUTPUT_VERBOSE) {
        return;
    }

    if (g_output_use_colors) {
        printf(ANSI_DIM);
    }
    printf(".");
    if (g_output_use_colors) {
        printf(ANSI_RESET);
    }
    fflush(stdout);
}
