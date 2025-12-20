/**
 * @file output.c
 * @brief Terminal output implementation for GMGN logger
 *
 * Provides formatted terminal output with ANSI color codes for
 * displaying token information, connection status, and statistics.
 *
 * Dependencies: "output.h", "gmgn_types.h", "filter.h"
 *
 * @date 2025-12-20
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <inttypes.h>

#include "output.h"
#include "filter.h"

/* ANSI color codes */
#define ANSI_RESET      "\033[0m"
#define ANSI_BOLD       "\033[1m"
#define ANSI_DIM        "\033[2m"
#define ANSI_RED        "\033[31m"
#define ANSI_GREEN      "\033[32m"
#define ANSI_YELLOW     "\033[33m"
#define ANSI_BLUE       "\033[34m"
#define ANSI_MAGENTA    "\033[35m"
#define ANSI_CYAN       "\033[36m"
#define ANSI_WHITE      "\033[37m"
#define ANSI_BG_RED     "\033[41m"
#define ANSI_BG_GREEN   "\033[42m"
#define ANSI_BG_BLUE    "\033[44m"

/* Static state */
static output_verbosity_t s_verbosity = OUTPUT_NORMAL;
static int s_use_colors = 1;
static uint64_t s_tokens_logged = 0;

/**
 * @brief Check if terminal supports colors
 */
static int check_color_support(void) {
    const char *term = getenv("TERM");
    if (!term) {
        return 0;
    }
    
    if (strstr(term, "color") || strstr(term, "xterm") || 
        strstr(term, "256") || strstr(term, "screen") ||
        strstr(term, "tmux")) {
        return 1;
    }
    
    return isatty(STDOUT_FILENO);
}

/**
 * @brief Get current timestamp string
 */
static void get_timestamp(char *buffer, size_t size) {
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    strftime(buffer, size, "%H:%M:%S", tm_info);
}

/**
 * @brief Print colored text
 */
static void print_color(const char *color, const char *text) {
    if (s_use_colors) {
        printf("%s%s%s", color, text, ANSI_RESET);
    } else {
        printf("%s", text);
    }
}

int output_init(output_verbosity_t verbosity) {
    s_verbosity = verbosity;
    s_use_colors = check_color_support();
    s_tokens_logged = 0;
    
    /* Disable buffering for real-time output */
    setvbuf(stdout, NULL, _IONBF, 0);
    
    return 0;
}

void output_cleanup(void) {
    /* Nothing to clean up currently */
}

void output_print_banner(const app_config_t *config) {
    printf("\n");
    
    if (s_use_colors) {
        printf(ANSI_BOLD ANSI_CYAN);
    }
    
    printf("=============================================================\n");
    printf("        GMGN Trenches Token Logger v1.0.0\n");
    printf("=============================================================\n");
    
    if (s_use_colors) {
        printf(ANSI_RESET);
    }
    
    if (config) {
        printf("  Target Chain: %s\n", config->chain);
        printf("  WebSocket:    %s\n", config->websocket_url);
        printf("  Verbosity:    %d\n", config->verbosity);
    }
    
    printf("\n");
}

void output_print_filter_config(const filter_config_t *filter) {
    if (!filter) {
        return;
    }
    
    char summary[512];
    filter_get_summary(filter, summary, sizeof(summary));
    
    if (s_use_colors) {
        printf(ANSI_BOLD);
    }
    printf("Active Filters:\n");
    if (s_use_colors) {
        printf(ANSI_RESET);
    }
    
    printf("  Market Cap:    $%.2fK - $%.2fK\n",
           filter->min_market_cap / 100000.0,
           filter->max_market_cap / 100000.0);
    
    printf("  Min Liquidity: $%.2fK\n",
           filter->min_liquidity / 100000.0);
    
    printf("  Min KOL:       %d\n", filter->min_kol_count);
    printf("  Max Age:       %u minutes\n", filter->max_age_seconds / 60);
    printf("  Min Holders:   %u\n", filter->min_holder_count);
    printf("  Max Top10:     %.1f%%\n", filter->max_top_10_ratio / 100.0);
    
    if (filter->exchange_count > 0) {
        printf("  Exchanges:     ");
        for (uint8_t i = 0; i < filter->exchange_count; i++) {
            printf("%s%s", filter->exchanges[i],
                   (i < filter->exchange_count - 1) ? ", " : "\n");
        }
    }
    
    if (filter->exclude_count > 0) {
        printf("  Excluded:      ");
        for (uint8_t i = 0; i < filter->exclude_count; i++) {
            printf("%s%s", filter->exclude_symbols[i],
                   (i < filter->exclude_count - 1) ? ", " : "\n");
        }
    }
    
    printf("\n");
}

void output_print_connection_status(gmgn_conn_state_t state, const char *url) {
    char timestamp[32];
    get_timestamp(timestamp, sizeof(timestamp));
    
    printf("[%s] ", timestamp);
    
    switch (state) {
        case GMGN_STATE_DISCONNECTED:
            print_color(ANSI_RED, "DISCONNECTED");
            break;
            
        case GMGN_STATE_CONNECTING:
            print_color(ANSI_YELLOW, "CONNECTING");
            if (url) {
                printf(" to %s", url);
            }
            break;
            
        case GMGN_STATE_CONNECTED:
            print_color(ANSI_GREEN, "CONNECTED");
            break;
            
        case GMGN_STATE_SUBSCRIBING:
            print_color(ANSI_CYAN, "SUBSCRIBING");
            break;
            
        case GMGN_STATE_ACTIVE:
            print_color(ANSI_GREEN ANSI_BOLD, "ACTIVE");
            printf(" - Monitoring for new tokens...");
            break;
            
        case GMGN_STATE_RECONNECTING:
            print_color(ANSI_YELLOW, "RECONNECTING");
            break;
            
        case GMGN_STATE_ERROR:
            print_color(ANSI_RED ANSI_BOLD, "ERROR");
            break;
    }
    
    printf("\n");
}

void output_log_token(const token_info_t *token, const pool_data_t *pool) {
    if (!token) {
        return;
    }
    
    char timestamp[32];
    get_timestamp(timestamp, sizeof(timestamp));
    
    s_tokens_logged++;
    
    /* Print separator */
    if (s_use_colors) {
        printf(ANSI_BOLD ANSI_GREEN);
    }
    printf("\n[%s] NEW TOKEN #%" PRIu64 " ", timestamp, s_tokens_logged);
    if (s_use_colors) {
        printf(ANSI_RESET);
    }
    
    /* Token symbol and name */
    if (s_use_colors) {
        printf(ANSI_BOLD ANSI_CYAN);
    }
    printf("%s", token->symbol);
    if (s_use_colors) {
        printf(ANSI_RESET);
    }
    
    if (token->name[0] != '\0') {
        printf(" (%s)", token->name);
    }
    printf("\n");
    
    /* Print horizontal line */
    printf("-------------------------------------------------------------\n");
    
    /* Market cap */
    char mc_str[32];
    output_format_market_cap(token->market_cap, mc_str, sizeof(mc_str));
    printf("  Market Cap:     ");
    if (s_use_colors) {
        printf(ANSI_GREEN);
    }
    printf("%s\n", mc_str);
    if (s_use_colors) {
        printf(ANSI_RESET);
    }
    
    /* Liquidity (from pool) */
    if (pool && pool->initial_liquidity > 0) {
        char liq_str[32];
        output_format_market_cap(pool->initial_liquidity, liq_str, sizeof(liq_str));
        printf("  Liquidity:      %s\n", liq_str);
    }
    
    /* 24h Volume */
    if (token->volume_24h > 0) {
        char vol_str[32];
        output_format_market_cap(token->volume_24h, vol_str, sizeof(vol_str));
        printf("  24h Volume:     %s\n", vol_str);
    }
    
    /* Holders */
    if (token->holder_count > 0) {
        printf("  Holders:        ");
        if (s_use_colors) {
            printf(ANSI_YELLOW);
        }
        printf("%u\n", token->holder_count);
        if (s_use_colors) {
            printf(ANSI_RESET);
        }
    }
    
    /* KOL count */
    if (token->kol_count > 0) {
        printf("  KOL Count:      ");
        if (s_use_colors) {
            printf(ANSI_MAGENTA);
        }
        printf("%d\n", token->kol_count);
        if (s_use_colors) {
            printf(ANSI_RESET);
        }
    }
    
    /* Token age */
    if (token->age_seconds > 0) {
        char age_str[32];
        output_format_age(token->age_seconds, age_str, sizeof(age_str));
        printf("  Age:            %s\n", age_str);
    }
    
    /* Top 10 concentration */
    if (token->top_10_ratio > 0) {
        printf("  Top 10 Hold:    %.1f%%\n", token->top_10_ratio / 100.0);
    }
    
    /* Exchange */
    if (pool && pool->exchange[0] != '\0') {
        printf("  Exchange:       %s\n", pool->exchange);
    }
    
    /* Token address */
    if (token->address[0] != '\0') {
        printf("  Address:        ");
        if (s_use_colors) {
            printf(ANSI_DIM);
        }
        printf("%.20s...\n", token->address);
        if (s_use_colors) {
            printf(ANSI_RESET);
        }
    }
    
    /* Pool address */
    if (pool && pool->pool_address[0] != '\0') {
        printf("  Pool:           ");
        if (s_use_colors) {
            printf(ANSI_DIM);
        }
        printf("%.20s...\n", pool->pool_address);
        if (s_use_colors) {
            printf(ANSI_RESET);
        }
    }
    
    /* GMGN Link */
    if (token->address[0] != '\0') {
        printf("  GMGN:           ");
        if (s_use_colors) {
            printf(ANSI_CYAN);
        }
        printf("https://gmgn.ai/sol/token/%s\n", token->address);
        if (s_use_colors) {
            printf(ANSI_RESET);
        }
    }
    
    printf("=============================================================\n");
}

void output_log_filtered(const token_info_t *token, const char *reason) {
    if (s_verbosity < OUTPUT_VERBOSE || !token) {
        return;
    }
    
    char timestamp[32];
    get_timestamp(timestamp, sizeof(timestamp));
    
    if (s_use_colors) {
        printf(ANSI_DIM);
    }
    printf("[%s] FILTERED: %s - %s\n", timestamp, token->symbol, 
           reason ? reason : "unknown");
    if (s_use_colors) {
        printf(ANSI_RESET);
    }
}

void output_print_stats(uint64_t tokens_seen, uint64_t tokens_passed,
                        uint64_t messages_received, uint32_t uptime_seconds) {
    char timestamp[32];
    get_timestamp(timestamp, sizeof(timestamp));
    
    printf("\n[%s] ", timestamp);
    if (s_use_colors) {
        printf(ANSI_BOLD);
    }
    printf("Statistics:\n");
    if (s_use_colors) {
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
    
    /* Format uptime */
    uint32_t hours = uptime_seconds / 3600;
    uint32_t minutes = (uptime_seconds % 3600) / 60;
    uint32_t seconds = uptime_seconds % 60;
    printf("  Uptime:         %02u:%02u:%02u\n", hours, minutes, seconds);
}

void output_print_error(int error_code, const char *message) {
    char timestamp[32];
    get_timestamp(timestamp, sizeof(timestamp));
    
    printf("[%s] ", timestamp);
    if (s_use_colors) {
        printf(ANSI_BOLD ANSI_RED);
    }
    printf("ERROR");
    if (s_use_colors) {
        printf(ANSI_RESET ANSI_RED);
    }
    printf(" (%d): %s\n", error_code, message ? message : "Unknown error");
    if (s_use_colors) {
        printf(ANSI_RESET);
    }
}

void output_print_warning(const char *message) {
    if (s_verbosity < OUTPUT_NORMAL || !message) {
        return;
    }
    
    char timestamp[32];
    get_timestamp(timestamp, sizeof(timestamp));
    
    printf("[%s] ", timestamp);
    if (s_use_colors) {
        printf(ANSI_YELLOW);
    }
    printf("WARNING: %s\n", message);
    if (s_use_colors) {
        printf(ANSI_RESET);
    }
}

void output_print_info(const char *message) {
    if (s_verbosity < OUTPUT_NORMAL || !message) {
        return;
    }
    
    char timestamp[32];
    get_timestamp(timestamp, sizeof(timestamp));
    
    printf("[%s] %s\n", timestamp, message);
}

void output_print_debug(const char *message) {
    if (s_verbosity < OUTPUT_VERBOSE || !message) {
        return;
    }
    
    char timestamp[32];
    get_timestamp(timestamp, sizeof(timestamp));
    
    if (s_use_colors) {
        printf(ANSI_DIM);
    }
    printf("[%s] DEBUG: %s\n", timestamp, message);
    if (s_use_colors) {
        printf(ANSI_RESET);
    }
}

void output_print_heartbeat(void) {
    if (s_verbosity < OUTPUT_VERBOSE) {
        return;
    }
    
    if (s_use_colors) {
        printf(ANSI_DIM);
    }
    printf(".");
    if (s_use_colors) {
        printf(ANSI_RESET);
    }
    fflush(stdout);
}

char *output_format_market_cap(uint64_t market_cap_cents, char *buffer, 
                               size_t buffer_size) {
    if (!buffer || buffer_size == 0) {
        return NULL;
    }
    
    double value = market_cap_cents / 100.0;  /* Convert cents to dollars */
    
    if (value >= 1000000000.0) {
        snprintf(buffer, buffer_size, "$%.2fB", value / 1000000000.0);
    } else if (value >= 1000000.0) {
        snprintf(buffer, buffer_size, "$%.2fM", value / 1000000.0);
    } else if (value >= 1000.0) {
        snprintf(buffer, buffer_size, "$%.2fK", value / 1000.0);
    } else {
        snprintf(buffer, buffer_size, "$%.2f", value);
    }
    
    return buffer;
}

char *output_format_age(uint32_t age_seconds, char *buffer, size_t buffer_size) {
    if (!buffer || buffer_size == 0) {
        return NULL;
    }
    
    if (age_seconds >= 86400) {
        uint32_t days = age_seconds / 86400;
        uint32_t hours = (age_seconds % 86400) / 3600;
        snprintf(buffer, buffer_size, "%ud %uh", days, hours);
    } else if (age_seconds >= 3600) {
        uint32_t hours = age_seconds / 3600;
        uint32_t minutes = (age_seconds % 3600) / 60;
        snprintf(buffer, buffer_size, "%uh %um", hours, minutes);
    } else if (age_seconds >= 60) {
        uint32_t minutes = age_seconds / 60;
        uint32_t seconds = age_seconds % 60;
        snprintf(buffer, buffer_size, "%um %us", minutes, seconds);
    } else {
        snprintf(buffer, buffer_size, "%us", age_seconds);
    }
    
    return buffer;
}

void output_set_verbosity(output_verbosity_t verbosity) {
    s_verbosity = verbosity;
}

output_verbosity_t output_get_verbosity(void) {
    return s_verbosity;
}
