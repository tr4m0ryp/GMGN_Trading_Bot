/**
 * @file output_display.c
 * @brief Banner, status, filter config, and token log display
 *
 * Implements the display functions for the startup banner, connection
 * status, filter configuration summary, and detailed token output.
 *
 * Dependencies: "output_internal.h"
 *
 * @date 2025-12-20
 */

#include "output_internal.h"

void output_print_banner(const app_config_t *config) {
    printf("\n");

    if (g_output_use_colors) {
        printf(ANSI_BOLD ANSI_CYAN);
    }

    printf("=============================================================\n");
    printf("        GMGN Trenches Token Logger v1.0.0\n");
    printf("=============================================================\n");

    if (g_output_use_colors) {
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

    if (g_output_use_colors) {
        printf(ANSI_BOLD);
    }
    printf("Active Filters:\n");
    if (g_output_use_colors) {
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
    output_get_timestamp(timestamp, sizeof(timestamp));

    printf("[%s] ", timestamp);

    switch (state) {
        case GMGN_STATE_DISCONNECTED:
            output_print_color(ANSI_RED, "DISCONNECTED");
            break;
        case GMGN_STATE_CONNECTING:
            output_print_color(ANSI_YELLOW, "CONNECTING");
            if (url) {
                printf(" to %s", url);
            }
            break;
        case GMGN_STATE_CONNECTED:
            output_print_color(ANSI_GREEN, "CONNECTED");
            break;
        case GMGN_STATE_SUBSCRIBING:
            output_print_color(ANSI_CYAN, "SUBSCRIBING");
            break;
        case GMGN_STATE_ACTIVE:
            output_print_color(ANSI_GREEN ANSI_BOLD, "ACTIVE");
            printf(" - Monitoring for new tokens...");
            break;
        case GMGN_STATE_RECONNECTING:
            output_print_color(ANSI_YELLOW, "RECONNECTING");
            break;
        case GMGN_STATE_ERROR:
            output_print_color(ANSI_RED ANSI_BOLD, "ERROR");
            break;
    }

    printf("\n");
}

void output_log_token(const token_info_t *token, const pool_data_t *pool) {
    if (!token) {
        return;
    }

    char timestamp[32];
    output_get_timestamp(timestamp, sizeof(timestamp));

    g_output_tokens_logged++;

    if (g_output_use_colors) {
        printf(ANSI_BOLD ANSI_GREEN);
    }
    printf("\n[%s] NEW TOKEN #%" PRIu64 " ", timestamp, g_output_tokens_logged);
    if (g_output_use_colors) {
        printf(ANSI_RESET);
    }

    if (g_output_use_colors) {
        printf(ANSI_BOLD ANSI_CYAN);
    }
    printf("%s", token->symbol);
    if (g_output_use_colors) {
        printf(ANSI_RESET);
    }

    if (token->name[0] != '\0') {
        printf(" (%s)", token->name);
    }
    printf("\n");

    printf("-------------------------------------------------------------\n");

    char mc_str[32];
    output_format_market_cap(token->market_cap, mc_str, sizeof(mc_str));
    printf("  Market Cap:     ");
    if (g_output_use_colors) {
        printf(ANSI_GREEN);
    }
    printf("%s\n", mc_str);
    if (g_output_use_colors) {
        printf(ANSI_RESET);
    }

    if (pool && pool->initial_liquidity > 0) {
        char liq_str[32];
        output_format_market_cap(pool->initial_liquidity, liq_str, sizeof(liq_str));
        printf("  Liquidity:      %s\n", liq_str);
    }

    if (token->volume_24h > 0) {
        char vol_str[32];
        output_format_market_cap(token->volume_24h, vol_str, sizeof(vol_str));
        printf("  24h Volume:     %s\n", vol_str);
    }

    if (token->holder_count > 0) {
        printf("  Holders:        ");
        if (g_output_use_colors) {
            printf(ANSI_YELLOW);
        }
        printf("%u\n", token->holder_count);
        if (g_output_use_colors) {
            printf(ANSI_RESET);
        }
    }

    if (token->kol_count > 0) {
        printf("  KOL Count:      ");
        if (g_output_use_colors) {
            printf(ANSI_MAGENTA);
        }
        printf("%d\n", token->kol_count);
        if (g_output_use_colors) {
            printf(ANSI_RESET);
        }
    }

    if (token->age_seconds > 0) {
        char age_str[32];
        output_format_age(token->age_seconds, age_str, sizeof(age_str));
        printf("  Age:            %s\n", age_str);
    }

    if (token->top_10_ratio > 0) {
        printf("  Top 10 Hold:    %.1f%%\n", token->top_10_ratio / 100.0);
    }

    if (pool && pool->exchange[0] != '\0') {
        printf("  Exchange:       %s\n", pool->exchange);
    }

    if (token->address[0] != '\0') {
        printf("  Address:        ");
        if (g_output_use_colors) {
            printf(ANSI_DIM);
        }
        printf("%.20s...\n", token->address);
        if (g_output_use_colors) {
            printf(ANSI_RESET);
        }
    }

    if (pool && pool->pool_address[0] != '\0') {
        printf("  Pool:           ");
        if (g_output_use_colors) {
            printf(ANSI_DIM);
        }
        printf("%.20s...\n", pool->pool_address);
        if (g_output_use_colors) {
            printf(ANSI_RESET);
        }
    }

    if (token->address[0] != '\0') {
        printf("  GMGN:           ");
        if (g_output_use_colors) {
            printf(ANSI_CYAN);
        }
        printf("https://gmgn.ai/sol/token/%s\n", token->address);
        if (g_output_use_colors) {
            printf(ANSI_RESET);
        }
    }

    printf("=============================================================\n");
}
