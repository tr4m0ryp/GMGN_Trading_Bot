/**
 * @file callbacks.c
 * @brief WebSocket event callbacks and periodic statistics
 *
 * Contains the callback functions invoked by the WebSocket client and
 * token tracker, plus the periodic statistics printer. These are wired
 * up by main() and operate on the global state defined in ai_main.c.
 *
 * Dependencies: "gmgn_types.h", "token_tracker.h", "ai_data_collector.h"
 *
 * @date 2025-12-20
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "gmgn_types.h"
#include "websocket_client.h"
#include "filter.h"
#include "output.h"
#include "token_tracker.h"
#include "ai_data_collector.h"
#include "ai_main_internal.h"

/**
 * @brief Callback when token passes filters - add to AI collector
 *
 * This is the hook point where we start tracking a token for AI data.
 * Called by the token tracker when a token passes all filter criteria.
 *
 * @param tracked Tracked token information from tracker
 * @param info Current token info from API
 * @param user_data Unused
 */
void on_token_passed(const tracked_token_t *tracked,
                     const token_info_t *info, void *user_data) {
    (void)user_data;
    (void)info;

    g_tokens_passed++;

    /* Log to output */
    pool_data_t pool;
    memset(&pool, 0, sizeof(pool));
    memcpy(pool.exchange, tracked->exchange, sizeof(pool.exchange));
    memcpy(pool.base_token.symbol, tracked->symbol,
           sizeof(pool.base_token.symbol));
    memcpy(pool.base_token.address, tracked->address,
           sizeof(pool.base_token.address));
    pool.base_token.market_cap = info->market_cap;
    pool.base_token.kol_count = info->kol_count;
    pool.base_token.holder_count = info->holder_count;
    pool.base_token.age_seconds = (uint32_t)(time(NULL) - tracked->discovered_at);

    output_log_token(&pool.base_token, &pool);

    /* Add to AI collector for chart data tracking */
    if (g_collector) {
        uint32_t age_sec = (uint32_t)(time(NULL) - tracked->discovered_at);
        int result = ai_collector_add_token(g_collector,
                                            tracked->address,
                                            tracked->symbol,
                                            age_sec);
        if (result == 0) {
            char msg[128];
            snprintf(msg, sizeof(msg), "AI tracking started: %s",
                     tracked->symbol);
            output_print_info(msg);
        }
    }
}

/**
 * @brief Callback for new pool events from WebSocket
 *
 * Adds token to tracker for periodic re-checking against filters.
 *
 * @param pool Pool data from WebSocket
 * @param user_data Unused
 */
void on_new_pool(const pool_data_t *pool, void *user_data) {
    (void)user_data;

    if (!pool) {
        return;
    }

    g_tokens_seen++;

    /* Add token to tracker for periodic re-checking */
    if (g_tracker) {
        tracker_add_token(g_tracker, pool);
    }
}

/**
 * @brief Callback for pair update events (price/volume updates)
 */
void on_pair_update(const pool_data_t *pool, void *user_data) {
    (void)user_data;
    (void)pool;
    /* Pair updates are handled by tracker internally */
}

/**
 * @brief Callback for token launch events
 */
void on_token_launch(const pool_data_t *pool, void *user_data) {
    (void)user_data;

    if (!pool) {
        return;
    }

    /* Treat launches same as new pools */
    if (g_tracker) {
        tracker_add_token(g_tracker, pool);
    }
}

/**
 * @brief Callback for WebSocket errors
 */
void on_error(int error_code, const char *error_msg, void *user_data) {
    (void)user_data;
    output_print_error(error_code, error_msg);
}

/**
 * @brief Print periodic statistics
 *
 * Gathers stats from the WebSocket client, token tracker, and AI collector
 * and prints a summary to stdout.
 */
void print_periodic_stats(void) {
    uint64_t messages = 0;
    uint64_t bytes = 0;
    uint32_t reconnects = 0;

    if (g_client) {
        ws_client_get_stats(g_client, &messages, &bytes, &reconnects);
    }

    uint32_t uptime = (uint32_t)(time(NULL) - g_start_time);

    /* Get tracker stats */
    uint32_t tracking = 0, passed = 0, expired = 0;
    if (g_tracker) {
        tracker_get_stats(g_tracker, &tracking, &passed, &expired);
    }

    /* Get AI collector stats */
    uint32_t ai_active = 0, ai_collected = 0;
    if (g_collector) {
        ai_active = ai_collector_get_active_count(g_collector);
        ai_collected = ai_collector_get_total_collected(g_collector);
    }

    output_print_stats(g_tokens_seen, g_tokens_passed, messages, uptime);
    printf("  Tracker: %u active | %u expired\n", tracking, expired);
    printf("  AI Data: %u tracking | %u collected to CSV\n",
           ai_active, ai_collected);
}
