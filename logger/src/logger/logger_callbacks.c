/**
 * @file logger_callbacks.c
 * @brief Event callbacks for GMGN logger
 *
 * Implements the callback functions invoked when WebSocket events
 * occur: new pool discovery, pair updates, token launches, errors,
 * and tracker pass notifications.
 *
 * Dependencies: "gmgn_types.h", "token_tracker.h", "output.h"
 *
 * @date 2025-12-20
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <inttypes.h>

#include "gmgn_types.h"
#include "token_tracker.h"
#include "output.h"

/* External global state from gmgn_logger.c (main) */
extern token_tracker_t *g_tracker;
extern uint64_t g_tokens_seen;
extern uint64_t g_tokens_passed;
extern uint64_t g_pair_updates;
extern uint64_t g_token_launches;

void logger_on_token_passed(const tracked_token_t *tracked,
                            const token_info_t *info, void *user_data) {
    (void)user_data;

    g_tokens_passed++;

    pool_data_t pool;
    memset(&pool, 0, sizeof(pool));
    memcpy(pool.exchange, tracked->exchange, sizeof(pool.exchange));
    memcpy(pool.base_token.symbol, tracked->symbol, sizeof(pool.base_token.symbol));
    memcpy(pool.base_token.address, tracked->address, sizeof(pool.base_token.address));
    pool.base_token.market_cap = info->market_cap;
    pool.base_token.kol_count = info->kol_count;
    pool.base_token.holder_count = info->holder_count;
    pool.base_token.age_seconds = (uint32_t)(time(NULL) - tracked->discovered_at);

    output_log_token(&pool.base_token, &pool);
}

void logger_on_new_pool(const pool_data_t *pool, void *user_data) {
    (void)user_data;

    if (!pool) {
        return;
    }

    g_tokens_seen++;

    const char *verbose = getenv("GMGN_DEBUG");
    if (verbose && verbose[0] == '1') {
        FILE *debug_log = fopen("/tmp/gmgn_debug.log", "a");
        if (debug_log) {
            fprintf(debug_log, "[CALLBACK] Token #%" PRIu64 " seen: %s (MC: $%.2fK, KOL: %u, Ex: %s, Addr: %.20s...)\n",
                    g_tokens_seen,
                    pool->base_token.symbol[0] ? pool->base_token.symbol : "???",
                    pool->base_token.market_cap / 100000.0,
                    pool->base_token.kol_count,
                    pool->exchange,
                    pool->base_token.address);
            fclose(debug_log);
        }
    }

    if (g_tracker) {
        int result = tracker_add_token(g_tracker, pool);
        if (verbose && verbose[0] == '1') {
            FILE *debug_log = fopen("/tmp/gmgn_debug.log", "a");
            if (debug_log) {
                if (result == 0) {
                    fprintf(debug_log, "[TRACKER] Token added to tracker: %s\n",
                            pool->base_token.symbol);
                } else if (result == 1) {
                    fprintf(debug_log, "[TRACKER] Token already in tracker: %s\n",
                            pool->base_token.symbol);
                } else {
                    fprintf(debug_log, "[TRACKER] Failed to add token to tracker: %s (error %d)\n",
                            pool->base_token.symbol, result);
                }
                fclose(debug_log);
            }
        }
    }
}

void logger_on_pair_update(const pool_data_t *pool, void *user_data) {
    (void)user_data;

    if (!pool) {
        return;
    }

    g_pair_updates++;

    const char *verbose = getenv("GMGN_DEBUG");
    if (verbose && verbose[0] == '1') {
        FILE *debug_log = fopen("/tmp/gmgn_debug.log", "a");
        if (debug_log) {
            fprintf(debug_log, "[PAIR_UPDATE] #%" PRIu64 ": %s (Price: $%.8f, MC: $%.2fK)\n",
                    g_pair_updates,
                    pool->base_token.symbol[0] ? pool->base_token.symbol : "???",
                    (double)pool->base_token.price / 100000000.0,
                    pool->base_token.market_cap / 100000.0);
            fclose(debug_log);
        }
    }
}

void logger_on_token_launch(const pool_data_t *pool, void *user_data) {
    (void)user_data;

    if (!pool) {
        return;
    }

    g_token_launches++;

    const char *verbose = getenv("GMGN_DEBUG");
    if (verbose && verbose[0] == '1') {
        FILE *debug_log = fopen("/tmp/gmgn_debug.log", "a");
        if (debug_log) {
            fprintf(debug_log, "[TOKEN_LAUNCH] #%" PRIu64 ": %s - %s (Ex: %s)\n",
                    g_token_launches,
                    pool->base_token.symbol[0] ? pool->base_token.symbol : "???",
                    pool->base_token.name[0] ? pool->base_token.name : "Unknown",
                    pool->exchange[0] ? pool->exchange : "???");
            fclose(debug_log);
        }
    }

    if (g_tracker) {
        tracker_add_token(g_tracker, pool);
    }
}

void logger_on_error(int error_code, const char *error_msg, void *user_data) {
    (void)user_data;
    output_print_error(error_code, error_msg);
}
