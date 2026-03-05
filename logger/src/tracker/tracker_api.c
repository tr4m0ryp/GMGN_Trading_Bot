/**
 * @file tracker_api.c
 * @brief Token tracker public API implementation
 *
 * Implements the public functions for initializing, starting,
 * stopping, and interacting with the token tracker.
 *
 * Dependencies: "token_tracker_internal.h"
 *
 * @date 2025-12-20
 */

#include "token_tracker_internal.h"

int tracker_init(token_tracker_t *tracker, filter_config_t *filter) {
    if (!tracker || !filter) {
        return -1;
    }

    memset(tracker, 0, sizeof(token_tracker_t));
    tracker->filter = filter;

    if (pthread_mutex_init(&tracker->lock, NULL) != 0) {
        return -1;
    }

    /* Initialize CURL globally */
    curl_global_init(CURL_GLOBAL_DEFAULT);

    /* Initialize persistent CURL handles for fast API calls */
    tracker_init_curl_handles();

    return 0;
}

int tracker_start(token_tracker_t *tracker) {
    if (!tracker) {
        return -1;
    }

    tracker->running = true;

    if (pthread_create(&tracker->thread, NULL, tracker_thread_func, tracker) != 0) {
        tracker->running = false;
        return -1;
    }

    return 0;
}

void tracker_stop(token_tracker_t *tracker) {
    if (!tracker) {
        return;
    }

    tracker->running = false;
    pthread_join(tracker->thread, NULL);
}

void tracker_cleanup(token_tracker_t *tracker) {
    if (!tracker) {
        return;
    }

    tracker_stop(tracker);
    pthread_mutex_destroy(&tracker->lock);

    /* Cleanup persistent CURL handles */
    tracker_cleanup_curl_handles();

    curl_global_cleanup();
}

int tracker_add_token(token_tracker_t *tracker, const pool_data_t *pool) {
    if (!tracker || !pool) {
        return -1;
    }

    pthread_mutex_lock(&tracker->lock);

    /* Check if already tracking */
    if (tracker_find_token(tracker, pool->base_token.address) >= 0) {
        pthread_mutex_unlock(&tracker->lock);
        return 1;
    }

    /* Find free slot */
    int slot = tracker_find_free_slot(tracker);
    if (slot < 0) {
        pthread_mutex_unlock(&tracker->lock);
        return -1;
    }

    tracked_token_t *t = &tracker->tokens[slot];
    memset(t, 0, sizeof(tracked_token_t));

    memcpy(t->address, pool->base_token.address, sizeof(t->address));
    memcpy(t->symbol, pool->base_token.symbol, sizeof(t->symbol));
    memcpy(t->exchange, pool->exchange, sizeof(t->exchange));
    t->discovered_at = time(NULL);
    t->last_check_ms = 0; /* Force immediate check */
    t->last_market_cap = pool->base_token.market_cap;
    t->last_kol_count = pool->base_token.kol_count;
    t->peak_kol_count = pool->base_token.kol_count;
    t->state = TOKEN_STATE_TRACKING;

    tracker->active_count++;

    const char *verbose = getenv("GMGN_DEBUG");
    if (verbose && verbose[0] == '1') {
        FILE *debug_log = fopen("/tmp/gmgn_debug.log", "a");
        if (debug_log) {
            fprintf(debug_log, "[TRACKER] Token added to tracker: %s (slot=%d, discovered_at=%ld)\n",
                    t->symbol, slot, (long)t->discovered_at);
            fclose(debug_log);
        }
    }

    pthread_mutex_unlock(&tracker->lock);

    return 0;
}

void tracker_set_callback(token_tracker_t *tracker,
                          void (*callback)(const tracked_token_t *,
                                          const token_info_t *, void *),
                          void *user_data) {
    if (!tracker) {
        return;
    }

    pthread_mutex_lock(&tracker->lock);
    tracker->on_token_passed = callback;
    tracker->callback_user_data = user_data;
    pthread_mutex_unlock(&tracker->lock);
}

void tracker_get_stats(const token_tracker_t *tracker,
                       uint32_t *active, uint32_t *passed, uint32_t *expired) {
    if (!tracker) {
        return;
    }

    if (active) *active = tracker->active_count;
    if (passed) *passed = tracker->passed_count;
    if (expired) *expired = tracker->expired_count;
}

bool tracker_check_token(token_tracker_t *tracker, const char *address) {
    if (!tracker || !address) {
        return false;
    }

    token_info_t info;
    memset(&info, 0, sizeof(info));

    if (tracker_fetch_token_info(address, &info) != 0) {
        return false;
    }

    bool mc_pass = filter_check_market_cap(info.market_cap, tracker->filter);
    bool kol_pass = filter_check_kol(info.kol_count, tracker->filter);

    return mc_pass && kol_pass;
}
