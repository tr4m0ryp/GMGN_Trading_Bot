/**
 * @file collector_api.c
 * @brief Public API implementation for the AI data collector
 *
 * Provides the external-facing functions declared in ai_data_collector.h:
 * initialization, start/stop lifecycle, token addition, statistics queries,
 * and the death reason string converter.
 *
 * Dependencies: <pthread.h>, <curl/curl.h>, "ai_data_collector_internal.h"
 *
 * @date 2025-12-20
 */

#include "ai_data_collector_internal.h"

/**
 * @brief Initialize the AI data collector
 *
 * Sets up the collector state, creates the data output directory if needed,
 * initializes the mutex and CURL subsystem.
 *
 * @param collector Pointer to collector structure to initialize
 * @param data_dir Directory path for CSV output (will be created if needed)
 *
 * @return 0 on success, -1 on failure
 */
int ai_collector_init(ai_data_collector_t *collector, const char *data_dir) {
    struct stat st = {0};

    if (!collector || !data_dir) {
        return -1;
    }

    memset(collector, 0, sizeof(ai_data_collector_t));

    /* Create data directory if needed */
    if (stat(data_dir, &st) == -1) {
        if (mkdir(data_dir, 0755) != 0) {
            fprintf(stderr, "[AI] Failed to create data dir: %s\n",
                    strerror(errno));
            return -1;
        }
    }

    strncpy(collector->data_dir, data_dir, sizeof(collector->data_dir) - 1);

    if (pthread_mutex_init(&collector->lock, NULL) != 0) {
        return -1;
    }

    /* Initialize CURL */
    curl_global_init(CURL_GLOBAL_DEFAULT);
    init_curl_handle();

    return 0;
}

/**
 * @brief Start the AI data collector background thread
 *
 * @param collector Initialized collector structure
 *
 * @return 0 on success, -1 on failure
 */
int ai_collector_start(ai_data_collector_t *collector) {
    if (!collector) {
        return -1;
    }

    collector->running = true;

    if (pthread_create(&collector->thread, NULL,
                       collector_thread, collector) != 0) {
        collector->running = false;
        return -1;
    }

    printf("[AI] Data collector started - writing to %s/\n",
           collector->data_dir);
    return 0;
}

/**
 * @brief Stop the AI data collector
 *
 * Signals the background thread to stop, joins it, then writes any
 * remaining active tokens to CSV with a timeout death reason.
 *
 * @param collector Running collector structure
 */
void ai_collector_stop(ai_data_collector_t *collector) {
    if (!collector) {
        return;
    }

    collector->running = false;
    pthread_join(collector->thread, NULL);

    /* Write remaining tokens with timeout reason */
    pthread_mutex_lock(&collector->lock);
    for (int i = 0; i < AI_MAX_TRACKED_TOKENS; i++) {
        ai_tracked_token_t *token = &collector->tokens[i];
        if (token->active && token->chart_data) {
            write_token_to_csv(collector, token, AI_DEATH_TIMEOUT);
            collector->tokens_collected++;
        }
    }
    pthread_mutex_unlock(&collector->lock);

    printf("[AI] Data collector stopped - %u tokens collected\n",
           collector->tokens_collected);
}

/**
 * @brief Cleanup the AI data collector
 *
 * Frees all token data, destroys the mutex, and cleans up CURL.
 *
 * @param collector Collector structure to cleanup
 */
void ai_collector_cleanup(ai_data_collector_t *collector) {
    if (!collector) {
        return;
    }

    for (int i = 0; i < AI_MAX_TRACKED_TOKENS; i++) {
        clear_token_slot(&collector->tokens[i]);
    }

    pthread_mutex_destroy(&collector->lock);
    cleanup_curl_handle();
    curl_global_cleanup();
}

/**
 * @brief Add a token for AI data tracking
 *
 * Thread-safe. Checks for duplicates before inserting into a free slot.
 *
 * @param collector Active collector structure
 * @param address Token mint address
 * @param symbol Token symbol
 * @param discovered_age_sec Age reported by WebSocket at discovery
 *
 * @return 0 on success, -1 if full or error
 */
int ai_collector_add_token(ai_data_collector_t *collector,
                           const char *address,
                           const char *symbol,
                           uint32_t discovered_age_sec) {
    int slot = -1;

    if (!collector || !address || !symbol) {
        return -1;
    }

    pthread_mutex_lock(&collector->lock);

    /* Check if already tracking */
    for (int i = 0; i < AI_MAX_TRACKED_TOKENS; i++) {
        if (collector->tokens[i].active &&
            strcmp(collector->tokens[i].address, address) == 0) {
            pthread_mutex_unlock(&collector->lock);
            return 0; /* Already tracking */
        }
    }

    /* Find free slot */
    for (int i = 0; i < AI_MAX_TRACKED_TOKENS; i++) {
        if (!collector->tokens[i].active) {
            slot = i;
            break;
        }
    }

    if (slot < 0) {
        pthread_mutex_unlock(&collector->lock);
        fprintf(stderr, "[AI] No free slots for token tracking\n");
        return -1;
    }

    /* Initialize token slot */
    ai_tracked_token_t *token = &collector->tokens[slot];
    memset(token, 0, sizeof(ai_tracked_token_t));

    strncpy(token->address, address, sizeof(token->address) - 1);
    strncpy(token->symbol, symbol, sizeof(token->symbol) - 1);
    token->discovered_at = time(NULL);
    token->discovered_age_sec = discovered_age_sec;
    token->last_poll_at = 0; /* Force immediate first poll */
    token->active = true;

    collector->tokens_active++;

    pthread_mutex_unlock(&collector->lock);

    return 0;
}

/**
 * @brief Get count of actively tracked tokens
 */
uint32_t ai_collector_get_active_count(ai_data_collector_t *collector) {
    if (!collector) {
        return 0;
    }
    return collector->tokens_active;
}

/**
 * @brief Get total tokens collected to CSV
 */
uint32_t ai_collector_get_total_collected(ai_data_collector_t *collector) {
    if (!collector) {
        return 0;
    }
    return collector->tokens_collected;
}

/**
 * @brief Convert death reason to string
 */
const char *ai_death_reason_str(ai_death_reason_t reason) {
    switch (reason) {
        case AI_DEATH_NONE:
            return "active";
        case AI_DEATH_VOLUME_LOW:
            return "volume_low";
        case AI_DEATH_CANDLE_GAP:
            return "candle_gap";
        case AI_DEATH_PRICE_STABLE:
            return "price_stable";
        case AI_DEATH_TIMEOUT:
            return "timeout";
        case AI_DEATH_API_FAIL:
            return "api_fail";
        default:
            return "unknown";
    }
}
