/**
 * @file collector_thread.c
 * @brief Background thread for polling chart data and detecting token death
 *
 * Runs the main collection loop that periodically fetches chart data for
 * each tracked token, analyzes it for death conditions, and writes results
 * to CSV when a token is confirmed inactive.
 *
 * Dependencies: <pthread.h>, "ai_data_collector_internal.h"
 *
 * @date 2025-12-20
 */

#include "ai_data_collector_internal.h"

/**
 * @brief Clear a token slot and free its chart data
 */
void clear_token_slot(ai_tracked_token_t *token) {
    if (token->chart_data) {
        free(token->chart_data);
        token->chart_data = NULL;
    }
    memset(token, 0, sizeof(ai_tracked_token_t));
}

/**
 * @brief Background collector thread
 *
 * Iterates over all tracked token slots, polling chart data and checking
 * for death conditions. When a token is confirmed dead (consecutive checks
 * exceed threshold), writes data to CSV and frees the slot.
 *
 * Thread-safe: acquires collector->lock before accessing token state,
 * releases it during HTTP fetches to avoid blocking other operations.
 */
void *collector_thread(void *arg) {
    ai_data_collector_t *collector = (ai_data_collector_t *)arg;
    time_t now;

    while (collector->running) {
        now = time(NULL);

        pthread_mutex_lock(&collector->lock);

        for (int i = 0; i < AI_MAX_TRACKED_TOKENS; i++) {
            ai_tracked_token_t *token = &collector->tokens[i];

            if (!token->active) {
                continue;
            }

            /* Check if time to poll */
            time_t since_poll = now - token->last_poll_at;
            if (since_poll < (AI_POLL_INTERVAL_MS / 1000)) {
                continue;
            }

            token->last_poll_at = now;

            /* Fetch chart data (release lock during HTTP call) */
            char *chart_data = NULL;
            size_t chart_len = 0;

            pthread_mutex_unlock(&collector->lock);

            int fetch_result = fetch_chart_data(token->address,
                                                &chart_data, &chart_len);

            pthread_mutex_lock(&collector->lock);

            /* Token might have been removed while we were fetching */
            if (!token->active) {
                free(chart_data);
                continue;
            }

            if (fetch_result != 0) {
                token->dead_check_count++;
                if (token->dead_check_count >= AI_CONSECUTIVE_DEAD_CHECKS * 2) {
                    /* API consistently failing */
                    if (token->chart_data) {
                        write_token_to_csv(collector, token, AI_DEATH_API_FAIL);
                        collector->tokens_collected++;
                    }
                    clear_token_slot(token);
                    collector->tokens_active--;
                }
                continue;
            }

            /* Update stored chart data */
            if (token->chart_data) {
                free(token->chart_data);
            }
            token->chart_data = chart_data;
            token->chart_data_len = chart_len;

            /* Analyze for death */
            time_t tracking_time = now - token->discovered_at;
            ai_death_reason_t death = AI_DEATH_NONE;

            /* Check for timeout */
            if (tracking_time >= AI_MAX_TRACK_TIME_SEC) {
                death = AI_DEATH_TIMEOUT;
            }
            /* Only check death conditions after minimum time */
            else if (tracking_time >= AI_MIN_TRACK_TIME_SEC) {
                death = analyze_chart_data(chart_data, token);
            }

            if (death != AI_DEATH_NONE) {
                token->dead_check_count++;

                if (token->dead_check_count >= AI_CONSECUTIVE_DEAD_CHECKS ||
                    death == AI_DEATH_TIMEOUT) {
                    /* Confirmed dead - write to CSV */
                    write_token_to_csv(collector, token, death);
                    collector->tokens_collected++;
                    printf("[AI] Token %s (%s) written to CSV - %s\n",
                           token->symbol, token->address,
                           ai_death_reason_str(death));
                    clear_token_slot(token);
                    collector->tokens_active--;
                }
            } else {
                /* Reset dead count if activity resumed */
                token->dead_check_count = 0;
            }
        }

        pthread_mutex_unlock(&collector->lock);

        /* Sleep between iterations */
        usleep(500000); /* 500ms */
    }

    return NULL;
}
