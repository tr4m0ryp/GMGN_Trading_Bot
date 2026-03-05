/**
 * @file tracker_thread.c
 * @brief Background tracking thread and helper utilities
 *
 * Implements the background thread that periodically re-checks
 * tracked tokens against the GMGN API and filter criteria.
 *
 * Dependencies: "token_tracker_internal.h"
 *
 * @date 2025-12-20
 */

#include "token_tracker_internal.h"

int tracker_find_free_slot(token_tracker_t *tracker) {
    for (int i = 0; i < TRACKER_MAX_TOKENS; i++) {
        if (tracker->tokens[i].state == TOKEN_STATE_REMOVED ||
            tracker->tokens[i].state == TOKEN_STATE_EXPIRED ||
            tracker->tokens[i].state == TOKEN_STATE_PASSED ||
            tracker->tokens[i].address[0] == '\0') {
            return i;
        }
    }
    return -1;
}

int tracker_find_token(token_tracker_t *tracker, const char *address) {
    for (int i = 0; i < TRACKER_MAX_TOKENS; i++) {
        if (tracker->tokens[i].state == TOKEN_STATE_TRACKING &&
            strcmp(tracker->tokens[i].address, address) == 0) {
            return i;
        }
    }
    return -1;
}

uint64_t tracker_get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)(tv.tv_sec) * 1000 + (uint64_t)(tv.tv_usec) / 1000;
}

void *tracker_thread_func(void *arg) {
    token_tracker_t *tracker = (token_tracker_t *)arg;
    const char *verbose = getenv("GMGN_DEBUG");

    while (tracker->running) {
        time_t now = time(NULL);
        uint64_t now_ms = tracker_get_time_ms();

        pthread_mutex_lock(&tracker->lock);

        for (int i = 0; i < TRACKER_MAX_TOKENS; i++) {
            tracked_token_t *t = &tracker->tokens[i];

            if (t->state != TOKEN_STATE_TRACKING) {
                continue;
            }

            if (t->address[0] == '\0') {
                continue;
            }

            /* Check if token has exceeded max age */
            uint32_t age = 0;
            if (now >= t->discovered_at) {
                age = (uint32_t)(now - t->discovered_at);
            }
            if (tracker->filter->max_age_seconds > 0 &&
                age > tracker->filter->max_age_seconds) {
                t->state = TOKEN_STATE_EXPIRED;
                tracker->expired_count++;
                if (tracker->active_count > 0) {
                    tracker->active_count--;
                }
                if (verbose && verbose[0] == '1') {
                    FILE *debug_log = fopen("/tmp/gmgn_debug.log", "a");
                    if (debug_log) {
                        fprintf(debug_log, "[TRACKER] Token expired (too old): %s (age=%us, discovered_at=%ld, now=%ld, max=%us)\n",
                                t->symbol, age, (long)t->discovered_at, (long)now, tracker->filter->max_age_seconds);
                        fclose(debug_log);
                    }
                }
                continue;
            }

            /* Check if it's time to re-check this token */
            if (now_ms - t->last_check_ms < TRACKER_CHECK_INTERVAL) {
                continue;
            }

            if (verbose && verbose[0] == '1') {
                FILE *debug_log = fopen("/tmp/gmgn_debug.log", "a");
                if (debug_log) {
                    fprintf(debug_log, "[TRACKER] Checking token %s (age: %us, check #%u)...\n",
                            t->symbol, age, t->check_count + 1);
                    fclose(debug_log);
                }
            }

            /* Fetch updated token info (release lock during network call) */
            token_info_t info;
            memset(&info, 0, sizeof(info));
            memcpy(info.symbol, t->symbol, sizeof(info.symbol));
            memcpy(info.address, t->address, sizeof(info.address));
            info.age_seconds = age;

            pthread_mutex_unlock(&tracker->lock);
            int fetch_result = tracker_fetch_token_info(t->address, &info);
            pthread_mutex_lock(&tracker->lock);

            /* Token might have been removed while we were fetching */
            if (t->state != TOKEN_STATE_TRACKING) {
                continue;
            }

            t->last_check_ms = now_ms;
            t->check_count++;

            if (fetch_result == 0) {
                t->last_market_cap = info.market_cap;
                t->last_kol_count = info.kol_count;

                if (info.kol_count > t->peak_kol_count) {
                    t->peak_kol_count = info.kol_count;
                }

                if (verbose && verbose[0] == '1') {
                    FILE *debug_log = fopen("/tmp/gmgn_debug.log", "a");
                    if (debug_log) {
                        fprintf(debug_log, "[TRACKER] API result for %s: MC=$%.2fK, KOL=%u (peak=%u)\n",
                                t->symbol, info.market_cap / 100000.0, info.kol_count, t->peak_kol_count);
                        fclose(debug_log);
                    }
                }

                bool mc_pass = filter_check_market_cap(info.market_cap, tracker->filter);
                bool kol_pass = filter_check_kol(t->peak_kol_count, tracker->filter);
                bool age_pass = filter_check_age(age, tracker->filter);

                if (verbose && verbose[0] == '1') {
                    FILE *debug_log = fopen("/tmp/gmgn_debug.log", "a");
                    if (debug_log) {
                        fprintf(debug_log, "[TRACKER] Filter check for %s: MC=%s, KOL=%s, Age=%s\n",
                                t->symbol,
                                mc_pass ? "PASS" : "FAIL",
                                kol_pass ? "PASS" : "FAIL",
                                age_pass ? "PASS" : "FAIL");
                        fclose(debug_log);
                    }
                }

                if (mc_pass && kol_pass && age_pass) {
                    t->state = TOKEN_STATE_PASSED;
                    tracker->passed_count++;
                    if (tracker->active_count > 0) {
                        tracker->active_count--;
                    }

                    if (verbose && verbose[0] == '1') {
                        FILE *debug_log = fopen("/tmp/gmgn_debug.log", "a");
                        if (debug_log) {
                            fprintf(debug_log, "[TRACKER] *** TOKEN PASSED FILTER: %s ***\n", t->symbol);
                            fclose(debug_log);
                        }
                    }

                    if (tracker->on_token_passed) {
                        pthread_mutex_unlock(&tracker->lock);
                        tracker->on_token_passed(t, &info, tracker->callback_user_data);
                        pthread_mutex_lock(&tracker->lock);
                    }
                }
            } else {
                if (verbose && verbose[0] == '1') {
                    FILE *debug_log = fopen("/tmp/gmgn_debug.log", "a");
                    if (debug_log) {
                        fprintf(debug_log, "[TRACKER] API fetch FAILED for %s\n", t->symbol);
                        fclose(debug_log);
                    }
                }
            }
        }

        pthread_mutex_unlock(&tracker->lock);

        /* Sleep before next round - ultra-fast polling */
        usleep(10000); /* 10ms */
    }

    return NULL;
}
