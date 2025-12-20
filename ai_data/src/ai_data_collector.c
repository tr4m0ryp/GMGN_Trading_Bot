/**
 * @file ai_data_collector.c
 * @brief Implementation of AI training data collection
 *
 * This module tracks tokens that pass filters and collects chart data
 * until the token becomes inactive. Data is written to CSV for AI training.
 *
 * The collector runs a background thread that:
 * 1. Polls chart data every AI_POLL_INTERVAL_MS
 * 2. Checks death conditions (volume, candle gap, price stability)
 * 3. Writes to CSV when token is confirmed dead
 *
 * Dependencies: <curl/curl.h>, <cjson/cJSON.h>, <pthread.h>
 *
 * @date 2025-12-20
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <errno.h>
#include <curl/curl.h>
#include <cjson/cJSON.h>

#include "ai_data_collector.h"

/* CURL response buffer */
typedef struct {
    char *data;
    size_t size;
} curl_buffer_t;

/* Global CURL handle for chart API (persistent connection) */
static CURL *g_curl_chart = NULL;
static struct curl_slist *g_headers_chart = NULL;

/**
 * @brief CURL write callback for API responses
 */
static size_t curl_write_cb(void *contents, size_t size, size_t nmemb, void *userp) {
    size_t realsize = size * nmemb;
    curl_buffer_t *buf = (curl_buffer_t *)userp;

    char *ptr = realloc(buf->data, buf->size + realsize + 1);
    if (!ptr) {
        return 0;
    }

    buf->data = ptr;
    memcpy(&buf->data[buf->size], contents, realsize);
    buf->size += realsize;
    buf->data[buf->size] = '\0';

    return realsize;
}

/**
 * @brief Initialize persistent CURL handle for chart API
 */
static void init_curl_handle(void) {
    char cookie_header[2048];

    if (g_curl_chart) {
        return;
    }

    snprintf(cookie_header, sizeof(cookie_header),
        "cf_clearance=%s; _ga=%s; _ga_0XM0LYXGC8=%s; __cf_bm=%s",
        getenv("GMGN_CF_CLEARANCE") ? getenv("GMGN_CF_CLEARANCE") : "",
        getenv("GMGN_GA") ? getenv("GMGN_GA") : "",
        getenv("GMGN_GA_SESSION") ? getenv("GMGN_GA_SESSION") : "",
        getenv("GMGN_CF_BM") ? getenv("GMGN_CF_BM") : "");

    g_headers_chart = curl_slist_append(g_headers_chart, "Accept: application/json");
    g_headers_chart = curl_slist_append(g_headers_chart, "Accept-Language: en-US,en;q=0.5");
    g_headers_chart = curl_slist_append(g_headers_chart, "Referer: https://gmgn.ai/");
    g_headers_chart = curl_slist_append(g_headers_chart, "Origin: https://gmgn.ai");
    g_headers_chart = curl_slist_append(g_headers_chart, "Sec-Fetch-Dest: empty");
    g_headers_chart = curl_slist_append(g_headers_chart, "Sec-Fetch-Mode: cors");
    g_headers_chart = curl_slist_append(g_headers_chart, "Sec-Fetch-Site: same-origin");

    g_curl_chart = curl_easy_init();
    if (g_curl_chart) {
        curl_easy_setopt(g_curl_chart, CURLOPT_TIMEOUT, 10L);
        curl_easy_setopt(g_curl_chart, CURLOPT_CONNECTTIMEOUT, 5L);
        curl_easy_setopt(g_curl_chart, CURLOPT_USERAGENT,
            "Mozilla/5.0 (X11; Linux x86_64; rv:145.0) Gecko/20100101 Firefox/145.0");
        curl_easy_setopt(g_curl_chart, CURLOPT_ACCEPT_ENCODING, "");
        curl_easy_setopt(g_curl_chart, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(g_curl_chart, CURLOPT_COOKIE, cookie_header);
        curl_easy_setopt(g_curl_chart, CURLOPT_HTTPHEADER, g_headers_chart);
        curl_easy_setopt(g_curl_chart, CURLOPT_TCP_KEEPALIVE, 1L);
        curl_easy_setopt(g_curl_chart, CURLOPT_DNS_CACHE_TIMEOUT, 300L);
    }
}

/**
 * @brief Cleanup CURL handle
 */
static void cleanup_curl_handle(void) {
    if (g_curl_chart) {
        curl_easy_cleanup(g_curl_chart);
        g_curl_chart = NULL;
    }
    if (g_headers_chart) {
        curl_slist_free_all(g_headers_chart);
        g_headers_chart = NULL;
    }
}

/**
 * @brief Fetch chart data from GMGN API
 *
 * @param address Token address
 * @param out_data Output buffer for JSON data (caller must free)
 * @param out_len Output length of data
 *
 * @return 0 on success, -1 on failure
 */
static int fetch_chart_data(const char *address, char **out_data, size_t *out_len) {
    char url[1024];
    curl_buffer_t buffer = {0};
    CURLcode res;

    if (!g_curl_chart || !address || !out_data || !out_len) {
        return -1;
    }

    snprintf(url, sizeof(url),
        "%s%s?pool_type=tpool&resolution=1s&limit=501",
        AI_MCAP_CANDLES_API, address);

    buffer.data = malloc(1);
    buffer.size = 0;

    curl_easy_setopt(g_curl_chart, CURLOPT_URL, url);
    curl_easy_setopt(g_curl_chart, CURLOPT_WRITEFUNCTION, curl_write_cb);
    curl_easy_setopt(g_curl_chart, CURLOPT_WRITEDATA, &buffer);

    res = curl_easy_perform(g_curl_chart);

    if (res == CURLE_OK && buffer.data && buffer.size > 0) {
        long http_code = 0;
        curl_easy_getinfo(g_curl_chart, CURLINFO_RESPONSE_CODE, &http_code);

        if (http_code == 200) {
            *out_data = buffer.data;
            *out_len = buffer.size;
            return 0;
        }
    }

    free(buffer.data);
    *out_data = NULL;
    *out_len = 0;
    return -1;
}

/**
 * @brief Analyze chart data for death conditions
 *
 * @param json_data Raw JSON chart data
 * @param token Token structure to update with analysis
 *
 * @return Death reason if dead, AI_DEATH_NONE if still active
 */
static ai_death_reason_t analyze_chart_data(const char *json_data,
                                             ai_tracked_token_t *token) {
    cJSON *json = NULL;
    cJSON *data = NULL;
    cJSON *list = NULL;
    int count = 0;
    double volume_sum = 0.0;
    double last_close = 0.0;
    double first_close = 0.0;
    time_t last_time = 0;
    time_t prev_time = 0;
    int gap_violations = 0;

    json = cJSON_Parse(json_data);
    if (!json) {
        return AI_DEATH_NONE;
    }

    data = cJSON_GetObjectItemCaseSensitive(json, "data");
    if (!data || !cJSON_IsObject(data)) {
        cJSON_Delete(json);
        return AI_DEATH_NONE;
    }

    list = cJSON_GetObjectItemCaseSensitive(data, "list");
    if (!list || !cJSON_IsArray(list)) {
        cJSON_Delete(json);
        return AI_DEATH_NONE;
    }

    count = cJSON_GetArraySize(list);
    if (count < 3) {
        cJSON_Delete(json);
        return AI_DEATH_NONE;
    }

    /* Analyze last 3 candles for volume and price */
    for (int i = count - 3; i < count; i++) {
        cJSON *candle = cJSON_GetArrayItem(list, i);
        if (!candle) {
            continue;
        }

        cJSON *volume = cJSON_GetObjectItemCaseSensitive(candle, "volume");
        cJSON *close = cJSON_GetObjectItemCaseSensitive(candle, "close");
        cJSON *time_val = cJSON_GetObjectItemCaseSensitive(candle, "time");

        if (volume) {
            if (cJSON_IsString(volume)) {
                volume_sum += atof(volume->valuestring);
            } else if (cJSON_IsNumber(volume)) {
                volume_sum += volume->valuedouble;
            }
        }

        if (close) {
            double close_val = 0.0;
            if (cJSON_IsString(close)) {
                close_val = atof(close->valuestring);
            } else if (cJSON_IsNumber(close)) {
                close_val = close->valuedouble;
            }

            if (i == count - 3) {
                first_close = close_val;
            }
            last_close = close_val;
        }

        if (time_val && cJSON_IsNumber(time_val)) {
            prev_time = last_time;
            last_time = (time_t)(time_val->valuedouble / 1000.0);

            if (prev_time > 0 && last_time > prev_time) {
                time_t gap = last_time - prev_time;
                if (gap > AI_CANDLE_GAP_THRESHOLD_SEC) {
                    gap_violations++;
                }
            }
        }
    }

    /* Update token tracking data */
    token->last_volume_avg = volume_sum / 3.0;
    token->last_close_price = last_close;
    token->last_candle_time = last_time;

    cJSON_Delete(json);

    /* Check time since last candle */
    time_t now = time(NULL);
    time_t since_last = now - last_time;

    if (since_last > AI_CANDLE_GAP_THRESHOLD_SEC) {
        gap_violations++;
    }

    /* Check death conditions */
    ai_death_reason_t reason = AI_DEATH_NONE;

    if (gap_violations >= 2) {
        reason = AI_DEATH_CANDLE_GAP;
    } else if (token->last_volume_avg < AI_VOLUME_THRESHOLD_SOL) {
        reason = AI_DEATH_VOLUME_LOW;
    } else if (first_close > 0.0 && last_close > 0.0) {
        double change = (last_close - first_close) / first_close;
        if (change > -AI_PRICE_CHANGE_THRESHOLD &&
            change < AI_PRICE_CHANGE_THRESHOLD) {
            reason = AI_DEATH_PRICE_STABLE;
        }
    }

    return reason;
}

/**
 * @brief Escape string for CSV (handles quotes and newlines)
 */
static void csv_escape(FILE *fp, const char *str) {
    fputc('"', fp);
    while (*str) {
        if (*str == '"') {
            fputs("\"\"", fp);
        } else {
            fputc(*str, fp);
        }
        str++;
    }
    fputc('"', fp);
}

/**
 * @brief Extract clean candle data from raw API response
 *
 * Parses the raw JSON and extracts only the candle list with essential fields:
 * time, open, high, low, close, volume
 *
 * This removes unnecessary wrapper fields like code, reason, message, _debug_tpool
 * to save storage space and simplify ML training data.
 *
 * @param raw_json Raw JSON response from API
 *
 * @return Newly allocated string with clean JSON array, or NULL on failure
 *         Caller must free the returned string
 */
static char *extract_candle_data(const char *raw_json) {
    cJSON *root = NULL;
    cJSON *data = NULL;
    cJSON *list = NULL;
    cJSON *clean_array = NULL;
    cJSON *candle = NULL;
    char *result = NULL;

    if (!raw_json) {
        return NULL;
    }

    root = cJSON_Parse(raw_json);
    if (!root) {
        return NULL;
    }

    /* Navigate to data.list */
    data = cJSON_GetObjectItemCaseSensitive(root, "data");
    if (!data || !cJSON_IsObject(data)) {
        cJSON_Delete(root);
        return NULL;
    }

    list = cJSON_GetObjectItemCaseSensitive(data, "list");
    if (!list || !cJSON_IsArray(list)) {
        cJSON_Delete(root);
        return NULL;
    }

    /* Create clean array with only essential fields */
    clean_array = cJSON_CreateArray();
    if (!clean_array) {
        cJSON_Delete(root);
        return NULL;
    }

    cJSON_ArrayForEach(candle, list) {
        cJSON *clean_candle = cJSON_CreateObject();
        if (!clean_candle) {
            continue;
        }

        /* Extract time (convert ms to seconds) */
        cJSON *time_val = cJSON_GetObjectItemCaseSensitive(candle, "time");
        if (time_val && cJSON_IsNumber(time_val)) {
            cJSON_AddNumberToObject(clean_candle, "t",
                                    (double)((int64_t)(time_val->valuedouble / 1000.0)));
        }

        /* Extract OHLCV - convert strings to numbers for efficiency */
        cJSON *open = cJSON_GetObjectItemCaseSensitive(candle, "open");
        if (open) {
            double val = cJSON_IsString(open) ? atof(open->valuestring) : open->valuedouble;
            cJSON_AddNumberToObject(clean_candle, "o", val);
        }

        cJSON *high = cJSON_GetObjectItemCaseSensitive(candle, "high");
        if (high) {
            double val = cJSON_IsString(high) ? atof(high->valuestring) : high->valuedouble;
            cJSON_AddNumberToObject(clean_candle, "h", val);
        }

        cJSON *low = cJSON_GetObjectItemCaseSensitive(candle, "low");
        if (low) {
            double val = cJSON_IsString(low) ? atof(low->valuestring) : low->valuedouble;
            cJSON_AddNumberToObject(clean_candle, "l", val);
        }

        cJSON *close = cJSON_GetObjectItemCaseSensitive(candle, "close");
        if (close) {
            double val = cJSON_IsString(close) ? atof(close->valuestring) : close->valuedouble;
            cJSON_AddNumberToObject(clean_candle, "c", val);
        }

        cJSON *volume = cJSON_GetObjectItemCaseSensitive(candle, "volume");
        if (volume) {
            double val = cJSON_IsString(volume) ? atof(volume->valuestring) : volume->valuedouble;
            cJSON_AddNumberToObject(clean_candle, "v", val);
        }

        cJSON_AddItemToArray(clean_array, clean_candle);
    }

    /* Generate compact JSON string (no formatting) */
    result = cJSON_PrintUnformatted(clean_array);

    cJSON_Delete(clean_array);
    cJSON_Delete(root);

    return result;
}

/**
 * @brief Write token data to CSV file
 *
 * Extracts clean candle data from raw JSON before writing.
 * Output format: [{"t":timestamp,"o":open,"h":high,"l":low,"c":close,"v":volume},...]
 *
 * @param collector Collector state
 * @param token Token to write
 * @param death_reason Why the token was declared dead
 *
 * @return 0 on success, -1 on failure
 */
static int write_token_to_csv(ai_data_collector_t *collector,
                               ai_tracked_token_t *token,
                               ai_death_reason_t death_reason) {
    char filepath[512];
    FILE *fp = NULL;
    struct stat st = {0};
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    bool write_header = false;
    char *clean_data = NULL;

    snprintf(filepath, sizeof(filepath),
        "%s/tokens_%04d-%02d-%02d.csv",
        collector->data_dir,
        tm_info->tm_year + 1900,
        tm_info->tm_mon + 1,
        tm_info->tm_mday);

    /* Check if file exists to determine if we need header */
    if (stat(filepath, &st) != 0) {
        write_header = true;
    }

    fp = fopen(filepath, "a");
    if (!fp) {
        fprintf(stderr, "[AI] Failed to open CSV: %s\n", strerror(errno));
        return -1;
    }

    if (write_header) {
        fprintf(fp, "token_address,symbol,discovered_at_unix,discovered_age_sec,"
                    "death_reason,candles\n");
    }

    /* Write row */
    fprintf(fp, "%s,", token->address);
    fprintf(fp, "%s,", token->symbol);
    fprintf(fp, "%ld,", (long)token->discovered_at);
    fprintf(fp, "%u,", token->discovered_age_sec);
    fprintf(fp, "%s,", ai_death_reason_str(death_reason));

    /* Extract clean candle data and write */
    if (token->chart_data) {
        clean_data = extract_candle_data(token->chart_data);
        if (clean_data) {
            csv_escape(fp, clean_data);
            free(clean_data);
        } else {
            fprintf(fp, "\"[]\"");
        }
    } else {
        fprintf(fp, "\"[]\"");
    }

    fprintf(fp, "\n");
    fclose(fp);

    return 0;
}

/**
 * @brief Clear a token slot
 */
static void clear_token_slot(ai_tracked_token_t *token) {
    if (token->chart_data) {
        free(token->chart_data);
        token->chart_data = NULL;
    }
    memset(token, 0, sizeof(ai_tracked_token_t));
}

/**
 * @brief Background collector thread
 */
static void *collector_thread(void *arg) {
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

            /* Fetch chart data */
            char *chart_data = NULL;
            size_t chart_len = 0;

            pthread_mutex_unlock(&collector->lock);

            int fetch_result = fetch_chart_data(token->address, &chart_data, &chart_len);

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

/* Public API implementation */

int ai_collector_init(ai_data_collector_t *collector, const char *data_dir) {
    struct stat st = {0};

    if (!collector || !data_dir) {
        return -1;
    }

    memset(collector, 0, sizeof(ai_data_collector_t));

    /* Create data directory if needed */
    if (stat(data_dir, &st) == -1) {
        if (mkdir(data_dir, 0755) != 0) {
            fprintf(stderr, "[AI] Failed to create data dir: %s\n", strerror(errno));
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

int ai_collector_start(ai_data_collector_t *collector) {
    if (!collector) {
        return -1;
    }

    collector->running = true;

    if (pthread_create(&collector->thread, NULL, collector_thread, collector) != 0) {
        collector->running = false;
        return -1;
    }

    printf("[AI] Data collector started - writing to %s/\n", collector->data_dir);
    return 0;
}

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

uint32_t ai_collector_get_active_count(ai_data_collector_t *collector) {
    if (!collector) {
        return 0;
    }
    return collector->tokens_active;
}

uint32_t ai_collector_get_total_collected(ai_data_collector_t *collector) {
    if (!collector) {
        return 0;
    }
    return collector->tokens_collected;
}

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
