/**
 * @file token_tracker.c
 * @brief Token tracking implementation
 *
 * Tracks newly discovered tokens and periodically re-checks them
 * via GMGN API until they either pass filters or exceed max age.
 *
 * @date 2025-12-20
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <curl/curl.h>
#include <cjson/cJSON.h>

#include "token_tracker.h"
#include "filter.h"
#include "output.h"

/* GMGN API endpoints */
#define GMGN_KOL_HOLDERS_API "https://gmgn.ai/vas/api/v1/token_holders/sol/"
#define GMGN_MCAP_CANDLES_API "https://gmgn.ai/api/v1/token_mcap_candles/sol/"

/**
 * @brief CURL response buffer
 */
typedef struct {
    char *data;
    size_t size;
} curl_buffer_t;

/**
 * @brief CURL write callback
 */
static size_t curl_write_cb(void *contents, size_t size, size_t nmemb, 
                            void *userp) {
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
 * @brief Setup common CURL options for GMGN API calls
 */
static void setup_curl_common(CURL *curl, curl_buffer_t *buffer, 
                              const char *url, struct curl_slist **headers) {
    char cookie_header[2048];
    
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, buffer);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, TRACKER_API_TIMEOUT);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, 
        "Mozilla/5.0 (X11; Linux x86_64; rv:145.0) Gecko/20100101 Firefox/145.0");
    curl_easy_setopt(curl, CURLOPT_ACCEPT_ENCODING, "");
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    
    /* Set Cloudflare session cookies */
    snprintf(cookie_header, sizeof(cookie_header),
        "cf_clearance=%s; _ga=%s; _ga_0XM0LYXGC8=%s; __cf_bm=%s",
        getenv("GMGN_CF_CLEARANCE") ? getenv("GMGN_CF_CLEARANCE") : "",
        getenv("GMGN_GA") ? getenv("GMGN_GA") : "",
        getenv("GMGN_GA_SESSION") ? getenv("GMGN_GA_SESSION") : "",
        getenv("GMGN_CF_BM") ? getenv("GMGN_CF_BM") : "");
    curl_easy_setopt(curl, CURLOPT_COOKIE, cookie_header);
    
    /* Browser-like headers */
    *headers = curl_slist_append(*headers, "Accept: application/json");
    *headers = curl_slist_append(*headers, "Accept-Language: en-US,en;q=0.5");
    *headers = curl_slist_append(*headers, "Referer: https://gmgn.ai/");
    *headers = curl_slist_append(*headers, "Origin: https://gmgn.ai");
    *headers = curl_slist_append(*headers, "Sec-Fetch-Dest: empty");
    *headers = curl_slist_append(*headers, "Sec-Fetch-Mode: cors");
    *headers = curl_slist_append(*headers, "Sec-Fetch-Site: same-origin");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, *headers);
}

/**
 * @brief Fetch KOL count from GMGN token holders API
 * 
 * Uses /vas/api/v1/token_holders/sol/{address}?tag=renowned endpoint
 * Returns count of KOL/smart_degen holders
 */
static int fetch_kol_count(const char *address, uint8_t *kol_count) {
    CURL *curl;
    CURLcode res;
    curl_buffer_t buffer = {0};
    char url[768];
    struct curl_slist *headers = NULL;
    int ret = -1;
    
    if (!address || !kol_count) {
        return -1;
    }
    
    *kol_count = 0;
    
    /* Build URL with query params for renowned/KOL holders */
    snprintf(url, sizeof(url), 
        "%s%s?tag=renowned&limit=10&orderby=amount_percentage&direction=desc",
        GMGN_KOL_HOLDERS_API, address);
    
    curl = curl_easy_init();
    if (!curl) {
        return -1;
    }
    
    buffer.data = malloc(1);
    buffer.size = 0;
    
    setup_curl_common(curl, &buffer, url, &headers);
    
    res = curl_easy_perform(curl);
    
    if (res == CURLE_OK && buffer.data) {
        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
        
        if (http_code == 200) {
            cJSON *json = cJSON_Parse(buffer.data);
            if (json) {
                cJSON *code = cJSON_GetObjectItemCaseSensitive(json, "code");
                cJSON *data = cJSON_GetObjectItemCaseSensitive(json, "data");
                
                if (code && cJSON_IsNumber(code) && code->valueint == 0 &&
                    data && cJSON_IsObject(data)) {
                    
                    cJSON *list = cJSON_GetObjectItemCaseSensitive(data, "list");
                    if (list && cJSON_IsArray(list)) {
                        int count = cJSON_GetArraySize(list);
                        
                        /* Count entries that have "kol" or "smart_degen" in tags */
                        int kol_total = 0;
                        cJSON *holder = NULL;
                        cJSON_ArrayForEach(holder, list) {
                            cJSON *tags = cJSON_GetObjectItemCaseSensitive(holder, "tags");
                            if (tags && cJSON_IsArray(tags)) {
                                cJSON *tag = NULL;
                                cJSON_ArrayForEach(tag, tags) {
                                    if (cJSON_IsString(tag)) {
                                        if (strcmp(tag->valuestring, "kol") == 0 ||
                                            strcmp(tag->valuestring, "smart_degen") == 0) {
                                            kol_total++;
                                            break; /* Count each holder once */
                                        }
                                    }
                                }
                            }
                        }
                        
                        /* If no explicit kol/smart_degen tags, use list count
                         * (renowned tag already filters for notable holders) */
                        if (kol_total == 0 && count > 0) {
                            kol_total = count;
                        }
                        
                        *kol_count = (uint8_t)(kol_total > 255 ? 255 : kol_total);
                        ret = 0;
                    }
                }
                cJSON_Delete(json);
            }
        }
    }
    
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    free(buffer.data);
    
    return ret;
}

/**
 * @brief Fetch market cap from GMGN token_mcap_candles API
 * 
 * Uses the latest close price from 1s candles as market cap.
 * Endpoint: /api/v1/token_mcap_candles/sol/{address}
 */
static int fetch_market_cap(const char *address, uint64_t *market_cap) {
    CURL *curl;
    CURLcode res;
    curl_buffer_t buffer = {0};
    char url[1024];
    struct curl_slist *headers = NULL;
    int ret = -1;
    
    if (!address || !market_cap) {
        return -1;
    }
    
    *market_cap = 0;
    
    /* Build URL for market cap candles - matches browser request format */
    snprintf(url, sizeof(url), 
        "%s%s?pool_type=tpool&resolution=1s&limit=5",
        GMGN_MCAP_CANDLES_API, address);
    
    curl = curl_easy_init();
    if (!curl) {
        return -1;
    }
    
    buffer.data = malloc(1);
    buffer.size = 0;
    
    setup_curl_common(curl, &buffer, url, &headers);
    
    res = curl_easy_perform(curl);
    
    if (res == CURLE_OK && buffer.data) {
        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
        
        if (http_code == 200) {
            cJSON *json = cJSON_Parse(buffer.data);
            if (json) {
                cJSON *code = cJSON_GetObjectItemCaseSensitive(json, "code");
                cJSON *data = cJSON_GetObjectItemCaseSensitive(json, "data");
                
                if (code && cJSON_IsNumber(code) && code->valueint == 0 &&
                    data && cJSON_IsObject(data)) {
                    
                    cJSON *list = cJSON_GetObjectItemCaseSensitive(data, "list");
                    if (list && cJSON_IsArray(list)) {
                        int count = cJSON_GetArraySize(list);
                        if (count > 0) {
                            /* Get the last (most recent) candle */
                            cJSON *candle = cJSON_GetArrayItem(list, count - 1);
                            if (candle) {
                                /* close field contains market cap as string */
                                cJSON *close = cJSON_GetObjectItemCaseSensitive(candle, "close");
                                if (close && cJSON_IsString(close)) {
                                    double mc = atof(close->valuestring);
                                    /* Store as cents for precision */
                                    *market_cap = (uint64_t)(mc * 100.0);
                                    ret = 0;
                                } else if (close && cJSON_IsNumber(close)) {
                                    *market_cap = (uint64_t)(close->valuedouble * 100.0);
                                    ret = 0;
                                }
                            }
                        }
                    }
                }
                cJSON_Delete(json);
            }
        }
    }
    
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    free(buffer.data);
    
    return ret;
}

/**
 * @brief Fetch token info from GMGN API (combines KOL and market cap calls)
 */
static int fetch_token_info(const char *address, token_info_t *info) {
    int kol_result, mc_result;
    
    if (!address || !info) {
        return -1;
    }
    
    /* Fetch KOL count */
    kol_result = fetch_kol_count(address, &info->kol_count);
    
    /* Fetch market cap */
    mc_result = fetch_market_cap(address, &info->market_cap);
    
    /* Consider success if we got at least one piece of data */
    return (kol_result == 0 || mc_result == 0) ? 0 : -1;
}

/**
 * @brief Find free slot in tracker
 */
static int find_free_slot(token_tracker_t *tracker) {
    for (int i = 0; i < TRACKER_MAX_TOKENS; i++) {
        if (tracker->tokens[i].state == TOKEN_STATE_REMOVED ||
            tracker->tokens[i].address[0] == '\0') {
            return i;
        }
    }
    return -1;
}

/**
 * @brief Find token by address
 */
static int find_token(token_tracker_t *tracker, const char *address) {
    for (int i = 0; i < TRACKER_MAX_TOKENS; i++) {
        if (tracker->tokens[i].state == TOKEN_STATE_TRACKING &&
            strcmp(tracker->tokens[i].address, address) == 0) {
            return i;
        }
    }
    return -1;
}

/**
 * @brief Background tracking thread function
 */
static void *tracker_thread(void *arg) {
    token_tracker_t *tracker = (token_tracker_t *)arg;
    
    while (tracker->running) {
        time_t now = time(NULL);
        
        pthread_mutex_lock(&tracker->lock);
        
        int tracking_count = 0;
        int checked_count = 0;
        
        for (int i = 0; i < TRACKER_MAX_TOKENS; i++) {
            tracked_token_t *t = &tracker->tokens[i];
            
            if (t->state != TOKEN_STATE_TRACKING) {
                continue;
            }
            
            tracking_count++;
            
            /* Sanity check - skip if no address */
            if (t->address[0] == '\0') {
                continue;
            }
            
            /* Check if token has exceeded max age */
            uint32_t age = (uint32_t)(now - t->discovered_at);
            if (tracker->filter->max_age_seconds > 0 && 
                age > tracker->filter->max_age_seconds) {
                t->state = TOKEN_STATE_EXPIRED;
                tracker->expired_count++;
                if (tracker->active_count > 0) {
                    tracker->active_count--;
                }
                continue;
            }
            
            /* Check if it's time to re-check this token */
            if (now - t->last_check < TRACKER_CHECK_INTERVAL) {
                continue;
            }
            
            checked_count++;
            
            /* Fetch updated token info */
            token_info_t info = {0};
            memcpy(info.symbol, t->symbol, sizeof(info.symbol));
            memcpy(info.address, t->address, sizeof(info.address));
            info.age_seconds = age;
            
            pthread_mutex_unlock(&tracker->lock);
            
            int fetch_result = fetch_token_info(t->address, &info);
            
            pthread_mutex_lock(&tracker->lock);
            
            /* Token might have been removed while we were fetching */
            if (t->state != TOKEN_STATE_TRACKING) {
                continue;
            }
            
            t->last_check = now;
            t->check_count++;
            
            if (fetch_result == 0) {
                t->last_market_cap = info.market_cap;
                t->last_kol_count = info.kol_count;
                
                /* Check if token now passes filters */
                bool mc_pass = filter_check_market_cap(info.market_cap, 
                                                       tracker->filter);
                bool kol_pass = filter_check_kol(info.kol_count, 
                                                 tracker->filter);
                bool age_pass = filter_check_age(age, tracker->filter);
                
                if (mc_pass && kol_pass && age_pass) {
                    t->state = TOKEN_STATE_PASSED;
                    tracker->passed_count++;
                    if (tracker->active_count > 0) {
                        tracker->active_count--;
                    }
                    
                    /* Invoke callback */
                    if (tracker->on_token_passed) {
                        pthread_mutex_unlock(&tracker->lock);
                        tracker->on_token_passed(t, &info, 
                                                 tracker->callback_user_data);
                        pthread_mutex_lock(&tracker->lock);
                    }
                }
            }
        }
        
        pthread_mutex_unlock(&tracker->lock);
        
        /* Sleep before next round */
        usleep(500000); /* 0.5 seconds */
    }
    
    return NULL;
}

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
    
    return 0;
}

int tracker_start(token_tracker_t *tracker) {
    if (!tracker) {
        return -1;
    }
    
    tracker->running = true;
    
    if (pthread_create(&tracker->thread, NULL, tracker_thread, tracker) != 0) {
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
    curl_global_cleanup();
}

int tracker_add_token(token_tracker_t *tracker, const pool_data_t *pool) {
    if (!tracker || !pool) {
        return -1;
    }
    
    pthread_mutex_lock(&tracker->lock);
    
    /* Check if already tracking */
    if (find_token(tracker, pool->base_token.address) >= 0) {
        pthread_mutex_unlock(&tracker->lock);
        return 1;
    }
    
    /* Find free slot */
    int slot = find_free_slot(tracker);
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
    t->last_check = 0; /* Force immediate check */
    t->last_market_cap = pool->base_token.market_cap;
    t->last_kol_count = pool->base_token.kol_count;
    t->state = TOKEN_STATE_TRACKING;
    
    tracker->active_count++;
    
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
    
    token_info_t info = {0};
    
    if (fetch_token_info(address, &info) != 0) {
        return false;
    }
    
    bool mc_pass = filter_check_market_cap(info.market_cap, tracker->filter);
    bool kol_pass = filter_check_kol(info.kol_count, tracker->filter);
    
    return mc_pass && kol_pass;
}
