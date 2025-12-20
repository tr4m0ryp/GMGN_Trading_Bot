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

/* GMGN API endpoint for token info */
#define GMGN_TOKEN_API "https://gmgn.ai/api/v1/token_info/sol/"

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
 * @brief Fetch token info from GMGN API
 */
static int fetch_token_info(const char *address, token_info_t *info) {
    CURL *curl;
    CURLcode res;
    curl_buffer_t buffer = {0};
    char url[512];
    char cookie_header[2048];
    int ret = -1;
    
    if (!address || !info) {
        return -1;
    }
    
    snprintf(url, sizeof(url), "%s%s", GMGN_TOKEN_API, address);
    
    curl = curl_easy_init();
    if (!curl) {
        return -1;
    }
    
    buffer.data = malloc(1);
    buffer.size = 0;
    
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buffer);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, TRACKER_API_TIMEOUT);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, 
        "Mozilla/5.0 (X11; Linux x86_64; rv:145.0) Gecko/20100101 Firefox/145.0");
    curl_easy_setopt(curl, CURLOPT_ACCEPT_ENCODING, ""); /* Let curl handle decompression */
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    
    /* Set Cloudflare session cookies - CRITICAL FOR API ACCESS */
    snprintf(cookie_header, sizeof(cookie_header),
        "cf_clearance=%s; _ga=%s; _ga_0XM0LYXGC8=%s; __cf_bm=%s",
        getenv("GMGN_CF_CLEARANCE") ? getenv("GMGN_CF_CLEARANCE") : "",
        getenv("GMGN_GA") ? getenv("GMGN_GA") : "GA1.1.1216464152.1766234082",
        getenv("GMGN_GA_SESSION") ? getenv("GMGN_GA_SESSION") : "GS1.1.1766242908.4.1.1766245615.56.0.0",
        getenv("GMGN_CF_BM") ? getenv("GMGN_CF_BM") : "");
    curl_easy_setopt(curl, CURLOPT_COOKIE, cookie_header);
    
    /* Add headers to look like browser */
    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Accept: application/json");
    headers = curl_slist_append(headers, "Accept-Language: en-US,en;q=0.5");
    headers = curl_slist_append(headers, "Referer: https://gmgn.ai/");
    headers = curl_slist_append(headers, "Origin: https://gmgn.ai");
    headers = curl_slist_append(headers, "Sec-Fetch-Dest: empty");
    headers = curl_slist_append(headers, "Sec-Fetch-Mode: cors");
    headers = curl_slist_append(headers, "Sec-Fetch-Site: same-origin");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    
    res = curl_easy_perform(curl);
    
    if (res == CURLE_OK && buffer.data) {
        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
        
        if (http_code == 200) {
            /* Parse JSON response */
            cJSON *json = cJSON_Parse(buffer.data);
            if (json) {
                cJSON *data = cJSON_GetObjectItemCaseSensitive(json, "data");
                if (data && cJSON_IsObject(data)) {
                    /* Debug: print response for first few chars */
                    char *json_str = cJSON_Print(data);
                    if (json_str) {
                        printf("[API] Response data (500 chars): %.500s\n", json_str);
                        fflush(stdout);
                        free(json_str);
                    }
                    
                    /* 
                     * API returns fields directly in data, not nested in "token".
                     * Look for market_cap, kol/smart_degen, holder_count directly.
                     */
                    cJSON *mc = cJSON_GetObjectItemCaseSensitive(data, "market_cap");
                    if (!mc) {
                        mc = cJSON_GetObjectItemCaseSensitive(data, "usd_market_cap");
                    }
                    if (mc && cJSON_IsNumber(mc)) {
                        info->market_cap = (uint64_t)(mc->valuedouble * 100.0);
                    }
                    
                    cJSON *kol = cJSON_GetObjectItemCaseSensitive(data, "smart_degen");
                    if (!kol) {
                        kol = cJSON_GetObjectItemCaseSensitive(data, "kol_count");
                    }
                    if (kol && cJSON_IsNumber(kol)) {
                        info->kol_count = (uint8_t)kol->valueint;
                    }
                    
                    cJSON *holders = cJSON_GetObjectItemCaseSensitive(data, "holder_count");
                    if (holders && cJSON_IsNumber(holders)) {
                        info->holder_count = (uint32_t)holders->valueint;
                    }
                    
                    printf("[API] Token %s: MC=$%.2fK, KOL=%d, holders=%d\n",
                           address,
                           info->market_cap / 100000.0,  /* Convert cents to thousands */
                           info->kol_count,
                           info->holder_count);
                    fflush(stdout);
                    
                    /* Consider success if we got at least market cap */
                    if (info->market_cap > 0 || info->holder_count > 0) {
                        ret = 0;
                    } else {
                        /* Check if there's any numeric field we got */
                        ret = 0; /* Still consider it success */
                    }
                } else {
                    printf("[API] Token %s: No 'data' object in response\n", address);
                    fflush(stdout);
                }
                cJSON_Delete(json);
            } else {
                /* Debug: API returned 200 but invalid JSON */
                printf("[API] Token %s: HTTP %ld but parse failed\n", 
                        address, http_code);
                /* Print first 500 chars of response for debugging */
                printf("[API] Response: %.500s\n", buffer.data ? buffer.data : "(null)");
                fflush(stdout);
            }
        }
    } else {
        /* Debug: API call failed */
        if (res != CURLE_OK) {
            printf("[API] Token %s: CURL error: %s\n", 
                    address, curl_easy_strerror(res));
        } else {
            long http_code = 0;
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
            printf("[API] Token %s: HTTP %ld\n", address, http_code);
        }
        fflush(stdout);
    }
    
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    free(buffer.data);
    
    return ret;
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
    
    printf("[TRACKER] Thread started\n");
    fflush(stdout);
    
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
            printf("[TRACKER] Checking token %s (age=%us, last_check=%lds ago)\n",
                    t->symbol, age, (long)(now - t->last_check));
            fflush(stdout);
            
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
