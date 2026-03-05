/**
 * @file api_fetchers.c
 * @brief CURL-based API fetchers for token tracker
 *
 * Implements persistent CURL handles and API call functions for
 * fetching KOL holder counts and market cap data from the GMGN API.
 *
 * Dependencies: curl, cJSON, "token_tracker_internal.h"
 *
 * @date 2025-12-20
 */

#include "token_tracker_internal.h"

/* Persistent CURL handles for connection reuse */
static CURL *s_curl_kol = NULL;
static CURL *s_curl_mcap = NULL;
static struct curl_slist *s_headers = NULL;

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

void tracker_init_curl_handles(void) {
    if (s_curl_kol || s_curl_mcap) return;

    char cookie_header[2048];
    snprintf(cookie_header, sizeof(cookie_header),
        "cf_clearance=%s; _ga=%s; _ga_0XM0LYXGC8=%s; __cf_bm=%s",
        getenv("GMGN_CF_CLEARANCE") ? getenv("GMGN_CF_CLEARANCE") : "",
        getenv("GMGN_GA") ? getenv("GMGN_GA") : "",
        getenv("GMGN_GA_SESSION") ? getenv("GMGN_GA_SESSION") : "",
        getenv("GMGN_CF_BM") ? getenv("GMGN_CF_BM") : "");

    /* Build shared headers once */
    s_headers = curl_slist_append(s_headers, "Accept: application/json");
    s_headers = curl_slist_append(s_headers, "Accept-Language: en-US,en;q=0.5");
    s_headers = curl_slist_append(s_headers, "Referer: https://gmgn.ai/");
    s_headers = curl_slist_append(s_headers, "Origin: https://gmgn.ai");
    s_headers = curl_slist_append(s_headers, "Sec-Fetch-Dest: empty");
    s_headers = curl_slist_append(s_headers, "Sec-Fetch-Mode: cors");
    s_headers = curl_slist_append(s_headers, "Sec-Fetch-Site: same-origin");

    /* Initialize KOL handle */
    s_curl_kol = curl_easy_init();
    if (s_curl_kol) {
        curl_easy_setopt(s_curl_kol, CURLOPT_TIMEOUT, TRACKER_API_TIMEOUT);
        curl_easy_setopt(s_curl_kol, CURLOPT_CONNECTTIMEOUT, 2L);
        curl_easy_setopt(s_curl_kol, CURLOPT_USERAGENT,
            "Mozilla/5.0 (X11; Linux x86_64; rv:145.0) Gecko/20100101 Firefox/145.0");
        curl_easy_setopt(s_curl_kol, CURLOPT_ACCEPT_ENCODING, "");
        curl_easy_setopt(s_curl_kol, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(s_curl_kol, CURLOPT_COOKIE, cookie_header);
        curl_easy_setopt(s_curl_kol, CURLOPT_HTTPHEADER, s_headers);
        curl_easy_setopt(s_curl_kol, CURLOPT_TCP_KEEPALIVE, 1L);
        curl_easy_setopt(s_curl_kol, CURLOPT_TCP_KEEPIDLE, 120L);
        curl_easy_setopt(s_curl_kol, CURLOPT_TCP_KEEPINTVL, 60L);
        curl_easy_setopt(s_curl_kol, CURLOPT_DNS_CACHE_TIMEOUT, 300L);
    }

    /* Initialize MCAP handle */
    s_curl_mcap = curl_easy_init();
    if (s_curl_mcap) {
        curl_easy_setopt(s_curl_mcap, CURLOPT_TIMEOUT, TRACKER_API_TIMEOUT);
        curl_easy_setopt(s_curl_mcap, CURLOPT_CONNECTTIMEOUT, 2L);
        curl_easy_setopt(s_curl_mcap, CURLOPT_USERAGENT,
            "Mozilla/5.0 (X11; Linux x86_64; rv:145.0) Gecko/20100101 Firefox/145.0");
        curl_easy_setopt(s_curl_mcap, CURLOPT_ACCEPT_ENCODING, "");
        curl_easy_setopt(s_curl_mcap, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(s_curl_mcap, CURLOPT_COOKIE, cookie_header);
        curl_easy_setopt(s_curl_mcap, CURLOPT_HTTPHEADER, s_headers);
        curl_easy_setopt(s_curl_mcap, CURLOPT_TCP_KEEPALIVE, 1L);
        curl_easy_setopt(s_curl_mcap, CURLOPT_TCP_KEEPIDLE, 120L);
        curl_easy_setopt(s_curl_mcap, CURLOPT_TCP_KEEPINTVL, 60L);
        curl_easy_setopt(s_curl_mcap, CURLOPT_DNS_CACHE_TIMEOUT, 300L);
    }
}

void tracker_cleanup_curl_handles(void) {
    if (s_curl_kol) {
        curl_easy_cleanup(s_curl_kol);
        s_curl_kol = NULL;
    }
    if (s_curl_mcap) {
        curl_easy_cleanup(s_curl_mcap);
        s_curl_mcap = NULL;
    }
    if (s_headers) {
        curl_slist_free_all(s_headers);
        s_headers = NULL;
    }
}

int tracker_fetch_kol_count(const char *address, uint8_t *kol_count) {
    curl_buffer_t buffer = {0};
    char url[768];

    if (!address || !kol_count) {
        return -1;
    }

    *kol_count = 0;

    if (!s_curl_kol) {
        return -1;
    }

    snprintf(url, sizeof(url),
        "%s%s?tag=renowned&limit=10&orderby=amount_percentage&direction=desc",
        GMGN_KOL_HOLDERS_API, address);

    buffer.data = malloc(1);
    buffer.size = 0;

    curl_easy_setopt(s_curl_kol, CURLOPT_URL, url);
    curl_easy_setopt(s_curl_kol, CURLOPT_WRITEFUNCTION, curl_write_cb);
    curl_easy_setopt(s_curl_kol, CURLOPT_WRITEDATA, &buffer);

    CURLcode res = curl_easy_perform(s_curl_kol);
    int ret = -1;

    if (res == CURLE_OK && buffer.data) {
        long http_code = 0;
        curl_easy_getinfo(s_curl_kol, CURLINFO_RESPONSE_CODE, &http_code);

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
                                            break;
                                        }
                                    }
                                }
                            }
                        }

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

    free(buffer.data);
    return ret;
}

int tracker_fetch_market_cap(const char *address, uint64_t *market_cap) {
    curl_buffer_t buffer = {0};
    char url[1024];

    if (!address || !market_cap) {
        return -1;
    }

    *market_cap = 0;

    if (!s_curl_mcap) {
        return -1;
    }

    snprintf(url, sizeof(url),
        "%s%s?pool_type=tpool&resolution=1s&limit=5",
        GMGN_MCAP_CANDLES_API, address);

    buffer.data = malloc(1);
    buffer.size = 0;

    curl_easy_setopt(s_curl_mcap, CURLOPT_URL, url);
    curl_easy_setopt(s_curl_mcap, CURLOPT_WRITEFUNCTION, curl_write_cb);
    curl_easy_setopt(s_curl_mcap, CURLOPT_WRITEDATA, &buffer);

    CURLcode res = curl_easy_perform(s_curl_mcap);
    int ret = -1;

    if (res == CURLE_OK && buffer.data) {
        long http_code = 0;
        curl_easy_getinfo(s_curl_mcap, CURLINFO_RESPONSE_CODE, &http_code);

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
                            cJSON *candle = cJSON_GetArrayItem(list, count - 1);
                            if (candle) {
                                cJSON *close_val = cJSON_GetObjectItemCaseSensitive(candle, "close");
                                if (close_val && cJSON_IsString(close_val)) {
                                    double mc = atof(close_val->valuestring);
                                    *market_cap = (uint64_t)(mc * 100.0);
                                    ret = 0;
                                } else if (close_val && cJSON_IsNumber(close_val)) {
                                    *market_cap = (uint64_t)(close_val->valuedouble * 100.0);
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

    free(buffer.data);
    return ret;
}

int tracker_fetch_token_info(const char *address, token_info_t *info) {
    int kol_result, mc_result;

    if (!address || !info) {
        return -1;
    }

    kol_result = tracker_fetch_kol_count(address, &info->kol_count);
    mc_result = tracker_fetch_market_cap(address, &info->market_cap);

    return (kol_result == 0 || mc_result == 0) ? 0 : -1;
}
