/**
 * @file chart_fetch.c
 * @brief Chart fetcher network operations and JSON parsing
 *
 * Handles CURL initialization, HTTP requests to GMGN API, and JSON parsing
 * of candle data. Manages the persistent CURL connection for 1s polling.
 *
 * @date 2026-03-05
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <curl/curl.h>
#include <cjson/cJSON.h>

#include "chart_fetcher.h"
#include "chart_fetcher_internal.h"

/* Global CURL handle (persistent connection) - defined here, declared extern in internal header */
CURL *g_chart_curl = NULL;
struct curl_slist *g_chart_headers = NULL;
bool g_chart_initialized = false;

size_t chart_curl_write_cb(void *contents, size_t size, size_t nmemb, void *userp) {
    size_t realsize = size * nmemb;
    chart_curl_buffer_t *buf = (chart_curl_buffer_t *)userp;

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

int chart_fetcher_init(const trader_config_t *config) {
    char cookie_header[2048];

    if (g_chart_initialized) {
        return 0;
    }

    /* Initialize CURL */
    curl_global_init(CURL_GLOBAL_DEFAULT);

    /* Build cookie string */
    snprintf(cookie_header, sizeof(cookie_header),
        "cf_clearance=%s; _ga=%s; _ga_0XM0LYXGC8=%s; __cf_bm=%s",
        config ? config->gmgn_cf_clearance : "",
        config ? config->gmgn_ga : "",
        config ? config->gmgn_ga_session : "",
        config ? config->gmgn_cf_bm : "");

    /* Setup headers */
    g_chart_headers = curl_slist_append(g_chart_headers, "Accept: application/json");
    g_chart_headers = curl_slist_append(g_chart_headers, "Accept-Language: en-US,en;q=0.5");
    g_chart_headers = curl_slist_append(g_chart_headers, "Referer: https://gmgn.ai/");
    g_chart_headers = curl_slist_append(g_chart_headers, "Origin: https://gmgn.ai");
    g_chart_headers = curl_slist_append(g_chart_headers, "Sec-Fetch-Dest: empty");
    g_chart_headers = curl_slist_append(g_chart_headers, "Sec-Fetch-Mode: cors");
    g_chart_headers = curl_slist_append(g_chart_headers, "Sec-Fetch-Site: same-origin");

    /* Create CURL handle */
    g_chart_curl = curl_easy_init();
    if (!g_chart_curl) {
        curl_slist_free_all(g_chart_headers);
        g_chart_headers = NULL;
        return -1;
    }

    /* Configure for fast trading - 2 second timeout */
    curl_easy_setopt(g_chart_curl, CURLOPT_TIMEOUT, CHART_API_TIMEOUT_SEC);
    curl_easy_setopt(g_chart_curl, CURLOPT_CONNECTTIMEOUT, 2L);
    curl_easy_setopt(g_chart_curl, CURLOPT_USERAGENT,
        "Mozilla/5.0 (X11; Linux x86_64; rv:145.0) Gecko/20100101 Firefox/145.0");
    curl_easy_setopt(g_chart_curl, CURLOPT_ACCEPT_ENCODING, "");
    curl_easy_setopt(g_chart_curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(g_chart_curl, CURLOPT_COOKIE, cookie_header);
    curl_easy_setopt(g_chart_curl, CURLOPT_HTTPHEADER, g_chart_headers);
    curl_easy_setopt(g_chart_curl, CURLOPT_TCP_KEEPALIVE, 1L);
    curl_easy_setopt(g_chart_curl, CURLOPT_DNS_CACHE_TIMEOUT, 300L);
    curl_easy_setopt(g_chart_curl, CURLOPT_TCP_NODELAY, 1L);  /* Disable Nagle for speed */

    g_chart_initialized = true;
    return 0;
}

/**
 * @brief Parse JSON candle array into buffer
 *
 * Navigates the GMGN API response structure to extract OHLCV candle data.
 * Handles both string and numeric value formats from the API.
 *
 * @param json_data  Raw JSON response string
 * @param buffer     Chart buffer to populate with parsed candles
 *
 * @return 0 on success, -1 on parse failure or empty data
 */
static int parse_candles(const char *json_data, chart_buffer_t *buffer) {
    cJSON *root = NULL;
    cJSON *data = NULL;
    cJSON *list = NULL;
    cJSON *candle = NULL;
    int count = 0;

    root = cJSON_Parse(json_data);
    if (!root) {
        return -1;
    }

    /* Check response code */
    cJSON *code = cJSON_GetObjectItemCaseSensitive(root, "code");
    if (!code || code->valueint != 0) {
        cJSON_Delete(root);
        return -1;
    }

    /* Navigate to data.list */
    data = cJSON_GetObjectItemCaseSensitive(root, "data");
    if (!data || !cJSON_IsObject(data)) {
        cJSON_Delete(root);
        return -1;
    }

    list = cJSON_GetObjectItemCaseSensitive(data, "list");
    if (!list || !cJSON_IsArray(list)) {
        cJSON_Delete(root);
        return -1;
    }

    count = cJSON_GetArraySize(list);
    if (count == 0) {
        cJSON_Delete(root);
        return -1;
    }

    /* Clear existing candles */
    buffer->candle_count = 0;

    /* Parse candles (oldest to newest) */
    int idx = 0;
    cJSON_ArrayForEach(candle, list) {
        if (idx >= CHART_MAX_CANDLES) {
            break;
        }

        candle_t *c = &buffer->candles[idx];

        /* Parse timestamp (API returns milliseconds) */
        cJSON *time_val = cJSON_GetObjectItemCaseSensitive(candle, "time");
        if (time_val && cJSON_IsNumber(time_val)) {
            c->timestamp = (time_t)(time_val->valuedouble / 1000.0);
        }

        /* Parse OHLCV - handle both string and number formats */
        cJSON *val;

        val = cJSON_GetObjectItemCaseSensitive(candle, "open");
        if (val) {
            c->open = cJSON_IsString(val) ? atof(val->valuestring) : val->valuedouble;
        }

        val = cJSON_GetObjectItemCaseSensitive(candle, "high");
        if (val) {
            c->high = cJSON_IsString(val) ? atof(val->valuestring) : val->valuedouble;
        }

        val = cJSON_GetObjectItemCaseSensitive(candle, "low");
        if (val) {
            c->low = cJSON_IsString(val) ? atof(val->valuestring) : val->valuedouble;
        }

        val = cJSON_GetObjectItemCaseSensitive(candle, "close");
        if (val) {
            c->close = cJSON_IsString(val) ? atof(val->valuestring) : val->valuedouble;
        }

        val = cJSON_GetObjectItemCaseSensitive(candle, "volume");
        if (val) {
            c->volume = cJSON_IsString(val) ? atof(val->valuestring) : val->valuedouble;
        }

        idx++;
    }

    buffer->candle_count = idx;

    if (idx > 0) {
        buffer->first_candle_time = buffer->candles[0].timestamp;
        buffer->last_candle_time = buffer->candles[idx - 1].timestamp;
    }

    cJSON_Delete(root);
    return 0;
}

fetch_status_t chart_fetch(const char *address, chart_buffer_t *buffer) {
    char url[1024];
    chart_curl_buffer_t curl_buf = {0};
    CURLcode res;
    long http_code = 0;

    if (!g_chart_initialized || !address || !buffer) {
        return FETCH_ERROR_CURL;
    }

    /* Build URL */
    snprintf(url, sizeof(url),
        "%s%s?pool_type=tpool&resolution=1s&limit=%d",
        CHART_API_BASE, address, CHART_MAX_CANDLES);

    /* Allocate response buffer */
    curl_buf.data = malloc(1);
    curl_buf.size = 0;

    /* Configure request */
    curl_easy_setopt(g_chart_curl, CURLOPT_URL, url);
    curl_easy_setopt(g_chart_curl, CURLOPT_WRITEFUNCTION, chart_curl_write_cb);
    curl_easy_setopt(g_chart_curl, CURLOPT_WRITEDATA, &curl_buf);

    /* Execute request */
    res = curl_easy_perform(g_chart_curl);

    buffer->last_fetch_time = time(NULL);

    /* Check CURL result */
    if (res != CURLE_OK) {
        free(curl_buf.data);
        buffer->fetch_failures++;

        if (res == CURLE_OPERATION_TIMEDOUT) {
            return FETCH_ERROR_TIMEOUT;
        }
        return FETCH_ERROR_CURL;
    }

    /* Check HTTP code */
    curl_easy_getinfo(g_chart_curl, CURLINFO_RESPONSE_CODE, &http_code);
    if (http_code != 200) {
        free(curl_buf.data);
        buffer->fetch_failures++;
        return FETCH_ERROR_HTTP;
    }

    /* Parse JSON response */
    if (parse_candles(curl_buf.data, buffer) != 0) {
        free(curl_buf.data);
        buffer->fetch_failures++;
        return FETCH_ERROR_PARSE;
    }

    free(curl_buf.data);

    if (buffer->candle_count == 0) {
        buffer->fetch_failures++;
        return FETCH_ERROR_EMPTY;
    }

    buffer->fetch_failures = 0;
    return FETCH_SUCCESS;
}

void chart_fetcher_cleanup(void) {
    if (!g_chart_initialized) {
        return;
    }

    if (g_chart_curl) {
        curl_easy_cleanup(g_chart_curl);
        g_chart_curl = NULL;
    }

    if (g_chart_headers) {
        curl_slist_free_all(g_chart_headers);
        g_chart_headers = NULL;
    }

    curl_global_cleanup();
    g_chart_initialized = false;
}

const char *fetch_status_str(fetch_status_t status) {
    switch (status) {
        case FETCH_SUCCESS:       return "success";
        case FETCH_ERROR_CURL:    return "curl_error";
        case FETCH_ERROR_HTTP:    return "http_error";
        case FETCH_ERROR_PARSE:   return "parse_error";
        case FETCH_ERROR_EMPTY:   return "empty_data";
        case FETCH_ERROR_TIMEOUT: return "timeout";
        default:                  return "unknown";
    }
}
