/**
 * @file chart_fetcher.c
 * @brief Implementation of real-time chart data fetcher
 *
 * Fetches OHLCV candle data from GMGN API using CURL.
 * Optimized for 1-second polling with persistent connections.
 *
 * Dependencies: <curl/curl.h>, <cjson/cJSON.h>
 *
 * @date 2025-12-24
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <curl/curl.h>
#include <cjson/cJSON.h>

#include "chart_fetcher.h"

/* CURL response buffer */
typedef struct {
    char *data;
    size_t size;
} curl_buffer_t;

/* Global CURL handle (persistent connection) */
static CURL *g_curl = NULL;
static struct curl_slist *g_headers = NULL;
static bool g_initialized = false;

/**
 * @brief CURL write callback
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

int chart_fetcher_init(const trader_config_t *config) {
    char cookie_header[2048];

    if (g_initialized) {
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
    g_headers = curl_slist_append(g_headers, "Accept: application/json");
    g_headers = curl_slist_append(g_headers, "Accept-Language: en-US,en;q=0.5");
    g_headers = curl_slist_append(g_headers, "Referer: https://gmgn.ai/");
    g_headers = curl_slist_append(g_headers, "Origin: https://gmgn.ai");
    g_headers = curl_slist_append(g_headers, "Sec-Fetch-Dest: empty");
    g_headers = curl_slist_append(g_headers, "Sec-Fetch-Mode: cors");
    g_headers = curl_slist_append(g_headers, "Sec-Fetch-Site: same-origin");

    /* Create CURL handle */
    g_curl = curl_easy_init();
    if (!g_curl) {
        curl_slist_free_all(g_headers);
        g_headers = NULL;
        return -1;
    }

    /* Configure for fast trading - 2 second timeout */
    curl_easy_setopt(g_curl, CURLOPT_TIMEOUT, CHART_API_TIMEOUT_SEC);
    curl_easy_setopt(g_curl, CURLOPT_CONNECTTIMEOUT, 2L);
    curl_easy_setopt(g_curl, CURLOPT_USERAGENT,
        "Mozilla/5.0 (X11; Linux x86_64; rv:145.0) Gecko/20100101 Firefox/145.0");
    curl_easy_setopt(g_curl, CURLOPT_ACCEPT_ENCODING, "");
    curl_easy_setopt(g_curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(g_curl, CURLOPT_COOKIE, cookie_header);
    curl_easy_setopt(g_curl, CURLOPT_HTTPHEADER, g_headers);
    curl_easy_setopt(g_curl, CURLOPT_TCP_KEEPALIVE, 1L);
    curl_easy_setopt(g_curl, CURLOPT_DNS_CACHE_TIMEOUT, 300L);
    curl_easy_setopt(g_curl, CURLOPT_TCP_NODELAY, 1L);  /* Disable Nagle for speed */

    g_initialized = true;
    return 0;
}

void chart_buffer_init(chart_buffer_t *buffer, const char *address,
                       const char *symbol) {
    if (!buffer) {
        return;
    }

    memset(buffer, 0, sizeof(chart_buffer_t));

    if (address) {
        strncpy(buffer->token_address, address, sizeof(buffer->token_address) - 1);
    }
    if (symbol) {
        strncpy(buffer->token_symbol, symbol, sizeof(buffer->token_symbol) - 1);
    }
}

/**
 * @brief Parse JSON candle array into buffer
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
    curl_buffer_t curl_buf = {0};
    CURLcode res;
    long http_code = 0;

    if (!g_initialized || !address || !buffer) {
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
    curl_easy_setopt(g_curl, CURLOPT_URL, url);
    curl_easy_setopt(g_curl, CURLOPT_WRITEFUNCTION, curl_write_cb);
    curl_easy_setopt(g_curl, CURLOPT_WRITEDATA, &curl_buf);

    /* Execute request */
    res = curl_easy_perform(g_curl);

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
    curl_easy_getinfo(g_curl, CURLINFO_RESPONSE_CODE, &http_code);
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

const candle_t *chart_get_latest(const chart_buffer_t *buffer) {
    if (!buffer || buffer->candle_count == 0) {
        return NULL;
    }
    return &buffer->candles[buffer->candle_count - 1];
}

const candle_t *chart_get_candle(const chart_buffer_t *buffer, int index) {
    if (!buffer || index < 0 || index >= buffer->candle_count) {
        return NULL;
    }
    return &buffer->candles[index];
}

double chart_get_price(const chart_buffer_t *buffer) {
    const candle_t *latest = chart_get_latest(buffer);
    return latest ? latest->close : 0.0;
}

double chart_get_price_change(const chart_buffer_t *buffer, int lookback) {
    if (!buffer || buffer->candle_count < 2) {
        return 0.0;
    }

    int start_idx = buffer->candle_count - 1 - lookback;
    if (start_idx < 0) {
        start_idx = 0;
    }

    double first_price = buffer->candles[start_idx].close;
    double last_price = buffer->candles[buffer->candle_count - 1].close;

    if (first_price <= 0.0) {
        return 0.0;
    }

    return (last_price - first_price) / first_price;
}

int chart_extract_features(const chart_buffer_t *buffer, float *features,
                           int max_features, int *out_length) {
    if (!buffer || !features || !out_length) {
        return -1;
    }

    int num_candles = buffer->candle_count;
    int features_per_candle = 5;  /* OHLCV */
    int total_features = num_candles * features_per_candle;

    if (total_features > max_features) {
        total_features = max_features;
        num_candles = max_features / features_per_candle;
    }

    /* Normalize prices relative to first candle */
    double base_price = buffer->candles[0].close;
    if (base_price <= 0.0) {
        base_price = 1.0;
    }

    int feat_idx = 0;
    for (int i = 0; i < num_candles && feat_idx < max_features; i++) {
        const candle_t *c = &buffer->candles[i];

        /* Normalize to base price (log returns would be better for ONNX) */
        features[feat_idx++] = (float)(c->open / base_price);
        features[feat_idx++] = (float)(c->high / base_price);
        features[feat_idx++] = (float)(c->low / base_price);
        features[feat_idx++] = (float)(c->close / base_price);
        features[feat_idx++] = (float)(c->volume);  /* Volume as-is for now */
    }

    *out_length = feat_idx;
    return 0;
}

void chart_fetcher_cleanup(void) {
    if (!g_initialized) {
        return;
    }

    if (g_curl) {
        curl_easy_cleanup(g_curl);
        g_curl = NULL;
    }

    if (g_headers) {
        curl_slist_free_all(g_headers);
        g_headers = NULL;
    }

    curl_global_cleanup();
    g_initialized = false;
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
