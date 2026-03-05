/**
 * @file curl_utils.c
 * @brief CURL initialization and cleanup helpers for chart API
 *
 * Manages a persistent CURL handle with pre-configured headers and cookies
 * for making requests to the GMGN chart API. The handle is reused across
 * requests for connection pooling and DNS caching.
 *
 * Dependencies: <curl/curl.h>, "ai_data_collector_internal.h"
 *
 * @date 2025-12-20
 */

#include "ai_data_collector_internal.h"

/* Global CURL handle for chart API (persistent connection) */
static CURL *g_curl_chart = NULL;
static struct curl_slist *g_headers_chart = NULL;

/**
 * @brief CURL write callback for API responses
 */
size_t curl_write_cb(void *contents, size_t size, size_t nmemb, void *userp) {
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
void init_curl_handle(void) {
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
void cleanup_curl_handle(void) {
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
 * @brief Get the global CURL handle for chart API requests
 *
 * @return Pointer to the global CURL handle, or NULL if not initialized
 */
CURL *get_curl_chart_handle(void) {
    return g_curl_chart;
}
