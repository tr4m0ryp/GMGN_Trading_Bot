/**
 * @file chart_fetcher.c
 * @brief Fetch chart data from the GMGN API
 *
 * Performs HTTP requests to retrieve token chart candle data using the
 * persistent CURL handle managed by curl_utils.c.
 *
 * Dependencies: <curl/curl.h>, "ai_data_collector_internal.h"
 *
 * @date 2025-12-20
 */

#include "ai_data_collector_internal.h"

/**
 * @brief Fetch chart data from GMGN API
 *
 * @param address Token address
 * @param out_data Output buffer for JSON data (caller must free)
 * @param out_len Output length of data
 *
 * @return 0 on success, -1 on failure
 */
int fetch_chart_data(const char *address, char **out_data, size_t *out_len) {
    char url[1024];
    curl_buffer_t buffer = {0};
    CURLcode res;
    CURL *curl = get_curl_chart_handle();

    if (!curl || !address || !out_data || !out_len) {
        return -1;
    }

    snprintf(url, sizeof(url),
        "%s%s?pool_type=tpool&resolution=1s&limit=501",
        AI_MCAP_CANDLES_API, address);

    buffer.data = malloc(1);
    buffer.size = 0;

    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buffer);

    res = curl_easy_perform(curl);

    if (res == CURLE_OK && buffer.data && buffer.size > 0) {
        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

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
