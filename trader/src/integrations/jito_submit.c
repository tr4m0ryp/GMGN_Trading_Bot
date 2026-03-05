/**
 * @file jito_submit.c
 * @brief Jito bundle submission via JSON-RPC
 *
 * Handles initialization, tip account selection, and bundle submission
 * to the Jito block engine. Uses CURL for HTTP POST requests with
 * JSON-RPC payloads.
 *
 * Dependencies: <curl/curl.h>, <cjson/cJSON.h>, "jito_client_internal.h"
 *
 * @date 2026-03-05
 */

/* Disable truncation warnings */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <curl/curl.h>
#include <cjson/cJSON.h>

#include "jito_client.h"
#include "jito_client_internal.h"

/* Tip accounts from Jito documentation */
const char *JITO_TIP_ACCOUNTS[JITO_TIP_ACCOUNT_COUNT] = {
    "96gYZGLnJYVFmbjzopPSU6QiEV5fGqZNyN9nmNhvrZU5",
    "HFqU5x63VTqvQss8hp11i4wVV8bD44PvwucfZ2bU7gRe",
    "Cw8CFyM9FkoMi7K7Crf6HNQqf4uEMzpKw6QNghXLvLkY",
    "ADaUMid9yfUytqMBgopwjb2DTLSokTSzL1zt6iGPaS49",
    "DfXygSm4jCyNCybVYYK6DwvWqjKee8pbDmJGcLWNDXjh",
    "ADuUkR4vqLUMWXxW9gh6D6L8pMSawimctcNZ5pGwDcEt",
    "DttWaMuVvTiduZRnguLF7jNxTgiMBZ1hyAumKUiL2KRL",
    "3AVi9Tg9Uo68tJfuvoKvqKNWKkC5wPdSSdeBnizKZ6jT"
};

size_t jito_curl_write_cb(void *contents, size_t size, size_t nmemb, void *userp) {
    size_t realsize = size * nmemb;
    jito_curl_buffer_t *buf = (jito_curl_buffer_t *)userp;

    char *ptr = realloc(buf->data, buf->size + realsize + 1);
    if (!ptr) return 0;

    buf->data = ptr;
    memcpy(&buf->data[buf->size], contents, realsize);
    buf->size += realsize;
    buf->data[buf->size] = '\0';

    return realsize;
}

int64_t jito_get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (int64_t)tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

int jito_init(jito_client_t *client, const char *endpoint, uint64_t tip_lamports) {
    if (!client || !endpoint) {
        return -1;
    }

    memset(client, 0, sizeof(jito_client_t));

    /* Build full endpoint URL */
    snprintf(client->endpoint, sizeof(client->endpoint), "%s%s", endpoint, JITO_API_PATH);

    /* Validate tip amount */
    if (tip_lamports < JITO_MIN_TIP) {
        tip_lamports = JITO_DEFAULT_TIP;
    }
    if (tip_lamports > JITO_MAX_TIP) {
        tip_lamports = JITO_MAX_TIP;
    }

    client->tip_lamports = tip_lamports;
    client->current_tip_account = 0;
    client->initialized = true;

    /* Initialize random seed for tip account selection */
    srand((unsigned int)time(NULL));

    return 0;
}

const char *jito_get_tip_account(jito_client_t *client) {
    if (!client) {
        return JITO_TIP_ACCOUNTS[0];
    }

    /* Round-robin with some randomization */
    int idx = (client->current_tip_account + rand()) % JITO_TIP_ACCOUNT_COUNT;
    client->current_tip_account = (client->current_tip_account + 1) % JITO_TIP_ACCOUNT_COUNT;

    return JITO_TIP_ACCOUNTS[idx];
}

int jito_submit_bundle(jito_client_t *client, const char **signed_txs, int tx_count,
                       jito_result_t *result) {
    CURL *curl;
    CURLcode res;
    struct curl_slist *headers = NULL;
    jito_curl_buffer_t response = {0};
    cJSON *root = NULL;
    cJSON *params = NULL;
    cJSON *tx_array = NULL;
    char *json_str = NULL;
    int ret = -1;
    int64_t start_time;

    if (!client || !client->initialized || !signed_txs || tx_count <= 0 || !result) {
        if (result) {
            result->success = false;
            snprintf(result->error, sizeof(result->error), "Invalid parameters");
        }
        return -1;
    }

    if (tx_count > JITO_MAX_BUNDLE_SIZE) {
        result->success = false;
        snprintf(result->error, sizeof(result->error),
                 "Too many transactions: %d (max %d)", tx_count, JITO_MAX_BUNDLE_SIZE);
        return -1;
    }

    memset(result, 0, sizeof(jito_result_t));
    start_time = jito_get_time_ms();

    /* Build JSON-RPC request */
    root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "jsonrpc", "2.0");
    cJSON_AddNumberToObject(root, "id", 1);
    cJSON_AddStringToObject(root, "method", "sendBundle");

    /* Build transactions array */
    tx_array = cJSON_CreateArray();
    for (int i = 0; i < tx_count; i++) {
        cJSON_AddItemToArray(tx_array, cJSON_CreateString(signed_txs[i]));
    }

    params = cJSON_CreateArray();
    cJSON_AddItemToArray(params, tx_array);
    cJSON_AddItemToObject(root, "params", params);

    json_str = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);

    if (!json_str) {
        result->success = false;
        snprintf(result->error, sizeof(result->error), "Failed to create JSON");
        return -1;
    }

    /* Initialize CURL */
    curl = curl_easy_init();
    if (!curl) {
        free(json_str);
        result->success = false;
        snprintf(result->error, sizeof(result->error), "CURL init failed");
        return -1;
    }

    /* Setup headers */
    headers = curl_slist_append(headers, "Content-Type: application/json");

    /* Response buffer */
    response.data = malloc(1);
    response.size = 0;

    /* Configure request */
    curl_easy_setopt(curl, CURLOPT_URL, client->endpoint);
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_str);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, jito_curl_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 5L);

    /* Execute request */
    res = curl_easy_perform(curl);

    result->submit_time_ms = jito_get_time_ms() - start_time;

    if (res != CURLE_OK) {
        result->success = false;
        snprintf(result->error, sizeof(result->error), "CURL error: %s",
                 curl_easy_strerror(res));
        goto cleanup;
    }

    /* Check HTTP status */
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

    if (http_code != 200) {
        result->success = false;
        snprintf(result->error, sizeof(result->error), "HTTP %ld: %s",
                 http_code, response.data ? response.data : "no response");
        goto cleanup;
    }

    /* Parse response */
    if (response.data && response.size > 0) {
        cJSON *resp = cJSON_Parse(response.data);
        if (resp) {
            cJSON *result_obj = cJSON_GetObjectItemCaseSensitive(resp, "result");
            cJSON *error_obj = cJSON_GetObjectItemCaseSensitive(resp, "error");

            if (error_obj) {
                cJSON *msg = cJSON_GetObjectItemCaseSensitive(error_obj, "message");
                result->success = false;
                snprintf(result->error, sizeof(result->error), "%s",
                         msg ? msg->valuestring : "Unknown error");
            } else if (result_obj && cJSON_IsString(result_obj)) {
                result->success = true;
                snprintf(result->bundle_id, sizeof(result->bundle_id), "%s",
                         result_obj->valuestring);
                ret = 0;
            } else {
                result->success = false;
                snprintf(result->error, sizeof(result->error), "Invalid response format");
            }

            cJSON_Delete(resp);
        } else {
            result->success = false;
            snprintf(result->error, sizeof(result->error), "JSON parse error");
        }
    }

cleanup:
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    free(json_str);
    free(response.data);

    return ret;
}

int jito_submit_tx(jito_client_t *client, const char *signed_tx, jito_result_t *result) {
    if (!signed_tx) {
        if (result) {
            result->success = false;
            snprintf(result->error, sizeof(result->error), "No transaction provided");
        }
        return -1;
    }

    const char *txs[] = { signed_tx };
    return jito_submit_bundle(client, txs, 1, result);
}
