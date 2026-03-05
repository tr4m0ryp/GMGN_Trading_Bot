/**
 * @file jito_status.c
 * @brief Jito bundle status checking and utilities
 *
 * Handles querying bundle status from Jito block engine via JSON-RPC
 * getBundleStatuses method. Also provides cleanup and status string
 * conversion utilities.
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
#include <curl/curl.h>
#include <cjson/cJSON.h>

#include "jito_client.h"
#include "jito_client_internal.h"

int jito_get_bundle_status(jito_client_t *client, const char *bundle_id,
                           jito_bundle_status_t *status) {
    CURL *curl;
    CURLcode res;
    struct curl_slist *headers = NULL;
    jito_curl_buffer_t response = {0};
    cJSON *root = NULL;
    cJSON *params = NULL;
    cJSON *id_array = NULL;
    char *json_str = NULL;
    int ret = -1;

    if (!client || !client->initialized || !bundle_id || !status) {
        return -1;
    }

    *status = JITO_STATUS_UNKNOWN;

    /* Build JSON-RPC request */
    root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "jsonrpc", "2.0");
    cJSON_AddNumberToObject(root, "id", 1);
    cJSON_AddStringToObject(root, "method", "getBundleStatuses");

    id_array = cJSON_CreateArray();
    cJSON_AddItemToArray(id_array, cJSON_CreateString(bundle_id));

    params = cJSON_CreateArray();
    cJSON_AddItemToArray(params, id_array);
    cJSON_AddItemToObject(root, "params", params);

    json_str = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);

    if (!json_str) return -1;

    /* Execute request */
    curl = curl_easy_init();
    if (!curl) {
        free(json_str);
        return -1;
    }

    headers = curl_slist_append(headers, "Content-Type: application/json");
    response.data = malloc(1);
    response.size = 0;

    curl_easy_setopt(curl, CURLOPT_URL, client->endpoint);
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_str);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, jito_curl_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);

    res = curl_easy_perform(curl);

    if (res == CURLE_OK && response.data) {
        cJSON *resp = cJSON_Parse(response.data);
        if (resp) {
            cJSON *result_obj = cJSON_GetObjectItemCaseSensitive(resp, "result");
            if (result_obj) {
                cJSON *value = cJSON_GetObjectItemCaseSensitive(result_obj, "value");
                if (value && cJSON_IsArray(value) && cJSON_GetArraySize(value) > 0) {
                    cJSON *first = cJSON_GetArrayItem(value, 0);
                    if (first) {
                        cJSON *confirmation_status = cJSON_GetObjectItemCaseSensitive(
                            first, "confirmation_status");
                        if (confirmation_status && cJSON_IsString(confirmation_status)) {
                            const char *s = confirmation_status->valuestring;
                            if (strcmp(s, "confirmed") == 0 ||
                                strcmp(s, "finalized") == 0) {
                                *status = JITO_STATUS_LANDED;
                            } else if (strcmp(s, "pending") == 0) {
                                *status = JITO_STATUS_PENDING;
                            } else {
                                *status = JITO_STATUS_FAILED;
                            }
                            ret = 0;
                        }
                    }
                }
            }
            cJSON_Delete(resp);
        }
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    free(json_str);
    free(response.data);

    return ret;
}

void jito_cleanup(jito_client_t *client) {
    if (client) {
        memset(client, 0, sizeof(jito_client_t));
    }
}

const char *jito_status_str(jito_bundle_status_t status) {
    switch (status) {
        case JITO_STATUS_PENDING: return "pending";
        case JITO_STATUS_LANDED:  return "landed";
        case JITO_STATUS_FAILED:  return "failed";
        case JITO_STATUS_DROPPED: return "dropped";
        default:                  return "unknown";
    }
}
