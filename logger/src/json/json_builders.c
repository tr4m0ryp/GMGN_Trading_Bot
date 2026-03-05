/**
 * @file json_builders.c
 * @brief JSON message builders for subscribe, unsubscribe, and ping
 *
 * Implements functions that create JSON messages for WebSocket
 * communication with the GMGN API.
 *
 * Dependencies: cJSON, "json_parser_internal.h"
 *
 * @date 2025-12-20
 */

#include "json_parser_internal.h"

int json_create_subscribe_msg(const char *channel, const char *chain,
                              char *buffer, size_t buffer_size) {
    if (!channel || !buffer || buffer_size == 0) {
        return -1;
    }

    cJSON *msg = cJSON_CreateObject();
    if (!msg) {
        return -1;
    }

    /* Required subscription fields per GMGN API */
    cJSON_AddStringToObject(msg, "action", "subscribe");
    cJSON_AddStringToObject(msg, "channel", channel);
    cJSON_AddStringToObject(msg, "f", "w");

    /* Generate a simple unique ID */
    static int msg_counter = 0;
    char id_buf[32];
    snprintf(id_buf, sizeof(id_buf), "gmgn_%08x", ++msg_counter);
    cJSON_AddStringToObject(msg, "id", id_buf);

    /* Create data array with chain object: [{"chain": "sol"}] */
    if (chain) {
        cJSON *data_array = cJSON_CreateArray();
        if (data_array) {
            cJSON *chain_obj = cJSON_CreateObject();
            if (chain_obj) {
                cJSON_AddStringToObject(chain_obj, "chain", chain);
                cJSON_AddItemToArray(data_array, chain_obj);
            }
            cJSON_AddItemToObject(msg, "data", data_array);
        }
    }

    char *json_str = cJSON_PrintUnformatted(msg);
    cJSON_Delete(msg);

    if (!json_str) {
        return -1;
    }

    size_t len = strlen(json_str);
    if (len >= buffer_size) {
        free(json_str);
        return -1;
    }

    strcpy(buffer, json_str);
    free(json_str);

    return (int)len;
}

int json_create_unsubscribe_msg(const char *channel, char *buffer,
                                size_t buffer_size) {
    if (!channel || !buffer || buffer_size == 0) {
        return -1;
    }

    cJSON *msg = cJSON_CreateObject();
    if (!msg) {
        return -1;
    }

    cJSON_AddStringToObject(msg, "action", "unsubscribe");
    cJSON_AddStringToObject(msg, "channel", channel);

    char *json_str = cJSON_PrintUnformatted(msg);
    cJSON_Delete(msg);

    if (!json_str) {
        return -1;
    }

    size_t len = strlen(json_str);
    if (len >= buffer_size) {
        free(json_str);
        return -1;
    }

    strcpy(buffer, json_str);
    free(json_str);

    return (int)len;
}

int json_create_ping_msg(char *buffer, size_t buffer_size) {
    if (!buffer || buffer_size == 0) {
        return -1;
    }

    cJSON *msg = cJSON_CreateObject();
    if (!msg) {
        return -1;
    }

    cJSON_AddStringToObject(msg, "type", "ping");

    char *json_str = cJSON_PrintUnformatted(msg);
    cJSON_Delete(msg);

    if (!json_str) {
        return -1;
    }

    size_t len = strlen(json_str);
    if (len >= buffer_size) {
        free(json_str);
        return -1;
    }

    strcpy(buffer, json_str);
    free(json_str);

    return (int)len;
}
