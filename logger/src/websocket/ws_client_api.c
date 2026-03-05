/**
 * @file ws_client_api.c
 * @brief WebSocket client public API: create, destroy, subscribe, stats
 *
 * Implements the public-facing functions for creating and destroying
 * the WebSocket client, subscribing to channels, sending pings,
 * and retrieving connection statistics.
 *
 * Dependencies: libwebsockets, "websocket_internal.h"
 *
 * @date 2025-12-20
 */

#include "websocket_internal.h"

ws_client_t *ws_client_create(const char *url, const char *access_token) {
    if (!url) {
        return NULL;
    }

    ws_client_t *client = calloc(1, sizeof(ws_client_t));
    if (!client) {
        return NULL;
    }

    /* Parse URL */
    char base_path[128];
    if (ws_parse_url(url, client->host, sizeof(client->host),
                     &client->port, base_path, sizeof(base_path),
                     &client->use_ssl) != 0) {
        free(client);
        return NULL;
    }

    /* Generate random device/client IDs */
    char device_id[37];
    char client_id[24];
    char fp_did[33];
    char uuid_str[33];

    /* Simple random hex generator for IDs */
    srand((unsigned int)time(NULL));
    snprintf(device_id, sizeof(device_id),
             "%08x-%04x-%04x-%04x-%08x%04x",
             rand(), rand() & 0xffff, rand() & 0xffff,
             rand() & 0xffff, rand(), rand() & 0xffff);
    snprintf(client_id, sizeof(client_id), "gmgn_c_%08x", rand());
    snprintf(fp_did, sizeof(fp_did), "%08x%08x%08x%08x",
             rand(), rand(), rand(), rand());
    snprintf(uuid_str, sizeof(uuid_str), "%08x%08x%08x%08x",
             rand(), rand(), rand(), rand());

    /* Build path with query parameters */
    snprintf(client->path, sizeof(client->path),
             "%s?device_id=%s&client_id=%s&from_app=gmgn"
             "&app_ver=20250729-1647-ffac485&tz_name=UTC&tz_offset=0"
             "&app_lang=en-US&fp_did=%s&os=linux&uuid=%s",
             base_path, device_id, client_id, fp_did, uuid_str);

    strncpy(client->url, url, sizeof(client->url) - 1);

    if (access_token) {
        strncpy(client->access_token, access_token,
                sizeof(client->access_token) - 1);
    }

    /* Set defaults */
    client->state = GMGN_STATE_DISCONNECTED;
    client->max_reconnect_attempts = 10;
    client->reconnect_delay_ms = 1000;
    client->current_reconnect_delay = client->reconnect_delay_ms;

    return client;
}

void ws_client_destroy(ws_client_t *client) {
    if (!client) {
        return;
    }

    /* Disconnect if connected */
    ws_client_disconnect(client);

    /* Free pending messages */
    for (int i = 0; i < client->pending_msg_count; i++) {
        free(client->pending_msgs[i]);
    }

    /* Destroy context */
    if (client->context) {
        lws_context_destroy(client->context);
    }

    free(client);
}

int ws_client_subscribe(ws_client_t *client, const char *channel,
                        const char *chain) {
    if (!client || !channel) {
        return -1;
    }

    char buffer[WS_TX_BUFFER_SIZE];
    int len = json_create_subscribe_msg(channel, chain, buffer, sizeof(buffer));

    if (len < 0) {
        return -1;
    }

    return ws_queue_message(client, buffer, (size_t)len);
}

int ws_client_unsubscribe(ws_client_t *client, const char *channel) {
    if (!client || !channel) {
        return -1;
    }

    char buffer[WS_TX_BUFFER_SIZE];
    int len = json_create_unsubscribe_msg(channel, buffer, sizeof(buffer));

    if (len < 0) {
        return -1;
    }

    return ws_queue_message(client, buffer, (size_t)len);
}

int ws_client_service(ws_client_t *client, int timeout_ms) {
    if (!client || !client->context) {
        return -1;
    }

    return lws_service(client->context, timeout_ms);
}

gmgn_conn_state_t ws_client_get_state(const ws_client_t *client) {
    return client ? client->state : GMGN_STATE_DISCONNECTED;
}

void ws_client_set_pool_callback(ws_client_t *client, pool_callback_fn callback,
                                  void *user_data) {
    if (client) {
        client->pool_callback = callback;
        client->pool_callback_data = user_data;
    }
}

void ws_client_set_error_callback(ws_client_t *client, error_callback_fn callback,
                                   void *user_data) {
    if (client) {
        client->error_callback = callback;
        client->error_callback_data = user_data;
    }
}

void ws_client_set_pair_update_callback(ws_client_t *client,
                                         pair_update_callback_fn callback,
                                         void *user_data) {
    if (client) {
        client->pair_update_callback = callback;
        client->pair_update_callback_data = user_data;
    }
}

void ws_client_set_token_launch_callback(ws_client_t *client,
                                          token_launch_callback_fn callback,
                                          void *user_data) {
    if (client) {
        client->token_launch_callback = callback;
        client->token_launch_callback_data = user_data;
    }
}

int ws_client_ping(ws_client_t *client) {
    if (!client) {
        return -1;
    }

    char buffer[WS_TX_BUFFER_SIZE];
    int len = json_create_ping_msg(buffer, sizeof(buffer));

    if (len < 0) {
        return -1;
    }

    client->last_ping = time(NULL);
    return ws_queue_message(client, buffer, (size_t)len);
}

void ws_client_get_stats(const ws_client_t *client, uint64_t *messages_received,
                         uint64_t *bytes_received, uint32_t *reconnect_count) {
    if (!client) {
        return;
    }

    if (messages_received) {
        *messages_received = client->messages_received;
    }
    if (bytes_received) {
        *bytes_received = client->bytes_received;
    }
    if (reconnect_count) {
        *reconnect_count = client->reconnect_count;
    }
}
