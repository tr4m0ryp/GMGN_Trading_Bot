/**
 * @file websocket_internal.h
 * @brief Internal shared state for WebSocket client module
 *
 * Exposes the ws_client struct and internal helpers shared across
 * the split websocket source files. Not part of the public API.
 *
 * Dependencies: libwebsockets, "websocket_client.h", "json_parser.h"
 *
 * @date 2025-12-20
 */

#ifndef WEBSOCKET_INTERNAL_H
#define WEBSOCKET_INTERNAL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <inttypes.h>

#include <libwebsockets.h>

#include "websocket_client.h"
#include "json_parser.h"

/* Buffer sizes */
#define WS_RX_BUFFER_SIZE       65536
#define WS_TX_BUFFER_SIZE       4096
#define WS_MAX_PENDING_MSGS     16

/**
 * @brief Internal client structure
 */
struct ws_client {
    /* libwebsockets context and connection */
    struct lws_context *context;
    struct lws *wsi;

    /* Connection parameters */
    char url[GMGN_MAX_URL_LEN];
    char host[128];
    char path[512];  /* Increased for query parameters */
    int port;
    int use_ssl;
    char access_token[GMGN_MAX_TOKEN_LEN];

    /* State */
    gmgn_conn_state_t state;
    time_t last_ping;
    time_t connected_at;

    /* Reconnection */
    uint32_t reconnect_attempts;
    uint32_t max_reconnect_attempts;
    uint32_t reconnect_delay_ms;
    uint32_t current_reconnect_delay;

    /* Statistics */
    uint64_t messages_received;
    uint64_t bytes_received;
    uint32_t reconnect_count;

    /* Message buffer */
    char rx_buffer[WS_RX_BUFFER_SIZE];
    size_t rx_buffer_len;

    /* Pending send messages */
    char *pending_msgs[WS_MAX_PENDING_MSGS];
    size_t pending_msg_lens[WS_MAX_PENDING_MSGS];
    int pending_msg_count;

    /* Callbacks */
    pool_callback_fn pool_callback;
    void *pool_callback_data;
    pair_update_callback_fn pair_update_callback;
    void *pair_update_callback_data;
    token_launch_callback_fn token_launch_callback;
    void *token_launch_callback_data;
    error_callback_fn error_callback;
    void *error_callback_data;
};

/* Flag to track if we've initialized lws logging */
extern int g_lws_log_initialized;

/* Protocol table used by lws */
extern const struct lws_protocols g_ws_protocols[];

/**
 * @brief Queue a message for sending via WebSocket
 *
 * @param client Client instance
 * @param msg Message string to send
 * @param len Length of message
 * @return 0 on success, -1 on error
 */
int ws_queue_message(ws_client_t *client, const char *msg, size_t len);

/**
 * @brief Process a received WebSocket message
 *
 * Parses message type and dispatches to appropriate callback.
 *
 * @param client Client instance
 * @param msg Received message string
 * @param len Length of message
 */
void ws_process_message(ws_client_t *client, const char *msg, size_t len);

/**
 * @brief Parse WebSocket URL into components
 *
 * @param url Full WebSocket URL
 * @param host Output host buffer
 * @param host_size Size of host buffer
 * @param port Output port
 * @param path Output path buffer
 * @param path_size Size of path buffer
 * @param use_ssl Output SSL flag
 * @return 0 on success, -1 on error
 */
int ws_parse_url(const char *url, char *host, size_t host_size,
                 int *port, char *path, size_t path_size, int *use_ssl);

#endif /* WEBSOCKET_INTERNAL_H */
