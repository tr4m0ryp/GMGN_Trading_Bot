/**
 * @file ws_messaging.c
 * @brief WebSocket message queuing and processing
 *
 * Handles queuing outbound messages with LWS_PRE padding and
 * processing inbound messages by dispatching to registered callbacks.
 *
 * Dependencies: libwebsockets, "websocket_internal.h"
 *
 * @date 2025-12-20
 */

#include "websocket_internal.h"

/**
 * @brief Queue a message for sending
 */
int ws_queue_message(ws_client_t *client, const char *msg, size_t len) {
    if (!client || !msg || client->pending_msg_count >= WS_MAX_PENDING_MSGS) {
        return -1;
    }

    /* Allocate with LWS_PRE padding */
    char *buf = malloc(LWS_PRE + len + 1);
    if (!buf) {
        return -1;
    }

    memcpy(buf + LWS_PRE, msg, len);
    buf[LWS_PRE + len] = '\0';

    client->pending_msgs[client->pending_msg_count] = buf;
    client->pending_msg_lens[client->pending_msg_count] = len;
    client->pending_msg_count++;

    /* Request write callback */
    if (client->wsi) {
        lws_callback_on_writable(client->wsi);
    }

    return 0;
}

/**
 * @brief Process received message
 */
void ws_process_message(ws_client_t *client, const char *msg, size_t len) {
    if (!client || !msg || len == 0) {
        return;
    }

    client->messages_received++;
    client->bytes_received += len;

    /* Debug: log all messages in verbose mode to dedicated debug file */
    const char *verbose = getenv("GMGN_DEBUG");
    if (verbose && verbose[0] == '1') {
        FILE *debug_log = fopen("/tmp/gmgn_debug.log", "a");
        if (debug_log) {
            time_t now = time(NULL);
            struct tm *tm_info = localtime(&now);
            char timestamp[32];
            strftime(timestamp, sizeof(timestamp), "%H:%M:%S", tm_info);

            fprintf(debug_log, "[%s] WS Message #%" PRIu64 " (%zu bytes): %.300s%s\n",
                    timestamp, client->messages_received, len, msg, len > 300 ? "..." : "");
            fclose(debug_log);
        }
    }

    /* Parse message type */
    gmgn_msg_type_t msg_type = json_parse_message_type(msg, len);

    switch (msg_type) {
        case GMGN_MSG_NEW_POOL: {
            pool_data_t pools[8];
            int count = json_parse_new_pools(msg, len, pools, 8);

            if (verbose && verbose[0] == '1') {
                FILE *debug_log = fopen("/tmp/gmgn_debug.log", "a");
                if (debug_log) {
                    fprintf(debug_log, "[DEBUG] Parsed %d pool(s) from NEW_POOL message\n", count);
                    fclose(debug_log);
                }
            }

            if (count > 0 && client->pool_callback) {
                for (int i = 0; i < count; i++) {
                    client->pool_callback(&pools[i], client->pool_callback_data);
                }
            }
            break;
        }

        case GMGN_MSG_PAIR_UPDATE: {
            pool_data_t pools[8];
            int count = json_parse_new_pools(msg, len, pools, 8);

            if (verbose && verbose[0] == '1') {
                FILE *debug_log = fopen("/tmp/gmgn_debug.log", "a");
                if (debug_log) {
                    fprintf(debug_log, "[DEBUG] Parsed %d pair update(s) from PAIR_UPDATE message\n", count);
                    fclose(debug_log);
                }
            }

            if (count > 0 && client->pair_update_callback) {
                for (int i = 0; i < count; i++) {
                    client->pair_update_callback(&pools[i], client->pair_update_callback_data);
                }
            }
            break;
        }

        case GMGN_MSG_TOKEN_LAUNCH: {
            pool_data_t pools[8];
            int count = json_parse_new_pools(msg, len, pools, 8);

            if (verbose && verbose[0] == '1') {
                FILE *debug_log = fopen("/tmp/gmgn_debug.log", "a");
                if (debug_log) {
                    fprintf(debug_log, "[DEBUG] Parsed %d token launch(es) from TOKEN_LAUNCH message\n", count);
                    fclose(debug_log);
                }
            }

            if (count > 0 && client->token_launch_callback) {
                for (int i = 0; i < count; i++) {
                    client->token_launch_callback(&pools[i], client->token_launch_callback_data);
                }
            }
            break;
        }

        case GMGN_MSG_PONG:
            if (verbose && verbose[0] == '1') {
                FILE *debug_log = fopen("/tmp/gmgn_debug.log", "a");
                if (debug_log) {
                    fprintf(debug_log, "[DEBUG] Received PONG\n");
                    fclose(debug_log);
                }
            }
            break;

        case GMGN_MSG_ERROR:
            if (verbose && verbose[0] == '1') {
                FILE *debug_log = fopen("/tmp/gmgn_debug.log", "a");
                if (debug_log) {
                    fprintf(debug_log, "[ERROR] WebSocket error message: %s\n", msg);
                    fclose(debug_log);
                }
            }
            if (client->error_callback) {
                client->error_callback(-1, msg, client->error_callback_data);
            }
            break;

        default:
            if (verbose && verbose[0] == '1') {
                FILE *debug_log = fopen("/tmp/gmgn_debug.log", "a");
                if (debug_log) {
                    fprintf(debug_log, "[DEBUG] Unknown message type, first 200 chars: %.200s\n", msg);
                    fclose(debug_log);
                }
            }
            break;
    }
}
