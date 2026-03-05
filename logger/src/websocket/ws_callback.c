/**
 * @file ws_callback.c
 * @brief libwebsockets callback handler
 *
 * Implements the lws protocol callback that handles connection
 * lifecycle events, message reception, and write scheduling.
 *
 * Dependencies: libwebsockets, "websocket_internal.h"
 *
 * @date 2025-12-20
 */

#include "websocket_internal.h"

/**
 * @brief libwebsockets callback handler
 */
static int ws_callback(struct lws *wsi, enum lws_callback_reasons reason,
                       void *user, void *in, size_t len) {
    /* Get client from user data directly */
    ws_client_t *client = (ws_client_t *)user;

    /* Also try to get from context if user is NULL */
    if (!client && wsi) {
        struct lws_context *ctx = lws_get_context(wsi);
        if (ctx) {
            client = (ws_client_t *)lws_context_user(ctx);
        }
    }

    switch (reason) {
        case LWS_CALLBACK_CLIENT_ESTABLISHED:
            if (client) {
                client->state = GMGN_STATE_CONNECTED;
                client->connected_at = time(NULL);
                client->last_ping = time(NULL);
                client->reconnect_attempts = 0;
                client->current_reconnect_delay = client->reconnect_delay_ms;
            }
            break;

        case LWS_CALLBACK_CLIENT_RECEIVE:
            if (client && in && len > 0) {
                /* Accumulate fragmented messages */
                if (client->rx_buffer_len + len < WS_RX_BUFFER_SIZE) {
                    memcpy(client->rx_buffer + client->rx_buffer_len, in, len);
                    client->rx_buffer_len += len;
                }

                /* Check if message is complete */
                if (lws_is_final_fragment(wsi)) {
                    client->rx_buffer[client->rx_buffer_len] = '\0';
                    ws_process_message(client, client->rx_buffer, client->rx_buffer_len);
                    client->rx_buffer_len = 0;
                }
            }
            break;

        case LWS_CALLBACK_CLIENT_WRITEABLE: {
            if (client && client->pending_msg_count > 0) {
                /* Send first pending message */
                char *buf = client->pending_msgs[0];
                size_t msg_len = client->pending_msg_lens[0];

                int written = lws_write(wsi, (unsigned char *)(buf + LWS_PRE),
                                        msg_len, LWS_WRITE_TEXT);

                if (written < 0 || (size_t)written < msg_len) {
                    return -1;
                }

                /* Remove from queue */
                free(buf);
                client->pending_msg_count--;
                memmove(client->pending_msgs, client->pending_msgs + 1,
                        client->pending_msg_count * sizeof(char *));
                memmove(client->pending_msg_lens, client->pending_msg_lens + 1,
                        client->pending_msg_count * sizeof(size_t));

                /* More messages to send? */
                if (client->pending_msg_count > 0) {
                    lws_callback_on_writable(wsi);
                }
            }
            break;
        }

        case LWS_CALLBACK_CLIENT_CONNECTION_ERROR:
            if (client) {
                client->state = GMGN_STATE_ERROR;
                client->wsi = NULL;

                if (client->error_callback) {
                    const char *err = in ? (const char *)in : "Connection error";
                    client->error_callback(-1, err, client->error_callback_data);
                }
            }
            break;

        case LWS_CALLBACK_CLIENT_CLOSED:
            if (client) {
                client->state = GMGN_STATE_DISCONNECTED;
                client->wsi = NULL;
            }
            break;

        case LWS_CALLBACK_CLIENT_APPEND_HANDSHAKE_HEADER: {
            /* Add custom headers for GMGN compatibility */
            unsigned char **p = (unsigned char **)in;
            unsigned char *end = (*p) + len;

            /* Add User-Agent header */
            const char *ua = "User-Agent: Mozilla/5.0 (X11; Linux x86_64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/120.0.0.0 Safari/537.36\r\n";
            size_t ua_len = strlen(ua);

            if (end - (*p) >= (long)ua_len) {
                memcpy(*p, ua, ua_len);
                *p += ua_len;
            }
            break;
        }

        default:
            break;
    }

    return 0;
}

/* Protocol table definition */
const struct lws_protocols g_ws_protocols[] = {
    {
        "gmgn-protocol",
        ws_callback,
        sizeof(ws_client_t *),
        WS_RX_BUFFER_SIZE,
        0, NULL, 0
    },
    { NULL, NULL, 0, 0, 0, NULL, 0 }
};
