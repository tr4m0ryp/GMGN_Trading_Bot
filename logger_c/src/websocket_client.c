/**
 * @file websocket_client.c
 * @brief WebSocket client implementation for GMGN connection
 *
 * Implements WebSocket connection management using libwebsockets.
 * Handles connection, reconnection, subscription, and message routing.
 *
 * Dependencies: libwebsockets, openssl, "websocket_client.h", "json_parser.h"
 *
 * @date 2025-12-20
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <libwebsockets.h>

#include "websocket_client.h"
#include "json_parser.h"

/* Buffer sizes */
#define WS_RX_BUFFER_SIZE       65536
#define WS_TX_BUFFER_SIZE       4096

/* Flag to track if we've initialized lws logging */
static int s_lws_log_initialized = 0;
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

/* Forward declarations for lws callbacks */
static int ws_callback(struct lws *wsi, enum lws_callback_reasons reason,
                       void *user, void *in, size_t len);

static const struct lws_protocols s_protocols[] = {
    {
        "gmgn-protocol",
        ws_callback,
        sizeof(ws_client_t *),
        WS_RX_BUFFER_SIZE,
        0, NULL, 0
    },
    { NULL, NULL, 0, 0, 0, NULL, 0 }
};

/**
 * @brief Parse URL into host, port, path components
 */
static int parse_url(const char *url, char *host, size_t host_size,
                     int *port, char *path, size_t path_size, int *use_ssl) {
    const char *p = url;
    
    /* Check scheme */
    if (strncmp(p, "wss://", 6) == 0) {
        *use_ssl = 1;
        *port = 443;
        p += 6;
    } else if (strncmp(p, "ws://", 5) == 0) {
        *use_ssl = 0;
        *port = 80;
        p += 5;
    } else {
        return -1;
    }
    
    /* Extract host */
    const char *path_start = strchr(p, '/');
    const char *port_start = strchr(p, ':');
    
    size_t host_len;
    if (port_start && (!path_start || port_start < path_start)) {
        host_len = (size_t)(port_start - p);
        *port = atoi(port_start + 1);
    } else if (path_start) {
        host_len = (size_t)(path_start - p);
    } else {
        host_len = strlen(p);
    }
    
    if (host_len >= host_size) {
        return -1;
    }
    
    memcpy(host, p, host_len);
    host[host_len] = '\0';
    
    /* Extract path */
    if (path_start) {
        size_t path_len = strlen(path_start);
        if (path_len >= path_size) {
            path_len = path_size - 1;
        }
        memcpy(path, path_start, path_len);
        path[path_len] = '\0';
    } else {
        path[0] = '/';
        path[1] = '\0';
    }
    
    return 0;
}

/**
 * @brief Queue a message for sending
 */
static int queue_message(ws_client_t *client, const char *msg, size_t len) {
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
static void process_message(ws_client_t *client, const char *msg, size_t len) {
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

            fprintf(debug_log, "[%s] WS Message #%lu (%zu bytes): %.300s%s\n",
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
            /* Pair update: price/volume updates for trading pairs */
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
            /* Token launch: new token launch notifications */
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
            /* Heartbeat response, connection is alive */
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
            /* Unknown or unhandled message type */
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
                    process_message(client, client->rx_buffer, client->rx_buffer_len);
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
    if (parse_url(url, client->host, sizeof(client->host),
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

int ws_client_connect(ws_client_t *client) {
    if (!client) {
        return -1;
    }
    
    /* Suppress noisy libwebsockets internal logging (only show errors) */
    if (!s_lws_log_initialized) {
        const char *debug = getenv("GMGN_LWS_DEBUG");
        if (debug && debug[0] == '1') {
            /* Full debug logging if GMGN_LWS_DEBUG=1 */
            lws_set_log_level(LLL_ERR | LLL_WARN | LLL_NOTICE | LLL_INFO, NULL);
        } else {
            /* Only show critical errors */
            lws_set_log_level(LLL_ERR, NULL);
        }
        s_lws_log_initialized = 1;
    }
    
    /* Create context if needed */
    if (!client->context) {
        struct lws_context_creation_info info;
        memset(&info, 0, sizeof(info));
        
        info.port = CONTEXT_PORT_NO_LISTEN;
        info.protocols = s_protocols;
        info.options = LWS_SERVER_OPTION_DO_SSL_GLOBAL_INIT;
        info.user = client;
        /* GMGN server sends large HTTP headers - need big buffers */
        info.pt_serv_buf_size = 32768;           /* Per-thread service buffer */
        info.max_http_header_data = 16384;       /* HTTP header parser buffer */
        info.max_http_header_pool = 4;           /* HTTP header pool size */
        
        client->context = lws_create_context(&info);
        if (!client->context) {
            return -1;
        }
    }
    
    /* Connect */
    struct lws_client_connect_info conn_info;
    memset(&conn_info, 0, sizeof(conn_info));
    
    /* Build origin URL */
    char origin_url[256];
    snprintf(origin_url, sizeof(origin_url), "https://%s", client->host);
    
    conn_info.context = client->context;
    conn_info.address = client->host;
    conn_info.port = client->port;
    conn_info.path = client->path;
    conn_info.host = client->host;
    conn_info.origin = origin_url;
    conn_info.protocol = s_protocols[0].name;
    conn_info.userdata = client;  /* Pass client directly */
    
    if (client->use_ssl) {
        conn_info.ssl_connection = LCCSCF_USE_SSL | 
                                   LCCSCF_ALLOW_SELFSIGNED |
                                   LCCSCF_SKIP_SERVER_CERT_HOSTNAME_CHECK;
    }
    
    client->state = GMGN_STATE_CONNECTING;
    client->wsi = lws_client_connect_via_info(&conn_info);
    
    if (!client->wsi) {
        client->state = GMGN_STATE_ERROR;
        return -1;
    }
    
    return 0;
}

int ws_client_disconnect(ws_client_t *client) {
    if (!client || !client->wsi) {
        return 0;
    }
    
    /* Close connection */
    lws_close_reason(client->wsi, LWS_CLOSE_STATUS_NORMAL, NULL, 0);
    client->state = GMGN_STATE_DISCONNECTED;
    client->wsi = NULL;
    
    return 0;
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
    
    return queue_message(client, buffer, (size_t)len);
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
    
    return queue_message(client, buffer, (size_t)len);
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
    return queue_message(client, buffer, (size_t)len);
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
