/**
 * @file ws_connection.c
 * @brief WebSocket connection and disconnection logic
 *
 * Handles establishing and tearing down the libwebsockets connection
 * to the GMGN WebSocket server, including SSL configuration and
 * URL parsing.
 *
 * Dependencies: libwebsockets, "websocket_internal.h"
 *
 * @date 2025-12-20
 */

#include "websocket_internal.h"

/* Shared lws log initialization flag */
int g_lws_log_initialized = 0;

/**
 * @brief Parse URL into host, port, path components
 */
int ws_parse_url(const char *url, char *host, size_t host_size,
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

int ws_client_connect(ws_client_t *client) {
    if (!client) {
        return -1;
    }

    /* Suppress noisy libwebsockets internal logging (only show errors) */
    if (!g_lws_log_initialized) {
        const char *debug = getenv("GMGN_LWS_DEBUG");
        if (debug && debug[0] == '1') {
            /* Full debug logging if GMGN_LWS_DEBUG=1 */
            lws_set_log_level(LLL_ERR | LLL_WARN | LLL_NOTICE | LLL_INFO, NULL);
        } else {
            /* Only show critical errors */
            lws_set_log_level(LLL_ERR, NULL);
        }
        g_lws_log_initialized = 1;
    }

    /* Create context if needed */
    if (!client->context) {
        struct lws_context_creation_info info;
        memset(&info, 0, sizeof(info));

        info.port = CONTEXT_PORT_NO_LISTEN;
        info.protocols = g_ws_protocols;
        info.options = LWS_SERVER_OPTION_DO_SSL_GLOBAL_INIT;
        info.user = client;
        /* GMGN server sends large HTTP headers - need big buffers */
        info.pt_serv_buf_size = 32768;
        info.max_http_header_data = 16384;
        info.max_http_header_pool = 4;

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
    conn_info.protocol = g_ws_protocols[0].name;
    conn_info.userdata = client;  /* Pass client directly */

    if (client->use_ssl) {
        conn_info.ssl_connection = LCCSCF_USE_SSL |
                                   LCCSCF_ALLOW_SELFSIGNED |
                                   LCCSCF_SKIP_SERVER_CERT_HOSTNAME_CHECK |
                                   LCCSCF_ALLOW_INSECURE;
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
