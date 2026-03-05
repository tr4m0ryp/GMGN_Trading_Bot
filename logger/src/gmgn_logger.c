/**
 * @file gmgn_logger.c
 * @brief GMGN Trenches token logger main application
 *
 * Entry point for the GMGN token logger. Initializes configuration,
 * sets up the WebSocket client and token tracker, then runs the
 * main event loop. Config parsing and callbacks are split into
 * logger/logger_config.c and logger/logger_callbacks.c respectively.
 *
 * Dependencies: libwebsockets, cJSON, openssl, pthread
 *
 * Usage: gmgn_logger [options]
 *   -c, --chain CHAIN      Target chain (default: sol)
 *   -v, --verbose          Increase verbosity
 *   -q, --quiet            Minimal output
 *   -h, --help             Show help message
 *
 * @date 2025-12-20
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>
#include <inttypes.h>

#include "gmgn_types.h"
#include "websocket_client.h"
#include "json_parser.h"
#include "filter.h"
#include "output.h"
#include "token_tracker.h"

/* Global state - shared with logger_callbacks.c */
volatile sig_atomic_t g_running = 1;
ws_client_t *g_client = NULL;
token_tracker_t *g_tracker = NULL;
time_t g_start_time = 0;
uint64_t g_tokens_seen = 0;
uint64_t g_tokens_passed = 0;
uint64_t g_pair_updates = 0;
uint64_t g_token_launches = 0;

/* External functions from logger_config.c */
extern void logger_init_config(app_config_t *config);
extern int logger_parse_args(int argc, char *argv[], app_config_t *config);

/* External callbacks from logger_callbacks.c */
extern void logger_on_token_passed(const tracked_token_t *, const token_info_t *, void *);
extern void logger_on_new_pool(const pool_data_t *, void *);
extern void logger_on_pair_update(const pool_data_t *, void *);
extern void logger_on_token_launch(const pool_data_t *, void *);
extern void logger_on_error(int, const char *, void *);

/**
 * @brief Signal handler for graceful shutdown
 */
static void signal_handler(int signum) {
    (void)signum;
    g_running = 0;
    output_print_info("Shutdown signal received, exiting...");
}

/**
 * @brief Print periodic statistics
 */
static void print_periodic_stats(void) {
    if (g_client) {
        uint64_t messages = 0;
        uint64_t bytes = 0;
        uint32_t reconnects = 0;

        ws_client_get_stats(g_client, &messages, &bytes, &reconnects);

        uint32_t uptime = (uint32_t)(time(NULL) - g_start_time);

        uint32_t tracking = 0, passed = 0, expired = 0;
        if (g_tracker) {
            tracker_get_stats(g_tracker, &tracking, &passed, &expired);
        }

        output_print_stats(g_tokens_seen, g_tokens_passed, messages, uptime);
        printf("  Tracking: %u | Expired: %u | Pair Updates: %" PRIu64 " | Token Launches: %" PRIu64 "\n",
               tracking, expired, g_pair_updates, g_token_launches);
    }
}

/**
 * @brief Main application entry point
 */
int main(int argc, char *argv[]) {
    app_config_t config;
    int ret = EXIT_SUCCESS;
    time_t last_ping = 0;
    time_t last_stats = 0;
    gmgn_conn_state_t last_state = GMGN_STATE_DISCONNECTED;

    setvbuf(stdout, NULL, _IOLBF, 0);

    logger_init_config(&config);

    int parse_result = logger_parse_args(argc, argv, &config);
    if (parse_result != 0) {
        return (parse_result > 0) ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    signal(SIGPIPE, SIG_IGN);

    output_init((output_verbosity_t)config.verbosity);
    output_print_banner(&config);
    output_print_filter_config(&config.filter);

    /* Initialize token tracker */
    g_tracker = malloc(sizeof(token_tracker_t));
    if (!g_tracker) {
        output_print_error(-1, "Failed to allocate token tracker");
        return EXIT_FAILURE;
    }

    if (tracker_init(g_tracker, &config.filter) != 0) {
        output_print_error(-1, "Failed to initialize token tracker");
        free(g_tracker);
        return EXIT_FAILURE;
    }

    tracker_set_callback(g_tracker, logger_on_token_passed, NULL);

    if (tracker_start(g_tracker) != 0) {
        output_print_error(-1, "Failed to start token tracker");
        tracker_cleanup(g_tracker);
        free(g_tracker);
        return EXIT_FAILURE;
    }

    output_print_info("Token tracker started (checking every 100ms for up to 10 min)");

    /* Create WebSocket client */
    g_client = ws_client_create(config.websocket_url,
                                 config.access_token[0] ? config.access_token : NULL);
    if (!g_client) {
        output_print_error(-1, "Failed to create WebSocket client");
        return EXIT_FAILURE;
    }

    ws_client_set_pool_callback(g_client, logger_on_new_pool, &config.filter);
    ws_client_set_pair_update_callback(g_client, logger_on_pair_update, &config.filter);
    ws_client_set_token_launch_callback(g_client, logger_on_token_launch, &config.filter);
    ws_client_set_error_callback(g_client, logger_on_error, NULL);

    output_print_connection_status(GMGN_STATE_CONNECTING, config.websocket_url);

    if (ws_client_connect(g_client) != 0) {
        output_print_error(-1, "Failed to initiate connection");
        ret = EXIT_FAILURE;
        goto cleanup;
    }

    g_start_time = time(NULL);
    last_ping = g_start_time;
    last_stats = g_start_time;

    /* Main event loop */
    while (g_running) {
        int events = ws_client_service(g_client, 10);
        if (events < 0) {
            output_print_error(-1, "WebSocket service error");
            sleep(1);
            if (ws_client_connect(g_client) != 0) {
                output_print_error(-1, "Reconnection failed");
                ret = EXIT_FAILURE;
                break;
            }
            continue;
        }

        gmgn_conn_state_t state = ws_client_get_state(g_client);
        if (state != last_state) {
            output_print_connection_status(state, config.websocket_url);

            if (state == GMGN_STATE_CONNECTED && last_state == GMGN_STATE_CONNECTING) {
                output_print_info("Subscribing to WebSocket channels...");

                if (ws_client_subscribe(g_client, "new_pool_info", config.chain) != 0) {
                    output_print_error(-1, "Failed to subscribe to new_pool_info channel");
                }
                if (ws_client_subscribe(g_client, "new_pair_update", config.chain) != 0) {
                    output_print_error(-1, "Failed to subscribe to new_pair_update channel");
                }
                if (ws_client_subscribe(g_client, "new_launched_info", config.chain) != 0) {
                    output_print_error(-1, "Failed to subscribe to new_launched_info channel");
                }

                output_print_info("Subscribed to: new_pool_info, new_pair_update, new_launched_info");
            }

            last_state = state;
        }

        time_t now = time(NULL);

        if (now - last_ping >= (time_t)(config.heartbeat_interval_ms / 1000)) {
            ws_client_ping(g_client);
            last_ping = now;
            output_print_heartbeat();
        }

        int stats_interval = (config.verbosity >= OUTPUT_VERBOSE) ? 60 : 300;
        if (now - last_stats >= stats_interval) {
            print_periodic_stats();
            last_stats = now;
        }
    }

    printf("\n");
    output_print_info("Shutting down...");
    print_periodic_stats();

cleanup:
    ws_client_destroy(g_client);

    if (g_tracker) {
        tracker_cleanup(g_tracker);
        free(g_tracker);
        g_tracker = NULL;
    }

    output_cleanup();

    printf("\nGoodbye!\n");

    return ret;
}
