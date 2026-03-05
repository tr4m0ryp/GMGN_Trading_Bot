/**
 * @file ai_main.c
 * @brief AI data collection main application entry point
 *
 * This application extends the GMGN token logger to collect chart data
 * for AI training. It connects to the WebSocket API, filters tokens,
 * and writes chart data to CSV files when tokens become inactive.
 *
 * Configuration parsing lives in main/config.c, event callbacks and
 * statistics live in main/callbacks.c.
 *
 * Dependencies: libwebsockets, cJSON, curl, openssl, pthread, logger_c
 *
 * Usage: ai_collector [options]
 *   -c, --chain CHAIN      Target chain (default: sol)
 *   -v, --verbose          Increase verbosity
 *   -d, --data-dir DIR     Output directory for CSV (default: ./data)
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
#include <sys/stat.h>
#include <sys/types.h>

#include "gmgn_types.h"
#include "websocket_client.h"
#include "filter.h"
#include "output.h"
#include "token_tracker.h"
#include "ai_data_collector.h"
#include "ai_main_internal.h"

/* Global state definitions (declared extern in ai_main_internal.h) */
volatile sig_atomic_t g_running = 1;
ws_client_t *g_client = NULL;
token_tracker_t *g_tracker = NULL;
ai_data_collector_t *g_collector = NULL;
time_t g_start_time = 0;
uint64_t g_tokens_seen = 0;
uint64_t g_tokens_passed = 0;

/**
 * @brief Signal handler for graceful shutdown
 */
static void signal_handler(int signum) {
    (void)signum;
    g_running = 0;
    output_print_info("Shutdown signal received, exiting...");
}


/**
 * @brief Main application entry point
 */
int main(int argc, char *argv[]) {
    app_config_t config;
    char data_dir[256];
    int ret = EXIT_SUCCESS;
    time_t last_ping = 0;
    time_t last_stats = 0;
    gmgn_conn_state_t last_state = GMGN_STATE_DISCONNECTED;

    /* Make stdout line-buffered */
    setvbuf(stdout, NULL, _IOLBF, 0);

    /* Initialize configuration */
    init_config(&config, data_dir, sizeof(data_dir));

    /* Parse command line arguments */
    int parse_result = parse_args(argc, argv, &config,
                                  data_dir, sizeof(data_dir));
    if (parse_result != 0) {
        return (parse_result > 0) ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    /* Setup signal handlers */
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    signal(SIGPIPE, SIG_IGN);

    /* Ensure data directory exists */
    if (ensure_data_dir(data_dir) != 0) {
        fprintf(stderr, "Failed to create data directory: %s\n", data_dir);
        return EXIT_FAILURE;
    }

    /* Initialize output system */
    output_init((output_verbosity_t)config.verbosity);

    /* Print banners */
    output_print_banner(&config);
    print_ai_banner(data_dir);
    output_print_filter_config(&config.filter);

    /* Initialize AI data collector */
    g_collector = malloc(sizeof(ai_data_collector_t));
    if (!g_collector) {
        output_print_error(-1, "Failed to allocate AI collector");
        return EXIT_FAILURE;
    }

    if (ai_collector_init(g_collector, data_dir) != 0) {
        output_print_error(-1, "Failed to initialize AI collector");
        free(g_collector);
        return EXIT_FAILURE;
    }

    if (ai_collector_start(g_collector) != 0) {
        output_print_error(-1, "Failed to start AI collector");
        ai_collector_cleanup(g_collector);
        free(g_collector);
        return EXIT_FAILURE;
    }

    output_print_info("AI data collector started");

    /* Initialize token tracker */
    g_tracker = malloc(sizeof(token_tracker_t));
    if (!g_tracker) {
        output_print_error(-1, "Failed to allocate token tracker");
        ret = EXIT_FAILURE;
        goto cleanup_collector;
    }

    if (tracker_init(g_tracker, &config.filter) != 0) {
        output_print_error(-1, "Failed to initialize token tracker");
        free(g_tracker);
        g_tracker = NULL;
        ret = EXIT_FAILURE;
        goto cleanup_collector;
    }

    tracker_set_callback(g_tracker, on_token_passed, NULL);

    if (tracker_start(g_tracker) != 0) {
        output_print_error(-1, "Failed to start token tracker");
        tracker_cleanup(g_tracker);
        free(g_tracker);
        g_tracker = NULL;
        ret = EXIT_FAILURE;
        goto cleanup_collector;
    }

    output_print_info("Token tracker started");

    /* Create WebSocket client */
    g_client = ws_client_create(config.websocket_url,
                                config.access_token[0] ?
                                config.access_token : NULL);
    if (!g_client) {
        output_print_error(-1, "Failed to create WebSocket client");
        ret = EXIT_FAILURE;
        goto cleanup_tracker;
    }

    /* Set callbacks */
    ws_client_set_pool_callback(g_client, on_new_pool, NULL);
    ws_client_set_pair_update_callback(g_client, on_pair_update, NULL);
    ws_client_set_token_launch_callback(g_client, on_token_launch, NULL);
    ws_client_set_error_callback(g_client, on_error, NULL);

    /* Connect */
    output_print_connection_status(GMGN_STATE_CONNECTING,
                                   config.websocket_url);

    if (ws_client_connect(g_client) != 0) {
        output_print_error(-1, "Failed to initiate connection");
        ret = EXIT_FAILURE;
        goto cleanup_client;
    }

    g_start_time = time(NULL);
    last_ping = g_start_time;
    last_stats = g_start_time;

    /* Main event loop */
    while (g_running) {
        /* Service WebSocket events */
        int events = ws_client_service(g_client, 10);
        if (events < 0) {
            output_print_error(-1, "WebSocket service error");

            /* Attempt reconnection */
            sleep(1);
            if (ws_client_connect(g_client) != 0) {
                output_print_error(-1, "Reconnection failed");
                ret = EXIT_FAILURE;
                break;
            }
            continue;
        }

        /* Check connection state changes */
        gmgn_conn_state_t state = ws_client_get_state(g_client);
        if (state != last_state) {
            output_print_connection_status(state, config.websocket_url);

            /* Subscribe when connected */
            if (state == GMGN_STATE_CONNECTED &&
                last_state == GMGN_STATE_CONNECTING) {
                output_print_info("Subscribing to WebSocket channels...");

                if (ws_client_subscribe(g_client, "new_pool_info",
                                        config.chain) != 0) {
                    output_print_error(-1,
                        "Failed to subscribe to new_pool_info");
                }

                if (ws_client_subscribe(g_client, "new_pair_update",
                                        config.chain) != 0) {
                    output_print_error(-1,
                        "Failed to subscribe to new_pair_update");
                }

                if (ws_client_subscribe(g_client, "new_launched_info",
                                        config.chain) != 0) {
                    output_print_error(-1,
                        "Failed to subscribe to new_launched_info");
                }

                output_print_info("Subscribed to channels");
            }

            last_state = state;
        }

        time_t now = time(NULL);

        /* Send periodic heartbeat */
        if (now - last_ping >=
            (time_t)(config.heartbeat_interval_ms / 1000)) {
            ws_client_ping(g_client);
            last_ping = now;
            output_print_heartbeat();
        }

        /* Print periodic stats */
        int stats_interval =
            (config.verbosity >= OUTPUT_VERBOSE) ? 60 : 300;
        if (now - last_stats >= stats_interval) {
            print_periodic_stats();
            last_stats = now;
        }
    }

    /* Print final stats */
    printf("\n");
    output_print_info("Shutting down...");
    print_periodic_stats();

cleanup_client:
    ws_client_destroy(g_client);
    g_client = NULL;

cleanup_tracker:
    if (g_tracker) {
        tracker_stop(g_tracker);
        tracker_cleanup(g_tracker);
        free(g_tracker);
        g_tracker = NULL;
    }

cleanup_collector:
    if (g_collector) {
        ai_collector_stop(g_collector);
        ai_collector_cleanup(g_collector);
        free(g_collector);
        g_collector = NULL;
    }

    output_cleanup();

    printf("\nGoodbye!\n");

    return ret;
}
