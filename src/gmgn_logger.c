/**
 * @file gmgn_logger.c
 * @brief GMGN Trenches token logger main application
 *
 * This application connects to GMGN.ai WebSocket API to monitor
 * newly created Solana tokens in real-time. It applies configurable
 * filters and logs matching tokens to the terminal.
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
#include <getopt.h>

#include "gmgn_types.h"
#include "websocket_client.h"
#include "json_parser.h"
#include "filter.h"
#include "output.h"

/* Version info */
#define VERSION_MAJOR   1
#define VERSION_MINOR   0
#define VERSION_PATCH   0

/* Default configuration */
#define DEFAULT_WS_URL          "wss://ws.gmgn.ai/quotation"
#define DEFAULT_CHAIN           "sol"
#define DEFAULT_HEARTBEAT_MS    30000
#define DEFAULT_RECONNECT_MS    1000
#define DEFAULT_MAX_RECONNECT   10

/* Global state */
static volatile sig_atomic_t g_running = 1;
static ws_client_t *g_client = NULL;
static time_t g_start_time = 0;
static uint64_t g_tokens_seen = 0;
static uint64_t g_tokens_passed = 0;

/**
 * @brief Signal handler for graceful shutdown
 */
static void signal_handler(int signum) {
    (void)signum;
    g_running = 0;
    
    output_print_info("Shutdown signal received, exiting...");
}

/**
 * @brief Print usage information
 */
static void print_usage(const char *program) {
    printf("\nUsage: %s [options]\n\n", program);
    printf("GMGN Trenches Token Logger - Monitor new Solana tokens in real-time\n\n");
    printf("Options:\n");
    printf("  -c, --chain CHAIN      Target blockchain (default: sol)\n");
    printf("  -u, --url URL          WebSocket URL (default: %s)\n", DEFAULT_WS_URL);
    printf("  -t, --token TOKEN      GMGN access token for auth channels\n");
    printf("  -v, --verbose          Increase verbosity (can use multiple times)\n");
    printf("  -q, --quiet            Minimal output (only matching tokens)\n");
    printf("  -h, --help             Show this help message\n");
    printf("\nFilter options:\n");
    printf("  --min-mc VALUE         Minimum market cap in USD (default: 5500)\n");
    printf("  --max-mc VALUE         Maximum market cap in USD (default: 10000)\n");
    printf("  --min-liq VALUE        Minimum liquidity in USD (default: 5000)\n");
    printf("  --min-kol VALUE        Minimum KOL count (default: 1)\n");
    printf("  --max-age VALUE        Maximum token age in minutes (default: 10)\n");
    printf("  --min-holders VALUE    Minimum holder count (default: 10)\n");
    printf("\nExamples:\n");
    printf("  %s                     Start with default filters\n", program);
    printf("  %s -v -v               Start with verbose output\n", program);
    printf("  %s --min-mc 10000      Filter tokens with MC >= $10K\n", program);
    printf("\n");
}

/**
 * @brief Parse command line arguments
 */
static int parse_args(int argc, char *argv[], app_config_t *config) {
    static struct option long_options[] = {
        {"chain",       required_argument, 0, 'c'},
        {"url",         required_argument, 0, 'u'},
        {"token",       required_argument, 0, 't'},
        {"verbose",     no_argument,       0, 'v'},
        {"quiet",       no_argument,       0, 'q'},
        {"help",        no_argument,       0, 'h'},
        {"min-mc",      required_argument, 0, 1001},
        {"max-mc",      required_argument, 0, 1002},
        {"min-liq",     required_argument, 0, 1003},
        {"min-kol",     required_argument, 0, 1004},
        {"max-age",     required_argument, 0, 1005},
        {"min-holders", required_argument, 0, 1006},
        {0, 0, 0, 0}
    };
    
    int opt;
    int option_index = 0;
    
    while ((opt = getopt_long(argc, argv, "c:u:t:vqh", long_options, 
                              &option_index)) != -1) {
        switch (opt) {
            case 'c':
                strncpy(config->chain, optarg, sizeof(config->chain) - 1);
                break;
                
            case 'u':
                strncpy(config->websocket_url, optarg, 
                        sizeof(config->websocket_url) - 1);
                break;
                
            case 't':
                strncpy(config->access_token, optarg, 
                        sizeof(config->access_token) - 1);
                break;
                
            case 'v':
                if (config->verbosity < 2) {
                    config->verbosity++;
                }
                break;
                
            case 'q':
                config->verbosity = 0;
                break;
                
            case 'h':
                print_usage(argv[0]);
                return 1;
                
            case 1001:  /* --min-mc */
                config->filter.min_market_cap = (uint64_t)(atof(optarg) * 100.0);
                break;
                
            case 1002:  /* --max-mc */
                config->filter.max_market_cap = (uint64_t)(atof(optarg) * 100.0);
                break;
                
            case 1003:  /* --min-liq */
                config->filter.min_liquidity = (uint64_t)(atof(optarg) * 100.0);
                break;
                
            case 1004:  /* --min-kol */
                config->filter.min_kol_count = (uint8_t)atoi(optarg);
                break;
                
            case 1005:  /* --max-age */
                config->filter.max_age_seconds = (uint32_t)(atoi(optarg) * 60);
                break;
                
            case 1006:  /* --min-holders */
                config->filter.min_holder_count = (uint32_t)atoi(optarg);
                break;
                
            default:
                return -1;
        }
    }
    
    return 0;
}

/**
 * @brief Initialize default configuration
 */
static void init_config(app_config_t *config) {
    memset(config, 0, sizeof(app_config_t));
    
    strncpy(config->websocket_url, DEFAULT_WS_URL, sizeof(config->websocket_url) - 1);
    strncpy(config->chain, DEFAULT_CHAIN, sizeof(config->chain) - 1);
    
    config->reconnect_attempts = DEFAULT_MAX_RECONNECT;
    config->reconnect_delay_ms = DEFAULT_RECONNECT_MS;
    config->heartbeat_interval_ms = DEFAULT_HEARTBEAT_MS;
    config->verbosity = OUTPUT_NORMAL;
    
    /* Initialize default filters */
    filter_init_defaults(&config->filter);
}

/**
 * @brief Callback for new pool events
 */
static void on_new_pool(const pool_data_t *pool, void *user_data) {
    filter_config_t *filter = (filter_config_t *)user_data;
    
    if (!pool) {
        return;
    }
    
    g_tokens_seen++;
    
    /* Apply filters */
    if (filter && !filter_check_pool(pool, filter)) {
        /* Token filtered out - don't log unless in verbose mode */
        return;
    }
    
    g_tokens_passed++;
    
    /* Log the matching token */
    output_log_token(&pool->base_token, pool);
}

/**
 * @brief Callback for error events
 */
static void on_error(int error_code, const char *error_msg, void *user_data) {
    (void)user_data;
    output_print_error(error_code, error_msg);
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
        
        output_print_stats(g_tokens_seen, g_tokens_passed, messages, uptime);
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
    
    /* Make stdout line-buffered for immediate debug output */
    setvbuf(stdout, NULL, _IOLBF, 0);
    
    /* Initialize configuration */
    init_config(&config);
    
    /* Parse command line arguments */
    int parse_result = parse_args(argc, argv, &config);
    if (parse_result != 0) {
        return (parse_result > 0) ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    
    /* Setup signal handlers */
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    signal(SIGPIPE, SIG_IGN);
    
    /* Initialize output system */
    output_init((output_verbosity_t)config.verbosity);
    
    /* Print banner */
    output_print_banner(&config);
    output_print_filter_config(&config.filter);
    
    /* Create WebSocket client */
    g_client = ws_client_create(config.websocket_url, 
                                 config.access_token[0] ? config.access_token : NULL);
    if (!g_client) {
        output_print_error(-1, "Failed to create WebSocket client");
        return EXIT_FAILURE;
    }
    
    /* Set callbacks */
    ws_client_set_pool_callback(g_client, on_new_pool, &config.filter);
    ws_client_set_error_callback(g_client, on_error, NULL);
    
    /* Connect */
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
        /* Service WebSocket events */
        int events = ws_client_service(g_client, 100);
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
            if (state == GMGN_STATE_CONNECTED && last_state == GMGN_STATE_CONNECTING) {
                output_print_info("Subscribing to new_pool_info channel...");
                
                if (ws_client_subscribe(g_client, "new_pool_info", config.chain) != 0) {
                    output_print_error(-1, "Failed to subscribe to channel");
                }
            }
            
            last_state = state;
        }
        
        time_t now = time(NULL);
        
        /* Send periodic heartbeat */
        if (now - last_ping >= (time_t)(config.heartbeat_interval_ms / 1000)) {
            ws_client_ping(g_client);
            last_ping = now;
            output_print_heartbeat();
        }
        
        /* Print periodic stats (every 5 minutes in normal mode, 1 minute in verbose) */
        int stats_interval = (config.verbosity >= OUTPUT_VERBOSE) ? 60 : 300;
        if (now - last_stats >= stats_interval) {
            print_periodic_stats();
            last_stats = now;
        }
    }
    
    /* Print final stats */
    printf("\n");
    output_print_info("Shutting down...");
    print_periodic_stats();
    
cleanup:
    ws_client_destroy(g_client);
    output_cleanup();
    
    printf("\nGoodbye!\n");
    
    return ret;
}
