/**
 * @file ai_main.c
 * @brief AI data collection main application
 *
 * This application extends the GMGN token logger to collect chart data
 * for AI training. It connects to the WebSocket API, filters tokens,
 * and writes chart data to CSV files when tokens become inactive.
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
#include <getopt.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "gmgn_types.h"
#include "websocket_client.h"
#include "filter.h"
#include "output.h"
#include "token_tracker.h"
#include "ai_data_collector.h"

/* Version info */
#define AI_VERSION_MAJOR    1
#define AI_VERSION_MINOR    0
#define AI_VERSION_PATCH    0

/* Default configuration */
#define DEFAULT_WS_URL          "wss://ws.gmgn.ai/quotation"
#define DEFAULT_CHAIN           "sol"
#define DEFAULT_HEARTBEAT_MS    30000
#define DEFAULT_DATA_DIR        "./data"

/* Global state */
static volatile sig_atomic_t g_running = 1;
static ws_client_t *g_client = NULL;
static token_tracker_t *g_tracker = NULL;
static ai_data_collector_t *g_collector = NULL;
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
    printf("AI Data Collector - Collect token chart data for ML training\n\n");
    printf("Options:\n");
    printf("  -c, --chain CHAIN      Target blockchain (default: sol)\n");
    printf("  -u, --url URL          WebSocket URL (default: %s)\n", DEFAULT_WS_URL);
    printf("  -t, --token TOKEN      GMGN access token for auth channels\n");
    printf("  -d, --data-dir DIR     Output directory for CSV (default: %s)\n", 
           DEFAULT_DATA_DIR);
    printf("  -v, --verbose          Increase verbosity\n");
    printf("  -h, --help             Show this help message\n");
    printf("\nFilter options (same as gmgn_logger):\n");
    printf("  --min-mc VALUE         Minimum market cap in USD (default: 5500)\n");
    printf("  --max-mc VALUE         Maximum market cap in USD (default: 20000)\n");
    printf("  --min-kol VALUE        Minimum KOL count (default: 1)\n");
    printf("  --max-age VALUE        Maximum token age in minutes (default: 10)\n");
    printf("\n");
}

/**
 * @brief Initialize application configuration with defaults
 */
static void init_config(app_config_t *config, char *data_dir, size_t dir_size) {
    memset(config, 0, sizeof(app_config_t));
    
    strncpy(config->websocket_url, DEFAULT_WS_URL, sizeof(config->websocket_url) - 1);
    strncpy(config->chain, DEFAULT_CHAIN, sizeof(config->chain) - 1);
    config->heartbeat_interval_ms = DEFAULT_HEARTBEAT_MS;
    config->reconnect_attempts = 10;
    config->reconnect_delay_ms = 1000;
    config->verbosity = OUTPUT_NORMAL;
    
    /* Initialize filter defaults */
    filter_init_defaults(&config->filter);
    
    /* Default data directory */
    strncpy(data_dir, DEFAULT_DATA_DIR, dir_size - 1);
}

/**
 * @brief Parse command line arguments
 */
static int parse_args(int argc, char *argv[], app_config_t *config, 
                      char *data_dir, size_t dir_size) {
    static struct option long_options[] = {
        {"chain",       required_argument, 0, 'c'},
        {"url",         required_argument, 0, 'u'},
        {"token",       required_argument, 0, 't'},
        {"data-dir",    required_argument, 0, 'd'},
        {"verbose",     no_argument,       0, 'v'},
        {"help",        no_argument,       0, 'h'},
        {"min-mc",      required_argument, 0, 1001},
        {"max-mc",      required_argument, 0, 1002},
        {"min-kol",     required_argument, 0, 1003},
        {"max-age",     required_argument, 0, 1004},
        {0, 0, 0, 0}
    };
    
    int opt;
    int option_index = 0;
    
    while ((opt = getopt_long(argc, argv, "c:u:t:d:vh", 
                               long_options, &option_index)) != -1) {
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
            case 'd':
                strncpy(data_dir, optarg, dir_size - 1);
                break;
            case 'v':
                config->verbosity++;
                break;
            case 'h':
                print_usage(argv[0]);
                return 1;
            case 1001:
                config->filter.min_market_cap = (uint64_t)atol(optarg) * 100;
                break;
            case 1002:
                config->filter.max_market_cap = (uint64_t)atol(optarg) * 100;
                break;
            case 1003:
                config->filter.min_kol_count = (uint8_t)atoi(optarg);
                break;
            case 1004:
                config->filter.max_age_seconds = (uint32_t)atoi(optarg) * 60;
                break;
            default:
                print_usage(argv[0]);
                return -1;
        }
    }
    
    return 0;
}

/**
 * @brief Callback when token passes filters - add to AI collector
 *
 * This is the hook point where we start tracking a token for AI data.
 * Called by the token tracker when a token passes all filter criteria.
 *
 * @param tracked Tracked token information from tracker
 * @param info Current token info from API
 * @param user_data Unused
 */
static void on_token_passed(const tracked_token_t *tracked,
                            const token_info_t *info, void *user_data) {
    (void)user_data;
    (void)info;
    
    g_tokens_passed++;
    
    /* Log to output */
    pool_data_t pool;
    memset(&pool, 0, sizeof(pool));
    memcpy(pool.exchange, tracked->exchange, sizeof(pool.exchange));
    memcpy(pool.base_token.symbol, tracked->symbol, sizeof(pool.base_token.symbol));
    memcpy(pool.base_token.address, tracked->address, sizeof(pool.base_token.address));
    pool.base_token.market_cap = info->market_cap;
    pool.base_token.kol_count = info->kol_count;
    pool.base_token.holder_count = info->holder_count;
    pool.base_token.age_seconds = (uint32_t)(time(NULL) - tracked->discovered_at);
    
    output_log_token(&pool.base_token, &pool);
    
    /* Add to AI collector for chart data tracking */
    if (g_collector) {
        uint32_t age_sec = (uint32_t)(time(NULL) - tracked->discovered_at);
        int result = ai_collector_add_token(g_collector, 
                                            tracked->address,
                                            tracked->symbol,
                                            age_sec);
        if (result == 0) {
            char msg[128];
            snprintf(msg, sizeof(msg), "AI tracking started: %s", tracked->symbol);
            output_print_info(msg);
        }
    }
}

/**
 * @brief Callback for new pool events from WebSocket
 *
 * Adds token to tracker for periodic re-checking against filters.
 *
 * @param pool Pool data from WebSocket
 * @param user_data Unused
 */
static void on_new_pool(const pool_data_t *pool, void *user_data) {
    (void)user_data;
    
    if (!pool) {
        return;
    }
    
    g_tokens_seen++;
    
    /* Add token to tracker for periodic re-checking */
    if (g_tracker) {
        tracker_add_token(g_tracker, pool);
    }
}

/**
 * @brief Callback for pair update events (price/volume updates)
 */
static void on_pair_update(const pool_data_t *pool, void *user_data) {
    (void)user_data;
    (void)pool;
    /* Pair updates are handled by tracker internally */
}

/**
 * @brief Callback for token launch events
 */
static void on_token_launch(const pool_data_t *pool, void *user_data) {
    (void)user_data;
    
    if (!pool) {
        return;
    }
    
    /* Treat launches same as new pools */
    if (g_tracker) {
        tracker_add_token(g_tracker, pool);
    }
}

/**
 * @brief Callback for WebSocket errors
 */
static void on_error(int error_code, const char *error_msg, void *user_data) {
    (void)user_data;
    output_print_error(error_code, error_msg);
}

/**
 * @brief Print periodic statistics
 */
static void print_periodic_stats(void) {
    uint64_t messages = 0;
    uint64_t bytes = 0;
    uint32_t reconnects = 0;
    
    if (g_client) {
        ws_client_get_stats(g_client, &messages, &bytes, &reconnects);
    }
    
    uint32_t uptime = (uint32_t)(time(NULL) - g_start_time);
    
    /* Get tracker stats */
    uint32_t tracking = 0, passed = 0, expired = 0;
    if (g_tracker) {
        tracker_get_stats(g_tracker, &tracking, &passed, &expired);
    }
    
    /* Get AI collector stats */
    uint32_t ai_active = 0, ai_collected = 0;
    if (g_collector) {
        ai_active = ai_collector_get_active_count(g_collector);
        ai_collected = ai_collector_get_total_collected(g_collector);
    }
    
    output_print_stats(g_tokens_seen, g_tokens_passed, messages, uptime);
    printf("  Tracker: %u active | %u expired\n", tracking, expired);
    printf("  AI Data: %u tracking | %u collected to CSV\n", ai_active, ai_collected);
}

/**
 * @brief Print AI collector banner
 */
static void print_ai_banner(const char *data_dir) {
    printf("\n");
    printf("================================================\n");
    printf("   AI DATA COLLECTOR v%d.%d.%d\n", 
           AI_VERSION_MAJOR, AI_VERSION_MINOR, AI_VERSION_PATCH);
    printf("   Collecting chart data for ML training\n");
    printf("================================================\n");
    printf("  Output: %s\n", data_dir);
    printf("  Death detection:\n");
    printf("    - Candle gap: %d sec\n", AI_CANDLE_GAP_THRESHOLD_SEC);
    printf("    - Min volume: %.1f SOL\n", AI_VOLUME_THRESHOLD_SOL);
    printf("    - Price change: %.1f%%\n", AI_PRICE_CHANGE_THRESHOLD * 100);
    printf("    - Track time: %d-%d sec\n", AI_MIN_TRACK_TIME_SEC, AI_MAX_TRACK_TIME_SEC);
    printf("================================================\n\n");
}

/**
 * @brief Ensure data directory exists
 */
static int ensure_data_dir(const char *dir) {
    struct stat st = {0};
    
    if (stat(dir, &st) == -1) {
        if (mkdir(dir, 0755) != 0) {
            perror("mkdir");
            return -1;
        }
    }
    
    return 0;
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
    int parse_result = parse_args(argc, argv, &config, data_dir, sizeof(data_dir));
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
                                config.access_token[0] ? config.access_token : NULL);
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
    output_print_connection_status(GMGN_STATE_CONNECTING, config.websocket_url);
    
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
                
                if (ws_client_subscribe(g_client, "new_pool_info", config.chain) != 0) {
                    output_print_error(-1, "Failed to subscribe to new_pool_info");
                }
                
                if (ws_client_subscribe(g_client, "new_pair_update", config.chain) != 0) {
                    output_print_error(-1, "Failed to subscribe to new_pair_update");
                }
                
                if (ws_client_subscribe(g_client, "new_launched_info", config.chain) != 0) {
                    output_print_error(-1, "Failed to subscribe to new_launched_info");
                }
                
                output_print_info("Subscribed to channels");
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
        
        /* Print periodic stats (every minute in verbose, 5 min otherwise) */
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
