/**
 * @file config.c
 * @brief Command-line argument parsing and configuration initialization
 *
 * Handles default configuration setup and getopt-based argument parsing
 * for the AI data collector application. Extracted from ai_main.c to
 * keep files under the 300-line limit.
 *
 * Dependencies: <getopt.h>, "gmgn_types.h", "filter.h", "output.h"
 *
 * @date 2025-12-20
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "gmgn_types.h"
#include "filter.h"
#include "output.h"
#include "ai_data_collector.h"
#include "ai_main_internal.h"

/**
 * @brief Print AI collector banner with death detection parameters
 */
void print_ai_banner(const char *data_dir) {
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
    printf("    - Track time: %d-%d sec\n",
           AI_MIN_TRACK_TIME_SEC, AI_MAX_TRACK_TIME_SEC);
    printf("================================================\n\n");
}

/**
 * @brief Ensure data directory exists, creating it if needed
 *
 * @return 0 on success, -1 on failure
 */
int ensure_data_dir(const char *dir) {
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
 * @brief Print usage information
 */
void print_usage(const char *program) {
    printf("\nUsage: %s [options]\n\n", program);
    printf("AI Data Collector - Collect token chart data for ML training\n\n");
    printf("Options:\n");
    printf("  -c, --chain CHAIN      Target blockchain (default: sol)\n");
    printf("  -u, --url URL          WebSocket URL (default: %s)\n",
           DEFAULT_WS_URL);
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
 *
 * @param config Application configuration to fill with defaults
 * @param data_dir Buffer for data directory path
 * @param dir_size Size of data_dir buffer
 */
void init_config(app_config_t *config, char *data_dir, size_t dir_size) {
    memset(config, 0, sizeof(app_config_t));

    strncpy(config->websocket_url, DEFAULT_WS_URL,
            sizeof(config->websocket_url) - 1);
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
 *
 * @param argc Argument count
 * @param argv Argument vector
 * @param config Configuration to populate
 * @param data_dir Buffer for data directory path
 * @param dir_size Size of data_dir buffer
 *
 * @return 0 on success, 1 if help was printed, -1 on error
 */
int parse_args(int argc, char *argv[], app_config_t *config,
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
                strncpy(config->chain, optarg,
                        sizeof(config->chain) - 1);
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
