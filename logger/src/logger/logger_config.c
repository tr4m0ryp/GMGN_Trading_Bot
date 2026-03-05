/**
 * @file logger_config.c
 * @brief Configuration parsing for GMGN logger
 *
 * Implements default configuration initialization and command-line
 * argument parsing for the GMGN token logger application.
 *
 * Dependencies: "gmgn_types.h", "filter.h", "output.h"
 *
 * @date 2025-12-20
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

#include "gmgn_types.h"
#include "filter.h"
#include "output.h"

/* Default configuration */
#define DEFAULT_WS_URL          "wss://ws.gmgn.ai/quotation"
#define DEFAULT_CHAIN           "sol"
#define DEFAULT_HEARTBEAT_MS    30000
#define DEFAULT_RECONNECT_MS    1000
#define DEFAULT_MAX_RECONNECT   10

/**
 * @brief Print usage information
 */
void logger_print_usage(const char *program) {
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
    printf("  --max-mc VALUE         Maximum market cap in USD (default: 20000)\n");
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
int logger_parse_args(int argc, char *argv[], app_config_t *config) {
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
                logger_print_usage(argv[0]);
                return 1;
            case 1001:
                config->filter.min_market_cap = (uint64_t)(atof(optarg) * 100.0);
                break;
            case 1002:
                config->filter.max_market_cap = (uint64_t)(atof(optarg) * 100.0);
                break;
            case 1003:
                config->filter.min_liquidity = (uint64_t)(atof(optarg) * 100.0);
                break;
            case 1004:
                config->filter.min_kol_count = (uint8_t)atoi(optarg);
                break;
            case 1005:
                config->filter.max_age_seconds = (uint32_t)(atoi(optarg) * 60);
                break;
            case 1006:
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
void logger_init_config(app_config_t *config) {
    memset(config, 0, sizeof(app_config_t));

    strncpy(config->websocket_url, DEFAULT_WS_URL, sizeof(config->websocket_url) - 1);
    strncpy(config->chain, DEFAULT_CHAIN, sizeof(config->chain) - 1);

    config->reconnect_attempts = DEFAULT_MAX_RECONNECT;
    config->reconnect_delay_ms = DEFAULT_RECONNECT_MS;
    config->heartbeat_interval_ms = DEFAULT_HEARTBEAT_MS;
    config->verbosity = OUTPUT_NORMAL;

    filter_init_defaults(&config->filter);
}
