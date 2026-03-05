/**
 * @file ai_main_internal.h
 * @brief Internal shared state for the ai_main module split files
 *
 * Exposes global state and function declarations shared between ai_main.c,
 * main/config.c, and main/callbacks.c. External callers should not include
 * this header.
 *
 * Dependencies: "gmgn_types.h", "websocket_client.h", "token_tracker.h",
 *               "ai_data_collector.h"
 *
 * @date 2025-12-20
 */

#ifndef AI_MAIN_INTERNAL_H
#define AI_MAIN_INTERNAL_H

#include <stdint.h>
#include <signal.h>

#include "gmgn_types.h"
#include "websocket_client.h"
#include "token_tracker.h"
#include "ai_data_collector.h"

/* Default configuration constants */
#define DEFAULT_WS_URL          "wss://ws.gmgn.ai/quotation"
#define DEFAULT_CHAIN           "sol"
#define DEFAULT_HEARTBEAT_MS    30000
#define DEFAULT_DATA_DIR        "./data"

/* Version info */
#define AI_VERSION_MAJOR    1
#define AI_VERSION_MINOR    0
#define AI_VERSION_PATCH    0

/* Global state shared across split files */
extern volatile sig_atomic_t g_running;
extern ws_client_t *g_client;
extern token_tracker_t *g_tracker;
extern ai_data_collector_t *g_collector;
extern time_t g_start_time;
extern uint64_t g_tokens_seen;
extern uint64_t g_tokens_passed;

/* ---- main/config.c ---- */

/**
 * @brief Print AI collector banner with death detection parameters
 */
void print_ai_banner(const char *data_dir);

/**
 * @brief Ensure data directory exists, creating it if needed
 *
 * @return 0 on success, -1 on failure
 */
int ensure_data_dir(const char *dir);

/**
 * @brief Print usage information
 */
void print_usage(const char *program);

/**
 * @brief Initialize application configuration with defaults
 */
void init_config(app_config_t *config, char *data_dir, size_t dir_size);

/**
 * @brief Parse command line arguments
 *
 * @return 0 on success, 1 if help was printed, -1 on error
 */
int parse_args(int argc, char *argv[], app_config_t *config,
               char *data_dir, size_t dir_size);

/* ---- main/callbacks.c ---- */

/**
 * @brief Callback when token passes filters - add to AI collector
 */
void on_token_passed(const tracked_token_t *tracked,
                     const token_info_t *info, void *user_data);

/**
 * @brief Callback for new pool events from WebSocket
 */
void on_new_pool(const pool_data_t *pool, void *user_data);

/**
 * @brief Callback for pair update events
 */
void on_pair_update(const pool_data_t *pool, void *user_data);

/**
 * @brief Callback for token launch events
 */
void on_token_launch(const pool_data_t *pool, void *user_data);

/**
 * @brief Callback for WebSocket errors
 */
void on_error(int error_code, const char *error_msg, void *user_data);

/**
 * @brief Print periodic statistics
 */
void print_periodic_stats(void);

#endif /* AI_MAIN_INTERNAL_H */
