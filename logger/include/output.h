/**
 * @file output.h
 * @brief Terminal output interface for GMGN logger
 *
 * Provides formatted terminal output for displaying token
 * information, connection status, and statistics.
 *
 * Dependencies: "gmgn_types.h"
 *
 * @date 2025-12-20
 */

#ifndef OUTPUT_H
#define OUTPUT_H

#include "gmgn_types.h"

/**
 * @brief Output verbosity levels
 */
typedef enum {
    OUTPUT_MINIMAL = 0,     /* Only matching tokens */
    OUTPUT_NORMAL = 1,      /* Tokens + status updates */
    OUTPUT_VERBOSE = 2      /* All messages + debug info */
} output_verbosity_t;

/**
 * @brief Initialize output system
 *
 * Sets up terminal output with specified verbosity.
 *
 * @param verbosity Output verbosity level
 *
 * @return 0 on success, -1 on error
 */
int output_init(output_verbosity_t verbosity);

/**
 * @brief Cleanup output system
 */
void output_cleanup(void);

/**
 * @brief Print startup banner
 *
 * Displays application banner with version and config info.
 *
 * @param config Application configuration
 */
void output_print_banner(const app_config_t *config);

/**
 * @brief Print filter configuration summary
 *
 * @param filter Filter configuration to display
 */
void output_print_filter_config(const filter_config_t *filter);

/**
 * @brief Print connection status
 *
 * @param state Current connection state
 * @param url WebSocket URL
 */
void output_print_connection_status(gmgn_conn_state_t state, const char *url);

/**
 * @brief Log new token that passed filters
 *
 * Displays formatted token information in terminal.
 *
 * @param token Token information
 * @param pool Pool data
 */
void output_log_token(const token_info_t *token, const pool_data_t *pool);

/**
 * @brief Log filtered out token (verbose mode only)
 *
 * @param token Token information
 * @param reason Reason for filtering
 */
void output_log_filtered(const token_info_t *token, const char *reason);

/**
 * @brief Print statistics summary
 *
 * @param tokens_seen Total tokens seen
 * @param tokens_passed Tokens that passed filters
 * @param messages_received Total WebSocket messages
 * @param uptime_seconds Connection uptime
 */
void output_print_stats(uint64_t tokens_seen, uint64_t tokens_passed,
                        uint64_t messages_received, uint32_t uptime_seconds);

/**
 * @brief Print error message
 *
 * @param error_code Error code
 * @param message Error message
 */
void output_print_error(int error_code, const char *message);

/**
 * @brief Print warning message
 *
 * @param message Warning message
 */
void output_print_warning(const char *message);

/**
 * @brief Print info message (normal verbosity)
 *
 * @param message Info message
 */
void output_print_info(const char *message);

/**
 * @brief Print debug message (verbose mode only)
 *
 * @param message Debug message
 */
void output_print_debug(const char *message);

/**
 * @brief Print heartbeat indicator
 *
 * Shows that connection is alive (verbose mode).
 */
void output_print_heartbeat(void);

/**
 * @brief Format market cap for display
 *
 * Converts cents to formatted USD string (e.g., "$5.5K").
 *
 * @param market_cap_cents Market cap in USD cents
 * @param buffer Output buffer
 * @param buffer_size Size of output buffer
 *
 * @return Pointer to buffer
 */
char *output_format_market_cap(uint64_t market_cap_cents, char *buffer, 
                               size_t buffer_size);

/**
 * @brief Format age for display
 *
 * Converts seconds to human-readable format (e.g., "5m 30s").
 *
 * @param age_seconds Age in seconds
 * @param buffer Output buffer
 * @param buffer_size Size of output buffer
 *
 * @return Pointer to buffer
 */
char *output_format_age(uint32_t age_seconds, char *buffer, size_t buffer_size);

/**
 * @brief Set output verbosity level
 *
 * @param verbosity New verbosity level
 */
void output_set_verbosity(output_verbosity_t verbosity);

/**
 * @brief Get current verbosity level
 *
 * @return Current verbosity level
 */
output_verbosity_t output_get_verbosity(void);

#endif /* OUTPUT_H */
