/**
 * @file logger_integration.h
 * @brief Integration with logger_c for new token discovery
 *
 * Connects to the token tracker in logger_c to receive callbacks
 * when new tokens pass the filter criteria.
 *
 * Usage:
 * 1. Initialize the logger_c token tracker
 * 2. Register ai_trader's callback with tracker_set_callback()
 * 3. New tokens are automatically added to trading
 *
 * Dependencies: ../logger_c/include/token_tracker.h
 *
 * @date 2025-12-24
 */

#ifndef LOGGER_INTEGRATION_H
#define LOGGER_INTEGRATION_H

#include <stdbool.h>

/* Forward declarations to avoid including logger_c headers directly */
struct tracked_token_s;
struct token_info_s;

/**
 * @brief Callback type for new token events
 *
 * Called when a new token passes the filter criteria in logger_c.
 *
 * @param address Token mint address
 * @param symbol Token symbol
 * @param user_data User-provided context
 */
typedef void (*new_token_callback_t)(const char *address, const char *symbol, void *user_data);

/**
 * @brief Logger integration state
 */
typedef struct {
    new_token_callback_t callback;      /* Callback to call for new tokens */
    void *user_data;                    /* User data for callback */
    bool connected;                     /* Connection state */
} logger_integration_t;

/**
 * @brief Initialize logger integration
 *
 * @param integration Integration state
 *
 * @return 0 on success, -1 on error
 */
int logger_integration_init(logger_integration_t *integration);

/**
 * @brief Set callback for new tokens
 *
 * @param integration Integration state
 * @param callback Function to call when new token is discovered
 * @param user_data Context to pass to callback
 */
void logger_integration_set_callback(logger_integration_t *integration,
                                     new_token_callback_t callback,
                                     void *user_data);

/**
 * @brief Connect to logger_c token tracker
 *
 * Attempts to connect to the shared token tracker instance.
 * Must be called after logger_c starts.
 *
 * @param integration Integration state
 *
 * @return 0 on success, -1 on error
 */
int logger_integration_connect(logger_integration_t *integration);

/**
 * @brief Disconnect from logger_c
 *
 * @param integration Integration state
 */
void logger_integration_disconnect(logger_integration_t *integration);

/**
 * @brief Cleanup integration resources
 *
 * @param integration Integration state
 */
void logger_integration_cleanup(logger_integration_t *integration);

#endif /* LOGGER_INTEGRATION_H */
