/**
 * @file websocket_client.h
 * @brief WebSocket client interface for GMGN connection
 *
 * Provides WebSocket connection management including connection,
 * subscription, message handling, and automatic reconnection.
 *
 * Dependencies: libwebsockets, "gmgn_types.h"
 *
 * @date 2025-12-20
 */

#ifndef WEBSOCKET_CLIENT_H
#define WEBSOCKET_CLIENT_H

#include "gmgn_types.h"

/* Forward declaration */
typedef struct ws_client ws_client_t;

/**
 * @brief Create a new WebSocket client instance
 *
 * Allocates and initializes a WebSocket client with the specified
 * connection parameters. Does not establish connection.
 *
 * @param url WebSocket URL (e.g., "wss://gmgn.ai/ws")
 * @param access_token Optional authentication token (NULL if none)
 *
 * @return Pointer to client instance, or NULL on allocation failure
 */
ws_client_t *ws_client_create(const char *url, const char *access_token);

/**
 * @brief Destroy WebSocket client and free resources
 *
 * Closes connection if active and frees all allocated memory.
 *
 * @param client Pointer to client instance (safe to pass NULL)
 */
void ws_client_destroy(ws_client_t *client);

/**
 * @brief Establish WebSocket connection
 *
 * Initiates connection to the configured WebSocket server.
 * This is an asynchronous operation; use ws_client_get_state()
 * to check connection status.
 *
 * @param client Pointer to client instance
 *
 * @return 0 on success, -1 on error
 */
int ws_client_connect(ws_client_t *client);

/**
 * @brief Disconnect from WebSocket server
 *
 * Gracefully closes the WebSocket connection.
 *
 * @param client Pointer to client instance
 *
 * @return 0 on success, -1 on error
 */
int ws_client_disconnect(ws_client_t *client);

/**
 * @brief Subscribe to a GMGN channel
 *
 * Sends subscription request for the specified channel.
 *
 * @param client Pointer to client instance
 * @param channel Channel name (e.g., "new_pools")
 * @param chain Chain identifier (e.g., "sol")
 *
 * @return 0 on success, -1 on error
 */
int ws_client_subscribe(ws_client_t *client, const char *channel, 
                        const char *chain);

/**
 * @brief Unsubscribe from a GMGN channel
 *
 * @param client Pointer to client instance
 * @param channel Channel name
 *
 * @return 0 on success, -1 on error
 */
int ws_client_unsubscribe(ws_client_t *client, const char *channel);

/**
 * @brief Service WebSocket events
 *
 * Must be called regularly to process WebSocket events.
 * This drives the event loop and triggers callbacks.
 *
 * @param client Pointer to client instance
 * @param timeout_ms Maximum time to wait for events (milliseconds)
 *
 * @return Number of events processed, or -1 on error
 */
int ws_client_service(ws_client_t *client, int timeout_ms);

/**
 * @brief Get current connection state
 *
 * @param client Pointer to client instance
 *
 * @return Current connection state
 */
gmgn_conn_state_t ws_client_get_state(const ws_client_t *client);

/**
 * @brief Set callback for new pool events
 *
 * @param client Pointer to client instance
 * @param callback Callback function
 * @param user_data User context passed to callback
 */
void ws_client_set_pool_callback(ws_client_t *client, pool_callback_fn callback,
                                  void *user_data);

/**
 * @brief Set callback for error events
 *
 * @param client Pointer to client instance
 * @param callback Callback function
 * @param user_data User context passed to callback
 */
void ws_client_set_error_callback(ws_client_t *client, error_callback_fn callback,
                                   void *user_data);

/**
 * @brief Set callback for pair update events (price/volume updates)
 *
 * @param client Pointer to client instance
 * @param callback Callback function
 * @param user_data User context passed to callback
 */
void ws_client_set_pair_update_callback(ws_client_t *client, 
                                         pair_update_callback_fn callback,
                                         void *user_data);

/**
 * @brief Set callback for token launch events
 *
 * @param client Pointer to client instance
 * @param callback Callback function
 * @param user_data User context passed to callback
 */
void ws_client_set_token_launch_callback(ws_client_t *client,
                                          token_launch_callback_fn callback,
                                          void *user_data);

/**
 * @brief Send ping to keep connection alive
 *
 * @param client Pointer to client instance
 *
 * @return 0 on success, -1 on error
 */
int ws_client_ping(ws_client_t *client);

/**
 * @brief Get connection statistics
 *
 * @param client Pointer to client instance
 * @param messages_received Output: total messages received
 * @param bytes_received Output: total bytes received
 * @param reconnect_count Output: number of reconnections
 */
void ws_client_get_stats(const ws_client_t *client, uint64_t *messages_received,
                         uint64_t *bytes_received, uint32_t *reconnect_count);

#endif /* WEBSOCKET_CLIENT_H */
