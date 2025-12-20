/**
 * @file json_parser.h
 * @brief JSON parsing interface for GMGN messages
 *
 * Provides functions for parsing GMGN WebSocket JSON messages
 * into typed C structures.
 *
 * Dependencies: cJSON, "gmgn_types.h"
 *
 * @date 2025-12-20
 */

#ifndef JSON_PARSER_H
#define JSON_PARSER_H

#include "gmgn_types.h"
#include <stddef.h>

/**
 * @brief Parse message type from JSON
 *
 * Determines the type of GMGN message from raw JSON.
 *
 * @param json_str JSON string
 * @param json_len Length of JSON string
 *
 * @return Message type enumeration
 */
gmgn_msg_type_t json_parse_message_type(const char *json_str, size_t json_len);

/**
 * @brief Parse new pool message
 *
 * Extracts pool data from a new_pools channel message.
 *
 * @param json_str JSON string
 * @param json_len Length of JSON string
 * @param pools Output array for pool data
 * @param max_pools Maximum number of pools to extract
 *
 * @return Number of pools parsed, or -1 on error
 */
int json_parse_new_pools(const char *json_str, size_t json_len,
                         pool_data_t *pools, size_t max_pools);

/**
 * @brief Parse single pool object
 *
 * Parses a single pool JSON object into pool_data_t structure.
 *
 * @param json_str JSON string containing pool object
 * @param json_len Length of JSON string
 * @param pool Output pool structure
 *
 * @return 0 on success, -1 on error
 */
int json_parse_pool(const char *json_str, size_t json_len, pool_data_t *pool);

/**
 * @brief Parse token info object
 *
 * Parses the bti (base token info) portion of a pool message.
 *
 * @param json_str JSON string containing token info
 * @param json_len Length of JSON string
 * @param token Output token structure
 *
 * @return 0 on success, -1 on error
 */
int json_parse_token_info(const char *json_str, size_t json_len, 
                          token_info_t *token);

/**
 * @brief Create subscription message
 *
 * Generates JSON for channel subscription request.
 *
 * @param channel Channel name
 * @param chain Chain identifier
 * @param buffer Output buffer
 * @param buffer_size Size of output buffer
 *
 * @return Number of characters written, or -1 on error
 */
int json_create_subscribe_msg(const char *channel, const char *chain,
                              char *buffer, size_t buffer_size);

/**
 * @brief Create unsubscribe message
 *
 * @param channel Channel name
 * @param buffer Output buffer
 * @param buffer_size Size of output buffer
 *
 * @return Number of characters written, or -1 on error
 */
int json_create_unsubscribe_msg(const char *channel, char *buffer, 
                                size_t buffer_size);

/**
 * @brief Create ping message
 *
 * @param buffer Output buffer
 * @param buffer_size Size of output buffer
 *
 * @return Number of characters written, or -1 on error
 */
int json_create_ping_msg(char *buffer, size_t buffer_size);

/**
 * @brief Validate JSON string
 *
 * Checks if string is valid JSON.
 *
 * @param json_str JSON string
 * @param json_len Length of JSON string
 *
 * @return true if valid JSON
 */
bool json_validate(const char *json_str, size_t json_len);

/**
 * @brief Get error message from last parse operation
 *
 * @return Error message string, or NULL if no error
 */
const char *json_get_last_error(void);

#endif /* JSON_PARSER_H */
