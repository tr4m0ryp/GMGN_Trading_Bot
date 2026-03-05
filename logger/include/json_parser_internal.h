/**
 * @file json_parser_internal.h
 * @brief Internal shared state for JSON parser module
 *
 * Exposes helper functions and error handling shared across the
 * split json_parser source files. Not part of the public API.
 *
 * Dependencies: cJSON, "json_parser.h"
 *
 * @date 2025-12-20
 */

#ifndef JSON_PARSER_INTERNAL_H
#define JSON_PARSER_INTERNAL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cjson/cJSON.h>

#include "json_parser.h"

/**
 * @brief Set thread-local error message
 *
 * @param msg Error message string
 */
void json_set_error(const char *msg);

/**
 * @brief Safe string copy from cJSON object field
 *
 * @param json Parent JSON object
 * @param key Field name to extract
 * @param dest Destination buffer
 * @param dest_size Size of destination buffer
 */
void json_safe_get_string(const cJSON *json, const char *key,
                          char *dest, size_t dest_size);

/**
 * @brief Get integer value from cJSON object field
 *
 * @param json Parent JSON object
 * @param key Field name to extract
 * @param default_val Default value if field missing
 * @return Extracted integer, or default_val
 */
int64_t json_safe_get_int(const cJSON *json, const char *key, int64_t default_val);

/**
 * @brief Get double value from cJSON object field
 *
 * @param json Parent JSON object
 * @param key Field name to extract
 * @param default_val Default value if field missing
 * @return Extracted double, or default_val
 */
double json_safe_get_double(const cJSON *json, const char *key, double default_val);

/**
 * @brief Parse token info from a bti (base token info) JSON object
 *
 * @param bti cJSON object containing token fields
 * @param token Output token info structure
 * @return 0 on success, -1 on error
 */
int json_parse_token_info_obj(const cJSON *bti, token_info_t *token);

/**
 * @brief Parse single pool from a JSON object
 *
 * @param pool_json cJSON object containing pool fields
 * @param pool Output pool data structure
 * @return 0 on success, -1 on error
 */
int json_parse_pool_obj(const cJSON *pool_json, pool_data_t *pool);

#endif /* JSON_PARSER_INTERNAL_H */
