/**
 * @file ai_data_collector_internal.h
 * @brief Internal shared state and types for the AI data collector module
 *
 * This header exposes internal helpers and global state shared across the
 * split source files that implement ai_data_collector.h. External callers
 * should use ai_data_collector.h instead.
 *
 * Dependencies: <curl/curl.h>, <cjson/cJSON.h>, "ai_data_collector.h"
 *
 * @date 2025-12-20
 */

#ifndef AI_DATA_COLLECTOR_INTERNAL_H
#define AI_DATA_COLLECTOR_INTERNAL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <errno.h>
#include <curl/curl.h>
#include <cjson/cJSON.h>

#include "ai_data_collector.h"

/* CURL response buffer used by write callback */
typedef struct {
    char *data;
    size_t size;
} curl_buffer_t;

/* ---- curl_utils.c ---- */

/**
 * @brief CURL write callback for API responses
 */
size_t curl_write_cb(void *contents, size_t size, size_t nmemb, void *userp);

/**
 * @brief Initialize the persistent CURL handle for chart API requests
 *
 * Reads cookie values from environment variables and configures HTTP headers.
 * Safe to call multiple times - only initializes once.
 */
void init_curl_handle(void);

/**
 * @brief Cleanup the persistent CURL handle and free associated resources
 */
void cleanup_curl_handle(void);

/**
 * @brief Get the global CURL handle for chart API requests
 *
 * @return Pointer to the global CURL handle, or NULL if not initialized
 */
CURL *get_curl_chart_handle(void);

/* ---- chart_fetcher.c ---- */

/**
 * @brief Fetch chart data from GMGN API
 *
 * @param address Token address
 * @param out_data Output buffer for JSON data (caller must free)
 * @param out_len Output length of data
 *
 * @return 0 on success, -1 on failure
 */
int fetch_chart_data(const char *address, char **out_data, size_t *out_len);

/* ---- chart_analyzer.c ---- */

/**
 * @brief Analyze chart data for death conditions
 *
 * @param json_data Raw JSON chart data
 * @param token Token structure to update with analysis
 *
 * @return Death reason if dead, AI_DEATH_NONE if still active
 */
ai_death_reason_t analyze_chart_data(const char *json_data,
                                     ai_tracked_token_t *token);

/* ---- csv_writer.c ---- */

/**
 * @brief Escape string for CSV (handles quotes and newlines)
 */
void csv_escape(FILE *fp, const char *str);

/**
 * @brief Extract clean candle data from raw API response
 *
 * Parses the raw JSON and extracts only the candle list with essential fields:
 * time, open, high, low, close, volume
 *
 * @param raw_json Raw JSON response from API
 *
 * @return Newly allocated string with clean JSON array, or NULL on failure
 *         Caller must free the returned string
 */
char *extract_candle_data(const char *raw_json);

/**
 * @brief Write token data to CSV file
 *
 * @param collector Collector state
 * @param token Token to write
 * @param death_reason Why the token was declared dead
 *
 * @return 0 on success, -1 on failure
 */
int write_token_to_csv(ai_data_collector_t *collector,
                       ai_tracked_token_t *token,
                       ai_death_reason_t death_reason);

/* ---- collector_thread.c ---- */

/**
 * @brief Clear a token slot and free its chart data
 */
void clear_token_slot(ai_tracked_token_t *token);

/**
 * @brief Background collector thread entry point
 */
void *collector_thread(void *arg);

#endif /* AI_DATA_COLLECTOR_INTERNAL_H */
