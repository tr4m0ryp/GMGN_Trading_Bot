/**
 * @file token_tracker_internal.h
 * @brief Internal shared state for token tracker module
 *
 * Exposes CURL handles, API fetch functions, and helper utilities
 * shared across the split tracker source files. Not part of the
 * public API.
 *
 * Dependencies: curl, cJSON, "token_tracker.h", "filter.h", "output.h"
 *
 * @date 2025-12-20
 */

#ifndef TOKEN_TRACKER_INTERNAL_H
#define TOKEN_TRACKER_INTERNAL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <curl/curl.h>
#include <cjson/cJSON.h>

#include "token_tracker.h"
#include "filter.h"
#include "output.h"

/* GMGN API endpoints */
#define GMGN_KOL_HOLDERS_API "https://gmgn.ai/vas/api/v1/token_holders/sol/"
#define GMGN_MCAP_CANDLES_API "https://gmgn.ai/api/v1/token_mcap_candles/sol/"

/**
 * @brief CURL response buffer
 */
typedef struct {
    char *data;
    size_t size;
} curl_buffer_t;

/**
 * @brief Initialize persistent CURL handles for fast API calls
 * Called once at startup to avoid per-request overhead.
 */
void tracker_init_curl_handles(void);

/**
 * @brief Cleanup persistent CURL handles
 */
void tracker_cleanup_curl_handles(void);

/**
 * @brief Fetch KOL count from GMGN token holders API
 *
 * @param address Token address
 * @param kol_count Output KOL count
 * @return 0 on success, -1 on error
 */
int tracker_fetch_kol_count(const char *address, uint8_t *kol_count);

/**
 * @brief Fetch market cap from GMGN token_mcap_candles API
 *
 * @param address Token address
 * @param market_cap Output market cap in cents
 * @return 0 on success, -1 on error
 */
int tracker_fetch_market_cap(const char *address, uint64_t *market_cap);

/**
 * @brief Fetch combined token info (KOL + market cap)
 *
 * @param address Token address
 * @param info Output token info structure
 * @return 0 on success, -1 on error
 */
int tracker_fetch_token_info(const char *address, token_info_t *info);

/**
 * @brief Find free slot in tracker array
 * Also reclaims expired and passed slots for reuse.
 *
 * @param tracker Tracker instance
 * @return Slot index, or -1 if full
 */
int tracker_find_free_slot(token_tracker_t *tracker);

/**
 * @brief Find token by address
 *
 * @param tracker Tracker instance
 * @param address Token address to find
 * @return Slot index, or -1 if not found
 */
int tracker_find_token(token_tracker_t *tracker, const char *address);

/**
 * @brief Get current time in milliseconds
 *
 * @return Current epoch time in milliseconds
 */
uint64_t tracker_get_time_ms(void);

/**
 * @brief Background tracking thread function
 *
 * @param arg Pointer to token_tracker_t
 * @return NULL
 */
void *tracker_thread_func(void *arg);

#endif /* TOKEN_TRACKER_INTERNAL_H */
