/**
 * @file filter.h
 * @brief Token filtering interface
 *
 * Provides functions for filtering tokens based on configurable
 * criteria including market cap, liquidity, holder count, and more.
 *
 * Dependencies: "gmgn_types.h"
 *
 * @date 2025-12-20
 */

#ifndef FILTER_H
#define FILTER_H

#include "gmgn_types.h"

/**
 * @brief Initialize filter with default values
 *
 * Sets up a filter configuration with reasonable defaults:
 * - Market cap: $5.5K - $10K
 * - Min KOL: 1
 * - Max age: 10 minutes
 * - Exchanges: raydium, orca, meteora
 *
 * @param filter Pointer to filter structure to initialize
 */
void filter_init_defaults(filter_config_t *filter);

/**
 * @brief Check if pool passes all filter criteria
 *
 * Evaluates the pool against all configured filter criteria.
 *
 * @param pool Pool data to evaluate
 * @param filter Filter configuration
 *
 * @return true if pool passes all filters, false otherwise
 */
bool filter_check_pool(const pool_data_t *pool, const filter_config_t *filter);

/**
 * @brief Check market cap range
 *
 * @param market_cap Market cap in USD cents
 * @param filter Filter configuration
 *
 * @return true if within range
 */
bool filter_check_market_cap(uint64_t market_cap, const filter_config_t *filter);

/**
 * @brief Check minimum liquidity
 *
 * @param liquidity Liquidity in USD cents
 * @param filter Filter configuration
 *
 * @return true if meets minimum
 */
bool filter_check_liquidity(uint64_t liquidity, const filter_config_t *filter);

/**
 * @brief Check token age
 *
 * @param age_seconds Token age in seconds
 * @param filter Filter configuration
 *
 * @return true if within age limit
 */
bool filter_check_age(uint32_t age_seconds, const filter_config_t *filter);

/**
 * @brief Check KOL count
 *
 * @param kol_count Number of KOLs
 * @param filter Filter configuration
 *
 * @return true if meets minimum
 */
bool filter_check_kol(uint8_t kol_count, const filter_config_t *filter);

/**
 * @brief Check holder count
 *
 * @param holder_count Number of holders
 * @param filter Filter configuration
 *
 * @return true if meets minimum
 */
bool filter_check_holders(uint32_t holder_count, const filter_config_t *filter);

/**
 * @brief Check if exchange is allowed
 *
 * @param exchange Exchange name
 * @param filter Filter configuration
 *
 * @return true if allowed (or no whitelist)
 */
bool filter_check_exchange(const char *exchange, const filter_config_t *filter);

/**
 * @brief Check if symbol is blacklisted
 *
 * @param symbol Token symbol
 * @param filter Filter configuration
 *
 * @return true if NOT blacklisted (passes check)
 */
bool filter_check_symbol(const char *symbol, const filter_config_t *filter);

/**
 * @brief Add exchange to whitelist
 *
 * @param filter Filter configuration
 * @param exchange Exchange name to add
 *
 * @return 0 on success, -1 if list full
 */
int filter_add_exchange(filter_config_t *filter, const char *exchange);

/**
 * @brief Add symbol to blacklist
 *
 * @param filter Filter configuration
 * @param symbol Symbol to exclude
 *
 * @return 0 on success, -1 if list full
 */
int filter_add_exclude_symbol(filter_config_t *filter, const char *symbol);

/**
 * @brief Get filter summary string
 *
 * Generates a human-readable summary of active filters.
 *
 * @param filter Filter configuration
 * @param buffer Output buffer
 * @param buffer_size Size of output buffer
 *
 * @return Number of characters written (excluding null terminator)
 */
int filter_get_summary(const filter_config_t *filter, char *buffer, 
                       size_t buffer_size);

#endif /* FILTER_H */
