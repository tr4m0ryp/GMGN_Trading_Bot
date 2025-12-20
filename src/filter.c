/**
 * @file filter.c
 * @brief Token filtering implementation
 *
 * Implements filtering logic for new token pools based on configurable
 * criteria including market cap, liquidity, holder count, KOL presence,
 * token age, and exchange whitelist.
 *
 * Dependencies: "filter.h", "gmgn_types.h"
 *
 * @date 2025-12-20
 */

#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <stdbool.h>
#include <stdint.h>
#include <limits.h>

#include "filter.h"

void filter_init_defaults(filter_config_t *filter) {
    if (!filter) {
        return;
    }
    
    memset(filter, 0, sizeof(filter_config_t));
    
    /* Default filters per user requirements:
     * MC min 5.5K, max 10K; KOL min 1; age max 10 min
     * Values stored in cents (x100) for precision
     */
    filter->min_market_cap = 550000;      /* $5.5K in cents */
    filter->max_market_cap = 1000000;     /* $10K in cents */
    
    /* Liquidity minimum: not specified, use reasonable default */
    filter->min_liquidity = 0;
    
    /* Volume minimum: not specified */
    filter->min_volume_24h = 0;
    
    /* Holder requirements: not specified */
    filter->min_holder_count = 0;
    filter->max_top_10_ratio = 10000;     /* 100% (no limit) */
    filter->max_creator_ratio = 10000;    /* 100% (no limit) */
    
    /* Token age: max 10 minutes */
    filter->max_age_seconds = 10 * 60;    /* 10 minutes in seconds */
    
    /* KOL minimum: 1 */
    filter->min_kol_count = 1;
    
    /* No exchange filtering */
    /* No excluded symbols */
}

bool filter_check_market_cap(uint64_t market_cap, const filter_config_t *filter) {
    if (!filter) {
        return true;
    }
    
    /* Check minimum */
    if (filter->min_market_cap > 0 && market_cap < filter->min_market_cap) {
        return false;
    }
    
    /* Check maximum */
    if (filter->max_market_cap > 0 && market_cap > filter->max_market_cap) {
        return false;
    }
    
    return true;
}

bool filter_check_liquidity(uint64_t liquidity, const filter_config_t *filter) {
    if (!filter) {
        return true;
    }
    
    if (filter->min_liquidity > 0 && liquidity < filter->min_liquidity) {
        return false;
    }
    
    return true;
}

bool filter_check_age(uint32_t age_seconds, const filter_config_t *filter) {
    if (!filter) {
        return true;
    }
    
    if (filter->max_age_seconds > 0 && age_seconds > filter->max_age_seconds) {
        return false;
    }
    
    return true;
}

bool filter_check_kol(uint8_t kol_count, const filter_config_t *filter) {
    if (!filter) {
        return true;
    }
    
    if (filter->min_kol_count > 0 && kol_count < filter->min_kol_count) {
        return false;
    }
    
    return true;
}

bool filter_check_holders(uint32_t holder_count, const filter_config_t *filter) {
    if (!filter) {
        return true;
    }
    
    if (filter->min_holder_count > 0 && holder_count < filter->min_holder_count) {
        return false;
    }
    
    return true;
}

bool filter_check_exchange(const char *exchange, const filter_config_t *filter) {
    if (!filter || !exchange) {
        return true;
    }
    
    /* If no exchanges specified, allow all */
    if (filter->exchange_count == 0) {
        return true;
    }
    
    /* Check if exchange is in whitelist */
    for (uint8_t i = 0; i < filter->exchange_count; i++) {
        if (strcasecmp(exchange, filter->exchanges[i]) == 0) {
            return true;
        }
    }
    
    return false;
}

bool filter_check_symbol(const char *symbol, const filter_config_t *filter) {
    if (!filter || !symbol) {
        return true;
    }
    
    /* Check if symbol contains any excluded terms */
    for (uint8_t i = 0; i < filter->exclude_count; i++) {
        if (strcasestr(symbol, filter->exclude_symbols[i]) != NULL) {
            return false;  /* Symbol contains excluded term */
        }
    }
    
    return true;
}

bool filter_check_pool(const pool_data_t *pool, const filter_config_t *filter) {
    if (!pool || !filter) {
        return true;
    }
    
    const token_info_t *token = &pool->base_token;
    
    /* Check symbol blacklist first (fastest check) */
    if (!filter_check_symbol(token->symbol, filter)) {
        return false;
    }
    
    /* Check exchange whitelist */
    if (!filter_check_exchange(pool->exchange, filter)) {
        return false;
    }
    
    /* Check market cap range */
    if (!filter_check_market_cap(token->market_cap, filter)) {
        return false;
    }
    
    /* Check liquidity */
    if (!filter_check_liquidity(pool->initial_liquidity, filter)) {
        return false;
    }
    
    /* Check 24h volume */
    if (filter->min_volume_24h > 0 && token->volume_24h < filter->min_volume_24h) {
        return false;
    }
    
    /* Check holder count */
    if (!filter_check_holders(token->holder_count, filter)) {
        return false;
    }
    
    /* Check token age */
    if (!filter_check_age(token->age_seconds, filter)) {
        return false;
    }
    
    /* Check KOL count */
    if (!filter_check_kol(token->kol_count, filter)) {
        return false;
    }
    
    /* Check top 10 holders concentration */
    if (filter->max_top_10_ratio > 0 && 
        token->top_10_ratio > filter->max_top_10_ratio) {
        return false;
    }
    
    /* Check creator balance ratio */
    if (filter->max_creator_ratio > 0 && 
        token->creator_balance_ratio > filter->max_creator_ratio) {
        return false;
    }
    
    return true;
}

int filter_add_exchange(filter_config_t *filter, const char *exchange) {
    if (!filter || !exchange) {
        return -1;
    }
    
    if (filter->exchange_count >= GMGN_MAX_EXCHANGES) {
        return -1;
    }
    
    strncpy(filter->exchanges[filter->exchange_count], exchange,
            GMGN_MAX_EXCHANGE_LEN - 1);
    filter->exchanges[filter->exchange_count][GMGN_MAX_EXCHANGE_LEN - 1] = '\0';
    filter->exchange_count++;
    
    return 0;
}

int filter_add_exclude_symbol(filter_config_t *filter, const char *symbol) {
    if (!filter || !symbol) {
        return -1;
    }
    
    if (filter->exclude_count >= GMGN_MAX_EXCLUDES) {
        return -1;
    }
    
    strncpy(filter->exclude_symbols[filter->exclude_count], symbol,
            GMGN_MAX_SYMBOL_LEN - 1);
    filter->exclude_symbols[filter->exclude_count][GMGN_MAX_SYMBOL_LEN - 1] = '\0';
    filter->exclude_count++;
    
    return 0;
}

int filter_get_summary(const filter_config_t *filter, char *buffer, 
                       size_t buffer_size) {
    if (!filter || !buffer || buffer_size == 0) {
        return -1;
    }
    
    int written = 0;
    int ret;
    
    /* Market cap */
    if (filter->min_market_cap > 0 || filter->max_market_cap > 0) {
        ret = snprintf(buffer + written, buffer_size - written,
                       "MC: $%.1fK-$%.1fK, ",
                       filter->min_market_cap / 100000.0,
                       filter->max_market_cap / 100000.0);
        if (ret > 0) written += ret;
    }
    
    /* Liquidity */
    if (filter->min_liquidity > 0) {
        ret = snprintf(buffer + written, buffer_size - written,
                       "Liq: >$%.1fK, ",
                       filter->min_liquidity / 100000.0);
        if (ret > 0) written += ret;
    }
    
    /* KOL */
    if (filter->min_kol_count > 0) {
        ret = snprintf(buffer + written, buffer_size - written,
                       "KOL: >=%d, ", filter->min_kol_count);
        if (ret > 0) written += ret;
    }
    
    /* Age */
    if (filter->max_age_seconds > 0) {
        ret = snprintf(buffer + written, buffer_size - written,
                       "Age: <%um, ", filter->max_age_seconds / 60);
        if (ret > 0) written += ret;
    }
    
    /* Holders */
    if (filter->min_holder_count > 0) {
        ret = snprintf(buffer + written, buffer_size - written,
                       "Holders: >=%u, ", filter->min_holder_count);
        if (ret > 0) written += ret;
    }
    
    /* Exchanges */
    if (filter->exchange_count > 0) {
        ret = snprintf(buffer + written, buffer_size - written, "Ex: ");
        if (ret > 0) written += ret;
        
        for (uint8_t i = 0; i < filter->exchange_count; i++) {
            ret = snprintf(buffer + written, buffer_size - written,
                           "%s%s", filter->exchanges[i],
                           (i < filter->exchange_count - 1) ? "," : "");
            if (ret > 0) written += ret;
        }
    }
    
    /* Remove trailing comma and space */
    if (written >= 2 && buffer[written - 2] == ',' && buffer[written - 1] == ' ') {
        buffer[written - 2] = '\0';
        written -= 2;
    }
    
    return written;
}
