/**
 * @file gmgn_types.h
 * @brief Core type definitions for GMGN Trenches logger
 *
 * This header defines all data structures used throughout the GMGN
 * token monitoring system including token info, pool data, filter
 * configuration, and WebSocket message types.
 *
 * Dependencies: <stdint.h>, <stdbool.h>, <time.h>
 *
 * @date 2025-12-20
 */

#ifndef GMGN_TYPES_H
#define GMGN_TYPES_H

#include <stdint.h>
#include <stdbool.h>
#include <time.h>

/* Maximum string lengths */
#define GMGN_MAX_SYMBOL_LEN      16
#define GMGN_MAX_NAME_LEN        64
#define GMGN_MAX_ADDRESS_LEN     64
#define GMGN_MAX_EXCHANGE_LEN    32
#define GMGN_MAX_CHAIN_LEN       16
#define GMGN_MAX_EXCHANGES       8
#define GMGN_MAX_EXCLUDES        16
#define GMGN_MAX_URL_LEN         256
#define GMGN_MAX_TOKEN_LEN       128

/* Price fixed-point multiplier (1e-8 precision) */
#define GMGN_PRICE_MULTIPLIER    100000000ULL

/* Percentage fixed-point multiplier (0.01% precision) */
#define GMGN_PERCENT_MULTIPLIER  10000

/**
 * @brief Token information structure
 *
 * Contains all relevant token metadata received from GMGN WebSocket.
 * Monetary values are stored in USD cents for integer arithmetic.
 * Percentages are stored as fixed-point (0-10000 = 0-100%).
 */
typedef struct {
    char symbol[GMGN_MAX_SYMBOL_LEN];          /* Token symbol (e.g., "SOL") */
    char name[GMGN_MAX_NAME_LEN];              /* Token full name */
    char address[GMGN_MAX_ADDRESS_LEN];        /* Token contract address */
    uint64_t market_cap;                        /* Market cap in USD cents */
    uint64_t volume_24h;                        /* 24h volume in USD cents */
    uint64_t price;                             /* Price (fixed-point 1e-8) */
    uint32_t holder_count;                      /* Number of token holders */
    uint16_t top_10_ratio;                      /* Top 10 holders % (0-10000) */
    uint16_t creator_balance_ratio;             /* Creator balance % (0-10000) */
    uint32_t age_seconds;                       /* Token age in seconds */
    uint8_t kol_count;                          /* KOL/influencer count */
    time_t created_at;                          /* Creation timestamp */
} token_info_t;

/**
 * @brief Pool data structure
 *
 * Represents a liquidity pool with associated token information.
 */
typedef struct {
    char pool_address[GMGN_MAX_ADDRESS_LEN];   /* Pool contract address */
    char base_token_addr[GMGN_MAX_ADDRESS_LEN];/* Base token address */
    char quote_token_addr[GMGN_MAX_ADDRESS_LEN];/* Quote token address */
    char exchange[GMGN_MAX_EXCHANGE_LEN];      /* Exchange name */
    char chain[GMGN_MAX_CHAIN_LEN];            /* Chain identifier */
    uint64_t initial_liquidity;                 /* Initial liquidity (USD cents) */
    token_info_t base_token;                    /* Base token info */
    time_t created_at;                          /* Creation timestamp */
} pool_data_t;

/**
 * @brief Filter configuration structure
 *
 * Configurable filter criteria for token screening.
 * Set values to 0 to disable specific filters.
 */
typedef struct {
    /* Market cap range (USD cents, 0 = no limit) */
    uint64_t min_market_cap;
    uint64_t max_market_cap;

    /* Liquidity and volume minimums (USD cents) */
    uint64_t min_liquidity;
    uint64_t min_volume_24h;

    /* Holder requirements */
    uint32_t min_holder_count;
    uint16_t max_top_10_ratio;                  /* Max concentration (0-10000) */
    uint16_t max_creator_ratio;                 /* Max creator balance (0-10000) */

    /* Token age (seconds, 0 = no limit) */
    uint32_t max_age_seconds;

    /* KOL/influencer minimum */
    uint8_t min_kol_count;

    /* Exchange whitelist */
    char exchanges[GMGN_MAX_EXCHANGES][GMGN_MAX_EXCHANGE_LEN];
    uint8_t exchange_count;

    /* Symbol blacklist */
    char exclude_symbols[GMGN_MAX_EXCLUDES][GMGN_MAX_SYMBOL_LEN];
    uint8_t exclude_count;
} filter_config_t;

/**
 * @brief Application configuration structure
 */
typedef struct {
    char websocket_url[GMGN_MAX_URL_LEN];      /* WebSocket endpoint */
    char access_token[GMGN_MAX_TOKEN_LEN];     /* GMGN auth token (optional) */
    char chain[GMGN_MAX_CHAIN_LEN];            /* Target chain (sol, eth, etc.) */
    uint32_t reconnect_attempts;                /* Max reconnect attempts */
    uint32_t reconnect_delay_ms;                /* Initial reconnect delay */
    uint32_t heartbeat_interval_ms;             /* Heartbeat ping interval */
    filter_config_t filter;                     /* Filter configuration */
    int verbosity;                              /* Output verbosity (0-2) */
} app_config_t;

/**
 * @brief WebSocket message types
 */
typedef enum {
    GMGN_MSG_UNKNOWN = 0,
    GMGN_MSG_NEW_POOL,
    GMGN_MSG_PAIR_UPDATE,
    GMGN_MSG_TOKEN_LAUNCH,
    GMGN_MSG_CHAIN_STATS,
    GMGN_MSG_WALLET_TRADE,
    GMGN_MSG_ERROR,
    GMGN_MSG_PONG
} gmgn_msg_type_t;

/**
 * @brief Connection state
 */
typedef enum {
    GMGN_STATE_DISCONNECTED = 0,
    GMGN_STATE_CONNECTING,
    GMGN_STATE_CONNECTED,
    GMGN_STATE_SUBSCRIBING,
    GMGN_STATE_ACTIVE,
    GMGN_STATE_RECONNECTING,
    GMGN_STATE_ERROR
} gmgn_conn_state_t;

/**
 * @brief Callback function type for new pool events
 *
 * @param pool Pointer to pool data
 * @param user_data User-provided context pointer
 */
typedef void (*pool_callback_fn)(const pool_data_t *pool, void *user_data);

/**
 * @brief Callback function type for pair update events
 *
 * @param pool Pointer to pool/pair data
 * @param user_data User-provided context pointer
 */
typedef void (*pair_update_callback_fn)(const pool_data_t *pool, void *user_data);

/**
 * @brief Callback function type for token launch events
 *
 * @param pool Pointer to pool/token data
 * @param user_data User-provided context pointer
 */
typedef void (*token_launch_callback_fn)(const pool_data_t *pool, void *user_data);

/**
 * @brief Callback function type for errors
 *
 * @param error_code Error code
 * @param error_msg Error message string
 * @param user_data User-provided context pointer
 */
typedef void (*error_callback_fn)(int error_code, const char *error_msg,
                                   void *user_data);

#endif /* GMGN_TYPES_H */
