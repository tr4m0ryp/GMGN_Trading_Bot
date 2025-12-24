/**
 * @file config_loader.h
 * @brief Environment configuration loader for AI trader
 *
 * Parses .env file and provides configuration for trading operations.
 * Supports paper trading mode toggle, wallet settings, and Jito configuration.
 *
 * Dependencies: <stdbool.h>, <stdint.h>
 *
 * @date 2025-12-24
 */

#ifndef CONFIG_LOADER_H
#define CONFIG_LOADER_H

#include <stdbool.h>
#include <stdint.h>

/* Configuration limits */
#define CONFIG_MAX_KEY_LEN      64
#define CONFIG_MAX_VALUE_LEN    256
#define CONFIG_MAX_PATH_LEN     512

/**
 * @brief Trader configuration structure
 *
 * Loaded from .env file. Sensitive fields like wallet_private_key
 * should never be logged or printed.
 */
typedef struct {
    /* Trading Mode */
    bool paper_trade;               /* true = paper trading, false = live */

    /* Wallet Configuration */
    char wallet_private_key[128];   /* Base58 private key (live mode only) */

    /* Position Settings */
    double trade_amount_sol;        /* SOL amount per trade */
    int max_positions;              /* Max concurrent positions */

    /* Paper Trading */
    double paper_wallet_balance;    /* Initial paper balance */

    /* Risk Management */
    double take_profit_pct;         /* Take profit threshold (0.30 = 30%) */
    double stop_loss_pct;           /* Stop loss threshold (0.08 = 8%) */

    /* API Settings */
    char gmgn_cf_clearance[256];    /* Cloudflare clearance cookie */
    char gmgn_cf_bm[256];           /* Cloudflare BM cookie */
    char gmgn_ga[64];               /* Google Analytics cookie */
    char gmgn_ga_session[128];      /* GA session cookie */

    /* Jito Settings */
    char jito_endpoint[256];        /* Jito block engine URL */
    uint64_t jito_tip_lamports;     /* Tip amount in lamports */

    /* Telegram Bot Settings */
    char telegram_bot_token[128];   /* Bot API token */
    char telegram_chat_id[64];      /* Chat ID for notifications */
    bool telegram_notify_trades;    /* Notify on trades */
    bool telegram_notify_errors;    /* Notify on errors */
} trader_config_t;

/**
 * @brief Load configuration from .env file
 *
 * Parses the .env file and populates the config structure.
 * Missing optional values use defaults. Required values cause error if missing.
 *
 * @param config Pointer to config structure to populate
 * @param env_path Path to .env file
 *
 * @return 0 on success, -1 on file error, -2 on parse error
 */
int config_load(trader_config_t *config, const char *env_path);

/**
 * @brief Print configuration (masks sensitive data)
 *
 * Prints configuration values for debugging. Private key and cookies
 * are masked with asterisks for security.
 *
 * @param config Configuration to print
 */
void config_print(const trader_config_t *config);

/**
 * @brief Get default configuration
 *
 * Populates config with default values. Useful for testing
 * or when .env file is not available.
 *
 * @param config Pointer to config structure to populate
 */
void config_set_defaults(trader_config_t *config);

/**
 * @brief Validate configuration
 *
 * Checks that required fields are set and values are within valid ranges.
 *
 * @param config Configuration to validate
 *
 * @return 0 if valid, -1 if invalid (error message printed to stderr)
 */
int config_validate(const trader_config_t *config);

#endif /* CONFIG_LOADER_H */
