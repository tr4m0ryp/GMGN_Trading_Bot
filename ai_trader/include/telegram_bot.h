/**
 * @file telegram_bot.h
 * @brief Telegram bot for notifications and commands
 *
 * Provides:
 * - Error notifications when something fails
 * - Commands for PnL, balance, and status
 * - Trade execution alerts
 *
 * Uses Telegram Bot HTTP API via libcurl.
 *
 * Dependencies: libcurl
 *
 * @date 2025-12-24
 */

#ifndef TELEGRAM_BOT_H
#define TELEGRAM_BOT_H

#include <stdbool.h>
#include <stdint.h>

/* Telegram API configuration */
#define TELEGRAM_API_URL        "https://api.telegram.org/bot"
#define TELEGRAM_MAX_MSG_LEN    4096
#define TELEGRAM_POLL_TIMEOUT   30      /* Long polling timeout seconds */

/**
 * @brief Telegram bot state
 */
typedef struct {
    char token[128];                    /* Bot API token */
    char chat_id[64];                   /* Chat ID for notifications */
    int64_t last_update_id;             /* Last processed update ID */
    bool initialized;                   /* Bot initialized */
    bool polling;                       /* Currently polling for commands */
    
    /* Callback for commands */
    void (*command_callback)(const char *command, const char *args, void *user_data);
    void *callback_user_data;
} telegram_bot_t;

/**
 * @brief Initialize Telegram bot
 *
 * @param bot Bot instance
 * @param token Bot API token from @BotFather
 * @param chat_id Chat ID to send notifications to
 *
 * @return 0 on success, -1 on error
 */
int telegram_init(telegram_bot_t *bot, const char *token, const char *chat_id);

/**
 * @brief Send a text message
 *
 * @param bot Bot instance
 * @param message Message text (supports Markdown)
 *
 * @return 0 on success, -1 on error
 */
int telegram_send_message(telegram_bot_t *bot, const char *message);

/**
 * @brief Send a formatted message (printf-style)
 *
 * @param bot Bot instance
 * @param fmt Format string
 * @param ... Format arguments
 *
 * @return 0 on success, -1 on error
 */
int telegram_send_fmt(telegram_bot_t *bot, const char *fmt, ...);

/**
 * @brief Send error notification
 *
 * Prefixes message with ⚠️ ERROR and sends to chat.
 *
 * @param bot Bot instance
 * @param error Error message
 *
 * @return 0 on success, -1 on error
 */
int telegram_notify_error(telegram_bot_t *bot, const char *error);

/**
 * @brief Send trade notification
 *
 * @param bot Bot instance
 * @param action "BUY" or "SELL"
 * @param symbol Token symbol
 * @param price Execution price
 * @param pnl PnL (for SELL only)
 * @param pnl_pct PnL percentage (for SELL only)
 *
 * @return 0 on success, -1 on error
 */
int telegram_notify_trade(telegram_bot_t *bot, const char *action,
                          const char *symbol, double price,
                          double pnl, double pnl_pct);

/**
 * @brief Set command callback
 *
 * Called when a command is received via polling.
 *
 * @param bot Bot instance
 * @param callback Function to call for commands
 * @param user_data Context to pass to callback
 */
void telegram_set_command_callback(telegram_bot_t *bot,
                                   void (*callback)(const char *, const char *, void *),
                                   void *user_data);

/**
 * @brief Poll for updates (commands)
 *
 * Uses long polling to check for new commands.
 * Should be called in a loop or thread.
 *
 * @param bot Bot instance
 *
 * @return Number of updates processed, or -1 on error
 */
int telegram_poll_updates(telegram_bot_t *bot);

/**
 * @brief Cleanup bot resources
 *
 * @param bot Bot instance
 */
void telegram_cleanup(telegram_bot_t *bot);

#endif /* TELEGRAM_BOT_H */
