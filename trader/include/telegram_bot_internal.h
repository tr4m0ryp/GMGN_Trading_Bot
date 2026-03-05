/**
 * @file telegram_bot_internal.h
 * @brief Internal shared state for Telegram bot split files
 *
 * Exposes CURL buffer type, write callback, and API request helper shared
 * between telegram_messaging.c (send ops) and telegram_polling.c (polling).
 *
 * Dependencies: <curl/curl.h>, "telegram_bot.h"
 *
 * @date 2026-03-05
 */

#ifndef TELEGRAM_BOT_INTERNAL_H
#define TELEGRAM_BOT_INTERNAL_H

#include <curl/curl.h>

#include "telegram_bot.h"

/* CURL response buffer */
typedef struct {
    char *data;
    size_t size;
} tg_curl_buffer_t;

/**
 * @brief CURL write callback for Telegram bot
 */
size_t tg_curl_write_cb(void *contents, size_t size, size_t nmemb, void *userp);

/**
 * @brief Make Telegram API request
 *
 * Sends an HTTP request to the Telegram Bot API.
 *
 * @param bot       Bot instance with token
 * @param method    API method name (e.g., "sendMessage")
 * @param params    JSON-encoded parameters (or NULL for GET)
 * @param response  Output buffer for response body
 *
 * @return 0 on success, -1 on CURL failure
 */
int tg_api_request(telegram_bot_t *bot, const char *method,
                   const char *params, tg_curl_buffer_t *response);

#endif /* TELEGRAM_BOT_INTERNAL_H */
