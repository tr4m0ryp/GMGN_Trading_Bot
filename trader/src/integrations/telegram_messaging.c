/**
 * @file telegram_messaging.c
 * @brief Telegram bot send operations
 *
 * Implements initialization, message sending, formatted messages,
 * trade notifications, error notifications, and reply functionality.
 * Contains the shared CURL write callback and API request helper.
 *
 * Dependencies: <curl/curl.h>, <cjson/cJSON.h>, "telegram_bot_internal.h"
 *
 * @date 2026-03-05
 */

/* Disable truncation warnings */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"
#pragma GCC diagnostic ignored "-Wstringop-truncation"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <curl/curl.h>
#include <cjson/cJSON.h>

#include "telegram_bot.h"
#include "telegram_bot_internal.h"

size_t tg_curl_write_cb(void *contents, size_t size, size_t nmemb, void *userp) {
    size_t realsize = size * nmemb;
    tg_curl_buffer_t *buf = (tg_curl_buffer_t *)userp;

    char *ptr = realloc(buf->data, buf->size + realsize + 1);
    if (!ptr) return 0;

    buf->data = ptr;
    memcpy(&buf->data[buf->size], contents, realsize);
    buf->size += realsize;
    buf->data[buf->size] = '\0';

    return realsize;
}

int tg_api_request(telegram_bot_t *bot, const char *method,
                   const char *params, tg_curl_buffer_t *response) {
    CURL *curl;
    CURLcode res;
    char url[512];
    struct curl_slist *headers = NULL;

    snprintf(url, sizeof(url), "%s%s/%s", TELEGRAM_API_URL, bot->token, method);

    curl = curl_easy_init();
    if (!curl) return -1;

    response->data = malloc(1);
    response->size = 0;

    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, tg_curl_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, TELEGRAM_POLL_TIMEOUT + 5);

    if (params) {
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, params);
    }

    res = curl_easy_perform(curl);

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    return (res == CURLE_OK) ? 0 : -1;
}

int telegram_init(telegram_bot_t *bot, const char *token, const char *chat_id) {
    if (!bot || !token) {
        return -1;
    }

    memset(bot, 0, sizeof(telegram_bot_t));
    strncpy(bot->token, token, sizeof(bot->token) - 1);

    /* chat_id is optional - if not provided, will use sender's chat_id */
    if (chat_id && chat_id[0] != '\0') {
        strncpy(bot->chat_id, chat_id, sizeof(bot->chat_id) - 1);
    }

    bot->last_update_id = 0;
    bot->initialized = true;

    return 0;
}

/**
 * @brief Send message to a specific chat
 *
 * @param bot      Bot instance
 * @param chat_id  Target chat ID
 * @param message  Message text (supports Markdown)
 *
 * @return 0 on success, -1 on error
 */
static int telegram_send_to_chat(telegram_bot_t *bot, const char *chat_id,
                                 const char *message) {
    char params[TELEGRAM_MAX_MSG_LEN + 256];
    tg_curl_buffer_t response = {0};
    cJSON *root;
    char *json_str;
    int ret;

    if (!bot || !bot->initialized || !message || !chat_id || chat_id[0] == '\0') {
        return -1;
    }

    /* Build JSON request */
    root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "chat_id", chat_id);
    cJSON_AddStringToObject(root, "text", message);
    cJSON_AddStringToObject(root, "parse_mode", "Markdown");

    json_str = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);

    if (!json_str) return -1;

    snprintf(params, sizeof(params), "%s", json_str);
    free(json_str);

    /* Make request */
    ret = tg_api_request(bot, "sendMessage", params, &response);
    free(response.data);

    return ret;
}

int telegram_send_message(telegram_bot_t *bot, const char *message) {
    char params[TELEGRAM_MAX_MSG_LEN + 256];
    tg_curl_buffer_t response = {0};
    cJSON *root;
    char *json_str;
    int ret;

    if (!bot || !bot->initialized || !message) {
        return -1;
    }

    /* Build JSON request */
    root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "chat_id", bot->chat_id);
    cJSON_AddStringToObject(root, "text", message);
    cJSON_AddStringToObject(root, "parse_mode", "Markdown");

    json_str = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);

    if (!json_str) return -1;

    snprintf(params, sizeof(params), "%s", json_str);
    free(json_str);

    /* Make request */
    ret = tg_api_request(bot, "sendMessage", params, &response);
    free(response.data);

    return ret;
}

int telegram_send_fmt(telegram_bot_t *bot, const char *fmt, ...) {
    char message[TELEGRAM_MAX_MSG_LEN];
    va_list args;

    va_start(args, fmt);
    vsnprintf(message, sizeof(message), fmt, args);
    va_end(args);

    return telegram_send_message(bot, message);
}

int telegram_notify_error(telegram_bot_t *bot, const char *error) {
    return telegram_send_fmt(bot, "WARNING *ERROR*\n```\n%s\n```", error);
}

int telegram_notify_trade(telegram_bot_t *bot, const char *action,
                          const char *symbol, double price,
                          double pnl, double pnl_pct) {
    if (strcmp(action, "BUY") == 0) {
        return telegram_send_fmt(bot,
            "[BUY] *%s* %s\n"
            "Price: `%.10f`",
            action, symbol, price);
    } else {
        const char *label = pnl >= 0 ? "PROFIT" : "LOSS";
        return telegram_send_fmt(bot,
            "[%s] *%s* %s\n"
            "Price: `%.10f`\n"
            "PnL: `%.6f SOL (%.2f%%)`",
            label, action, symbol, price, pnl, pnl_pct * 100);
    }
}

void telegram_set_command_callback(telegram_bot_t *bot,
                                   void (*callback)(const char *, const char *, const char *, void *),
                                   void *user_data) {
    if (!bot) return;
    bot->command_callback = callback;
    bot->callback_user_data = user_data;
}

int telegram_reply(telegram_bot_t *bot, const char *message) {
    if (!bot || !bot->initialized || bot->last_chat_id[0] == '\0') {
        return -1;
    }
    return telegram_send_to_chat(bot, bot->last_chat_id, message);
}

int telegram_reply_fmt(telegram_bot_t *bot, const char *fmt, ...) {
    char message[TELEGRAM_MAX_MSG_LEN];
    va_list args;

    va_start(args, fmt);
    vsnprintf(message, sizeof(message), fmt, args);
    va_end(args);

    return telegram_reply(bot, message);
}
