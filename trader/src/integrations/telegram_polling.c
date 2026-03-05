/**
 * @file telegram_polling.c
 * @brief Telegram bot update polling and cleanup
 *
 * Implements long polling for Telegram bot updates (commands) and
 * bot cleanup. Parses incoming messages, extracts commands and
 * arguments, and dispatches to the registered callback.
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
#include <curl/curl.h>
#include <cjson/cJSON.h>

#include "telegram_bot.h"
#include "telegram_bot_internal.h"

int telegram_poll_updates(telegram_bot_t *bot) {
    char params[256];
    tg_curl_buffer_t response = {0};
    cJSON *root, *result, *update;
    int count = 0;

    if (!bot || !bot->initialized) {
        return -1;
    }

    /* Build request with offset */
    snprintf(params, sizeof(params),
             "{\"offset\":%lld,\"timeout\":%d,\"allowed_updates\":[\"message\"]}",
             (long long)(bot->last_update_id + 1), TELEGRAM_POLL_TIMEOUT);

    if (tg_api_request(bot, "getUpdates", params, &response) != 0) {
        free(response.data);
        return -1;
    }

    /* Parse response */
    root = cJSON_Parse(response.data);
    free(response.data);

    if (!root) return -1;

    result = cJSON_GetObjectItemCaseSensitive(root, "result");
    if (!result || !cJSON_IsArray(result)) {
        cJSON_Delete(root);
        return 0;
    }

    /* Process each update */
    cJSON_ArrayForEach(update, result) {
        cJSON *update_id = cJSON_GetObjectItemCaseSensitive(update, "update_id");
        cJSON *message = cJSON_GetObjectItemCaseSensitive(update, "message");

        if (update_id) {
            bot->last_update_id = update_id->valuedouble;
        }

        if (message && bot->command_callback) {
            cJSON *chat = cJSON_GetObjectItemCaseSensitive(message, "chat");
            cJSON *text = cJSON_GetObjectItemCaseSensitive(message, "text");

            /* Extract sender's chat_id */
            char sender_chat_id[64] = {0};
            if (chat) {
                cJSON *chat_id_json = cJSON_GetObjectItemCaseSensitive(chat, "id");
                if (chat_id_json && cJSON_IsNumber(chat_id_json)) {
                    snprintf(sender_chat_id, sizeof(sender_chat_id), "%lld",
                             (long long)chat_id_json->valuedouble);
                    /* Store for telegram_reply() */
                    strncpy(bot->last_chat_id, sender_chat_id,
                            sizeof(bot->last_chat_id) - 1);
                }
            }

            if (text && cJSON_IsString(text)) {
                const char *msg_text = text->valuestring;

                /* Check if it's a command */
                if (msg_text[0] == '/') {
                    char cmd[64] = {0};
                    char args[256] = {0};

                    /* Extract command and args */
                    const char *space = strchr(msg_text, ' ');
                    if (space) {
                        size_t cmd_len = space - msg_text;
                        if (cmd_len > sizeof(cmd) - 1) cmd_len = sizeof(cmd) - 1;
                        strncpy(cmd, msg_text, cmd_len);
                        strncpy(args, space + 1, sizeof(args) - 1);
                    } else {
                        strncpy(cmd, msg_text, sizeof(cmd) - 1);
                    }

                    bot->command_callback(sender_chat_id, cmd, args,
                                          bot->callback_user_data);
                }
            }
        }

        count++;
    }

    cJSON_Delete(root);
    return count;
}

void telegram_cleanup(telegram_bot_t *bot) {
    if (bot) {
        memset(bot, 0, sizeof(telegram_bot_t));
    }
}
