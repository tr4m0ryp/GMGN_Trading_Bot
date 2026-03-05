/**
 * @file json_helpers.c
 * @brief JSON accessor helpers, validation, and error handling
 *
 * Implements safe accessor functions for extracting typed values
 * from cJSON objects, plus JSON validation and error reporting.
 *
 * Dependencies: cJSON, "json_parser_internal.h"
 *
 * @date 2025-12-20
 */

#include "json_parser_internal.h"

/* Thread-local error message storage */
static __thread char s_last_error[256] = {0};

void json_set_error(const char *msg) {
    if (msg) {
        strncpy(s_last_error, msg, sizeof(s_last_error) - 1);
        s_last_error[sizeof(s_last_error) - 1] = '\0';
    }
}

void json_safe_get_string(const cJSON *json, const char *key,
                          char *dest, size_t dest_size) {
    if (!json || !key || !dest || dest_size == 0) {
        return;
    }

    const cJSON *item = cJSON_GetObjectItemCaseSensitive(json, key);
    if (cJSON_IsString(item) && item->valuestring) {
        strncpy(dest, item->valuestring, dest_size - 1);
        dest[dest_size - 1] = '\0';
    }
}

int64_t json_safe_get_int(const cJSON *json, const char *key, int64_t default_val) {
    if (!json || !key) {
        return default_val;
    }

    const cJSON *item = cJSON_GetObjectItemCaseSensitive(json, key);
    if (cJSON_IsNumber(item)) {
        return (int64_t)item->valuedouble;
    }
    return default_val;
}

double json_safe_get_double(const cJSON *json, const char *key, double default_val) {
    if (!json || !key) {
        return default_val;
    }

    const cJSON *item = cJSON_GetObjectItemCaseSensitive(json, key);
    if (cJSON_IsNumber(item)) {
        return item->valuedouble;
    }
    return default_val;
}

bool json_validate(const char *json_str, size_t json_len) {
    if (!json_str || json_len == 0) {
        return false;
    }

    cJSON *json = cJSON_ParseWithLength(json_str, json_len);
    if (json) {
        cJSON_Delete(json);
        return true;
    }

    return false;
}

const char *json_get_last_error(void) {
    return s_last_error[0] ? s_last_error : NULL;
}
