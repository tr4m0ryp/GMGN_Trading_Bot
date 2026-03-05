/**
 * @file csv_writer.c
 * @brief CSV export and candle data extraction for AI training data
 *
 * Handles writing token chart data to daily CSV files. Includes logic for
 * extracting clean candle data from raw GMGN API responses, stripping
 * unnecessary wrapper fields to produce compact training data.
 *
 * Output format per row:
 *   token_address,symbol,discovered_at_unix,discovered_age_sec,death_reason,candles
 *
 * Dependencies: <cjson/cJSON.h>, "ai_data_collector_internal.h"
 *
 * @date 2025-12-20
 */

#include "ai_data_collector_internal.h"

/**
 * @brief Escape string for CSV (handles quotes and newlines)
 */
void csv_escape(FILE *fp, const char *str) {
    fputc('"', fp);
    while (*str) {
        if (*str == '"') {
            fputs("\"\"", fp);
        } else {
            fputc(*str, fp);
        }
        str++;
    }
    fputc('"', fp);
}

/**
 * @brief Extract clean candle data from raw API response
 *
 * Parses the raw JSON and extracts only the candle list with essential fields:
 * time, open, high, low, close, volume
 *
 * This removes unnecessary wrapper fields like code, reason, message, _debug_tpool
 * to save storage space and simplify ML training data.
 *
 * @param raw_json Raw JSON response from API
 *
 * @return Newly allocated string with clean JSON array, or NULL on failure
 *         Caller must free the returned string
 */
char *extract_candle_data(const char *raw_json) {
    cJSON *root = NULL;
    cJSON *data = NULL;
    cJSON *list = NULL;
    cJSON *clean_array = NULL;
    cJSON *candle = NULL;
    char *result = NULL;

    if (!raw_json) {
        return NULL;
    }

    root = cJSON_Parse(raw_json);
    if (!root) {
        return NULL;
    }

    /* Navigate to data.list */
    data = cJSON_GetObjectItemCaseSensitive(root, "data");
    if (!data || !cJSON_IsObject(data)) {
        cJSON_Delete(root);
        return NULL;
    }

    list = cJSON_GetObjectItemCaseSensitive(data, "list");
    if (!list || !cJSON_IsArray(list)) {
        cJSON_Delete(root);
        return NULL;
    }

    /* Create clean array with only essential fields */
    clean_array = cJSON_CreateArray();
    if (!clean_array) {
        cJSON_Delete(root);
        return NULL;
    }

    cJSON_ArrayForEach(candle, list) {
        cJSON *clean_candle = cJSON_CreateObject();
        if (!clean_candle) {
            continue;
        }

        /* Extract time (convert ms to seconds) */
        cJSON *time_val = cJSON_GetObjectItemCaseSensitive(candle, "time");
        if (time_val && cJSON_IsNumber(time_val)) {
            cJSON_AddNumberToObject(clean_candle, "t",
                                    (double)((int64_t)(time_val->valuedouble / 1000.0)));
        }

        /* Extract OHLCV - convert strings to numbers for efficiency */
        cJSON *open = cJSON_GetObjectItemCaseSensitive(candle, "open");
        if (open) {
            double val = cJSON_IsString(open) ? atof(open->valuestring) : open->valuedouble;
            cJSON_AddNumberToObject(clean_candle, "o", val);
        }

        cJSON *high = cJSON_GetObjectItemCaseSensitive(candle, "high");
        if (high) {
            double val = cJSON_IsString(high) ? atof(high->valuestring) : high->valuedouble;
            cJSON_AddNumberToObject(clean_candle, "h", val);
        }

        cJSON *low = cJSON_GetObjectItemCaseSensitive(candle, "low");
        if (low) {
            double val = cJSON_IsString(low) ? atof(low->valuestring) : low->valuedouble;
            cJSON_AddNumberToObject(clean_candle, "l", val);
        }

        cJSON *close_field = cJSON_GetObjectItemCaseSensitive(candle, "close");
        if (close_field) {
            double val = cJSON_IsString(close_field) ?
                atof(close_field->valuestring) : close_field->valuedouble;
            cJSON_AddNumberToObject(clean_candle, "c", val);
        }

        cJSON *volume = cJSON_GetObjectItemCaseSensitive(candle, "volume");
        if (volume) {
            double val = cJSON_IsString(volume) ?
                atof(volume->valuestring) : volume->valuedouble;
            cJSON_AddNumberToObject(clean_candle, "v", val);
        }

        cJSON_AddItemToArray(clean_array, clean_candle);
    }

    /* Generate compact JSON string (no formatting) */
    result = cJSON_PrintUnformatted(clean_array);

    cJSON_Delete(clean_array);
    cJSON_Delete(root);

    return result;
}

/**
 * @brief Write token data to CSV file
 *
 * Extracts clean candle data from raw JSON before writing.
 * Output format: [{"t":timestamp,"o":open,"h":high,"l":low,"c":close,"v":volume},...]
 *
 * @param collector Collector state
 * @param token Token to write
 * @param death_reason Why the token was declared dead
 *
 * @return 0 on success, -1 on failure
 */
int write_token_to_csv(ai_data_collector_t *collector,
                       ai_tracked_token_t *token,
                       ai_death_reason_t death_reason) {
    char filepath[512];
    FILE *fp = NULL;
    struct stat st = {0};
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    bool write_header = false;
    char *clean_data = NULL;

    snprintf(filepath, sizeof(filepath),
        "%s/tokens_%04d-%02d-%02d.csv",
        collector->data_dir,
        tm_info->tm_year + 1900,
        tm_info->tm_mon + 1,
        tm_info->tm_mday);

    /* Check if file exists to determine if we need header */
    if (stat(filepath, &st) != 0) {
        write_header = true;
    }

    fp = fopen(filepath, "a");
    if (!fp) {
        fprintf(stderr, "[AI] Failed to open CSV: %s\n", strerror(errno));
        return -1;
    }

    if (write_header) {
        fprintf(fp, "token_address,symbol,discovered_at_unix,discovered_age_sec,"
                    "death_reason,candles\n");
    }

    /* Write row */
    fprintf(fp, "%s,", token->address);
    fprintf(fp, "%s,", token->symbol);
    fprintf(fp, "%ld,", (long)token->discovered_at);
    fprintf(fp, "%u,", token->discovered_age_sec);
    fprintf(fp, "%s,", ai_death_reason_str(death_reason));

    /* Extract clean candle data and write */
    if (token->chart_data) {
        clean_data = extract_candle_data(token->chart_data);
        if (clean_data) {
            csv_escape(fp, clean_data);
            free(clean_data);
        } else {
            fprintf(fp, "\"[]\"");
        }
    } else {
        fprintf(fp, "\"[]\"");
    }

    fprintf(fp, "\n");
    fclose(fp);

    return 0;
}
