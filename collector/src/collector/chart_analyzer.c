/**
 * @file chart_analyzer.c
 * @brief Analyze chart data for token death conditions
 *
 * Parses candle data from the GMGN API response and checks for death
 * conditions: low volume, candle time gaps, and price stability. These
 * signals determine when a token should be written to CSV and removed
 * from active tracking.
 *
 * Dependencies: <cjson/cJSON.h>, "ai_data_collector_internal.h"
 *
 * @date 2025-12-20
 */

#include "ai_data_collector_internal.h"

/**
 * @brief Analyze chart data for death conditions
 *
 * Examines the last 3 candles in the chart data to detect inactivity:
 * - Volume below threshold
 * - Time gaps between candles exceeding threshold
 * - Price not moving beyond threshold
 *
 * @param json_data Raw JSON chart data
 * @param token Token structure to update with analysis
 *
 * @return Death reason if dead, AI_DEATH_NONE if still active
 */
ai_death_reason_t analyze_chart_data(const char *json_data,
                                     ai_tracked_token_t *token) {
    cJSON *json = NULL;
    cJSON *data = NULL;
    cJSON *list = NULL;
    int count = 0;
    double volume_sum = 0.0;
    double last_close = 0.0;
    double first_close = 0.0;
    time_t last_time = 0;
    time_t prev_time = 0;
    int gap_violations = 0;

    json = cJSON_Parse(json_data);
    if (!json) {
        return AI_DEATH_NONE;
    }

    data = cJSON_GetObjectItemCaseSensitive(json, "data");
    if (!data || !cJSON_IsObject(data)) {
        cJSON_Delete(json);
        return AI_DEATH_NONE;
    }

    list = cJSON_GetObjectItemCaseSensitive(data, "list");
    if (!list || !cJSON_IsArray(list)) {
        cJSON_Delete(json);
        return AI_DEATH_NONE;
    }

    count = cJSON_GetArraySize(list);
    if (count < 3) {
        cJSON_Delete(json);
        return AI_DEATH_NONE;
    }

    /* Analyze last 3 candles for volume and price */
    for (int i = count - 3; i < count; i++) {
        cJSON *candle = cJSON_GetArrayItem(list, i);
        if (!candle) {
            continue;
        }

        cJSON *volume = cJSON_GetObjectItemCaseSensitive(candle, "volume");
        cJSON *close = cJSON_GetObjectItemCaseSensitive(candle, "close");
        cJSON *time_val = cJSON_GetObjectItemCaseSensitive(candle, "time");

        if (volume) {
            if (cJSON_IsString(volume)) {
                volume_sum += atof(volume->valuestring);
            } else if (cJSON_IsNumber(volume)) {
                volume_sum += volume->valuedouble;
            }
        }

        if (close) {
            double close_val = 0.0;
            if (cJSON_IsString(close)) {
                close_val = atof(close->valuestring);
            } else if (cJSON_IsNumber(close)) {
                close_val = close->valuedouble;
            }

            if (i == count - 3) {
                first_close = close_val;
            }
            last_close = close_val;
        }

        if (time_val && cJSON_IsNumber(time_val)) {
            prev_time = last_time;
            last_time = (time_t)(time_val->valuedouble / 1000.0);

            if (prev_time > 0 && last_time > prev_time) {
                time_t gap = last_time - prev_time;
                if (gap > AI_CANDLE_GAP_THRESHOLD_SEC) {
                    gap_violations++;
                }
            }
        }
    }

    /* Update token tracking data */
    token->last_volume_avg = volume_sum / 3.0;
    token->last_close_price = last_close;
    token->last_candle_time = last_time;

    cJSON_Delete(json);

    /* Check time since last candle */
    time_t now = time(NULL);
    time_t since_last = now - last_time;

    if (since_last > AI_CANDLE_GAP_THRESHOLD_SEC) {
        gap_violations++;
    }

    /* Check death conditions */
    ai_death_reason_t reason = AI_DEATH_NONE;

    if (gap_violations >= 2) {
        reason = AI_DEATH_CANDLE_GAP;
    } else if (token->last_volume_avg < AI_VOLUME_THRESHOLD_SOL) {
        reason = AI_DEATH_VOLUME_LOW;
    } else if (first_close > 0.0 && last_close > 0.0) {
        double change = (last_close - first_close) / first_close;
        if (change > -AI_PRICE_CHANGE_THRESHOLD &&
            change < AI_PRICE_CHANGE_THRESHOLD) {
            reason = AI_DEATH_PRICE_STABLE;
        }
    }

    return reason;
}
