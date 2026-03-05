/**
 * @file output_format.c
 * @brief Formatting helpers for market cap, age, and other values
 *
 * Implements human-readable formatting functions for monetary values
 * and time durations used in terminal output.
 *
 * Dependencies: "output_internal.h"
 *
 * @date 2025-12-20
 */

#include "output_internal.h"

char *output_format_market_cap(uint64_t market_cap_cents, char *buffer,
                               size_t buffer_size) {
    if (!buffer || buffer_size == 0) {
        return NULL;
    }

    double value = market_cap_cents / 100.0;

    if (value >= 1000000000.0) {
        snprintf(buffer, buffer_size, "$%.2fB", value / 1000000000.0);
    } else if (value >= 1000000.0) {
        snprintf(buffer, buffer_size, "$%.2fM", value / 1000000.0);
    } else if (value >= 1000.0) {
        snprintf(buffer, buffer_size, "$%.2fK", value / 1000.0);
    } else {
        snprintf(buffer, buffer_size, "$%.2f", value);
    }

    return buffer;
}

char *output_format_age(uint32_t age_seconds, char *buffer, size_t buffer_size) {
    if (!buffer || buffer_size == 0) {
        return NULL;
    }

    if (age_seconds >= 86400) {
        uint32_t days = age_seconds / 86400;
        uint32_t hours = (age_seconds % 86400) / 3600;
        snprintf(buffer, buffer_size, "%ud %uh", days, hours);
    } else if (age_seconds >= 3600) {
        uint32_t hours = age_seconds / 3600;
        uint32_t minutes = (age_seconds % 3600) / 60;
        snprintf(buffer, buffer_size, "%uh %um", hours, minutes);
    } else if (age_seconds >= 60) {
        uint32_t minutes = age_seconds / 60;
        uint32_t seconds = age_seconds % 60;
        snprintf(buffer, buffer_size, "%um %us", minutes, seconds);
    } else {
        snprintf(buffer, buffer_size, "%us", age_seconds);
    }

    return buffer;
}
