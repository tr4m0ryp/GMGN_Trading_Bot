/**
 * @file chart_access.c
 * @brief Chart data access and feature extraction
 *
 * Provides functions for accessing candle data from the chart buffer
 * and extracting normalized features for ONNX model inference.
 *
 * Dependencies: <string.h>, "chart_fetcher.h"
 *
 * @date 2026-03-05
 */

#include <stdlib.h>
#include <string.h>

#include "chart_fetcher.h"

void chart_buffer_init(chart_buffer_t *buffer, const char *address,
                       const char *symbol) {
    if (!buffer) {
        return;
    }

    memset(buffer, 0, sizeof(chart_buffer_t));

    if (address) {
        strncpy(buffer->token_address, address, sizeof(buffer->token_address) - 1);
    }
    if (symbol) {
        strncpy(buffer->token_symbol, symbol, sizeof(buffer->token_symbol) - 1);
    }
}

const candle_t *chart_get_latest(const chart_buffer_t *buffer) {
    if (!buffer || buffer->candle_count == 0) {
        return NULL;
    }
    return &buffer->candles[buffer->candle_count - 1];
}

const candle_t *chart_get_candle(const chart_buffer_t *buffer, int index) {
    if (!buffer || index < 0 || index >= buffer->candle_count) {
        return NULL;
    }
    return &buffer->candles[index];
}

double chart_get_price(const chart_buffer_t *buffer) {
    const candle_t *latest = chart_get_latest(buffer);
    return latest ? latest->close : 0.0;
}

double chart_get_price_change(const chart_buffer_t *buffer, int lookback) {
    if (!buffer || buffer->candle_count < 2) {
        return 0.0;
    }

    int start_idx = buffer->candle_count - 1 - lookback;
    if (start_idx < 0) {
        start_idx = 0;
    }

    double first_price = buffer->candles[start_idx].close;
    double last_price = buffer->candles[buffer->candle_count - 1].close;

    if (first_price <= 0.0) {
        return 0.0;
    }

    return (last_price - first_price) / first_price;
}

int chart_extract_features(const chart_buffer_t *buffer, float *features,
                           int max_features, int *out_length) {
    if (!buffer || !features || !out_length) {
        return -1;
    }

    int num_candles = buffer->candle_count;
    int features_per_candle = 5;  /* OHLCV */
    int total_features = num_candles * features_per_candle;

    if (total_features > max_features) {
        total_features = max_features;
        num_candles = max_features / features_per_candle;
    }

    /* Normalize prices relative to first candle */
    double base_price = buffer->candles[0].close;
    if (base_price <= 0.0) {
        base_price = 1.0;
    }

    int feat_idx = 0;
    for (int i = 0; i < num_candles && feat_idx < max_features; i++) {
        const candle_t *c = &buffer->candles[i];

        /* Normalize to base price (log returns would be better for ONNX) */
        features[feat_idx++] = (float)(c->open / base_price);
        features[feat_idx++] = (float)(c->high / base_price);
        features[feat_idx++] = (float)(c->low / base_price);
        features[feat_idx++] = (float)(c->close / base_price);
        features[feat_idx++] = (float)(c->volume);  /* Volume as-is for now */
    }

    *out_length = feat_idx;
    return 0;
}
