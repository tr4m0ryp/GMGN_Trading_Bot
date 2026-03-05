/**
 * @file chart_fetcher.h
 * @brief Real-time chart data fetcher from GMGN API
 *
 * Fetches candlestick data at 1-second resolution for active tokens.
 * Uses persistent CURL handle for efficient polling.
 *
 * API endpoint:
 * https://gmgn.ai/api/v1/token_candles/sol/{address}?resolution=1m&limit=501
 *
 * Dependencies: <stdbool.h>, <stdint.h>, <time.h>
 *
 * @date 2025-12-24
 */

#ifndef CHART_FETCHER_H
#define CHART_FETCHER_H

#include <stdbool.h>
#include <stdint.h>
#include <time.h>

#include "config_loader.h"

/* Fetcher configuration */
#define CHART_POLL_INTERVAL_MS  1000    /* 1 second polling */
#define CHART_API_TIMEOUT_SEC   2       /* Fast timeout for trading */
#define CHART_MAX_CANDLES       512     /* Maximum candles to store */

/* API base URL */
#define CHART_API_BASE "https://gmgn.ai/api/v1/token_candles/sol/"

/**
 * @brief Single candle (OHLCV) data
 */
typedef struct {
    time_t timestamp;           /* Unix timestamp */
    double open;                /* Open price */
    double high;                /* High price */
    double low;                 /* Low price */
    double close;               /* Close price */
    double volume;              /* Volume in SOL */
} candle_t;

/**
 * @brief Chart buffer for a token
 *
 * Maintains rolling window of candle data for a single token.
 */
typedef struct {
    char token_address[64];     /* Token mint address */
    char token_symbol[32];      /* Token symbol */
    candle_t candles[CHART_MAX_CANDLES];
    int candle_count;           /* Current number of candles */
    time_t first_candle_time;   /* Earliest candle timestamp */
    time_t last_candle_time;    /* Latest candle timestamp */
    time_t last_fetch_time;     /* Last successful API fetch */
    int fetch_failures;         /* Consecutive fetch failures */
} chart_buffer_t;

/**
 * @brief Fetch result status
 */
typedef enum {
    FETCH_SUCCESS = 0,          /* Data fetched successfully */
    FETCH_ERROR_CURL,           /* CURL error */
    FETCH_ERROR_HTTP,           /* Non-200 HTTP response */
    FETCH_ERROR_PARSE,          /* JSON parse error */
    FETCH_ERROR_EMPTY,          /* No candle data */
    FETCH_ERROR_TIMEOUT         /* Request timeout */
} fetch_status_t;

/**
 * @brief Initialize chart fetcher
 *
 * Sets up CURL handle with cookies from config.
 * Must be called before fetching any charts.
 *
 * @param config Trader configuration with API cookies
 *
 * @return 0 on success, -1 on failure
 */
int chart_fetcher_init(const trader_config_t *config);

/**
 * @brief Fetch chart data for token
 *
 * Retrieves latest candles from GMGN API and updates buffer.
 * New candles are appended; existing candles preserved.
 *
 * @param address Token mint address
 * @param buffer Chart buffer to update
 *
 * @return fetch_status_t indicating result
 */
fetch_status_t chart_fetch(const char *address, chart_buffer_t *buffer);

/**
 * @brief Initialize chart buffer
 *
 * Clears buffer and sets token address.
 *
 * @param buffer Buffer to initialize
 * @param address Token mint address
 * @param symbol Token symbol
 */
void chart_buffer_init(chart_buffer_t *buffer, const char *address,
                       const char *symbol);

/**
 * @brief Get latest candle
 *
 * @param buffer Chart buffer
 *
 * @return Pointer to latest candle, or NULL if empty
 */
const candle_t *chart_get_latest(const chart_buffer_t *buffer);

/**
 * @brief Get candle at index
 *
 * @param buffer Chart buffer
 * @param index Index (0 = oldest, count-1 = newest)
 *
 * @return Pointer to candle, or NULL if out of range
 */
const candle_t *chart_get_candle(const chart_buffer_t *buffer, int index);

/**
 * @brief Get current price (last close)
 *
 * @param buffer Chart buffer
 *
 * @return Last close price, or 0.0 if no data
 */
double chart_get_price(const chart_buffer_t *buffer);

/**
 * @brief Calculate price change percentage
 *
 * Computes (last_close - first_close) / first_close.
 *
 * @param buffer Chart buffer
 * @param lookback Number of candles to look back
 *
 * @return Price change as decimal (0.10 = 10% gain)
 */
double chart_get_price_change(const chart_buffer_t *buffer, int lookback);

/**
 * @brief Extract features for model input
 *
 * Extracts feature array from candle history for ONNX model inference.
 * Features: [open, high, low, close, volume] per candle, normalized.
 *
 * @param buffer Chart buffer
 * @param features Output array (must be large enough)
 * @param max_features Maximum features to extract
 * @param out_length Output: actual features extracted
 *
 * @return 0 on success, -1 on failure
 */
int chart_extract_features(const chart_buffer_t *buffer, float *features,
                           int max_features, int *out_length);

/**
 * @brief Cleanup chart fetcher
 *
 * Releases CURL resources.
 */
void chart_fetcher_cleanup(void);

/**
 * @brief Convert fetch status to string
 *
 * @param status Fetch status enum
 *
 * @return Human-readable string
 */
const char *fetch_status_str(fetch_status_t status);

#endif /* CHART_FETCHER_H */
