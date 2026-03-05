/**
 * @file ai_data_collector.h
 * @brief AI training data collection from token chart data
 *
 * This module collects chart data for tokens that pass filters and writes
 * to CSV when the token becomes inactive ("dead"). The data is used for
 * training AI models to predict token behavior.
 *
 * Death detection criteria:
 * - Time gap between candles exceeds threshold
 * - Volume drops below threshold
 * - Price change below threshold
 * - Consecutive dead checks required
 *
 * Dependencies: <stdint.h>, <stdbool.h>, <time.h>, <pthread.h>, <curl/curl.h>
 *
 * @date 2025-12-20
 */

#ifndef AI_DATA_COLLECTOR_H
#define AI_DATA_COLLECTOR_H

#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include <pthread.h>

/* Death detection configuration */
#define AI_MIN_TRACK_TIME_SEC       60      /* Min seconds before can be dead */
#define AI_MAX_TRACK_TIME_SEC       600     /* Max seconds (10 min) force write */
#define AI_CANDLE_GAP_THRESHOLD_SEC 30      /* Candle gap indicating inactivity */
#define AI_VOLUME_THRESHOLD_SOL     10.0    /* Min SOL volume (avg last 3) */
#define AI_PRICE_CHANGE_THRESHOLD   0.02    /* Min price change (2%) */
#define AI_CONSECUTIVE_DEAD_CHECKS  3       /* Checks before confirmed dead */
#define AI_POLL_INTERVAL_MS         5000    /* Chart poll interval (5 sec) */

/* Collector limits */
#define AI_MAX_TRACKED_TOKENS       64      /* Max concurrent tracked tokens */
#define AI_MAX_ADDRESS_LEN          64      /* Token address buffer size */
#define AI_MAX_SYMBOL_LEN           32      /* Token symbol buffer size */
#define AI_MAX_CHART_DATA_LEN       131072  /* Max chart JSON size (128KB) */

/* API endpoint */
#define AI_MCAP_CANDLES_API "https://gmgn.ai/api/v1/token_mcap_candles/sol/"

/**
 * @brief Death reason for token
 */
typedef enum {
    AI_DEATH_NONE = 0,          /* Still active */
    AI_DEATH_VOLUME_LOW,        /* Volume dropped below threshold */
    AI_DEATH_CANDLE_GAP,        /* Time gap between candles too large */
    AI_DEATH_PRICE_STABLE,      /* Price not moving */
    AI_DEATH_TIMEOUT,           /* Max tracking time exceeded */
    AI_DEATH_API_FAIL           /* API consistently failing */
} ai_death_reason_t;

/**
 * @brief Tracked token for AI data collection
 */
typedef struct {
    char address[AI_MAX_ADDRESS_LEN];   /* Token mint address */
    char symbol[AI_MAX_SYMBOL_LEN];     /* Token symbol */
    time_t discovered_at;               /* Unix timestamp when discovered */
    uint32_t discovered_age_sec;        /* Age reported by WebSocket */
    time_t last_poll_at;                /* Last chart poll time */
    uint32_t dead_check_count;          /* Consecutive dead signals */
    double last_volume_avg;             /* Avg volume of last 3 candles */
    double last_close_price;            /* Last close price */
    time_t last_candle_time;            /* Time of last candle received */
    char *chart_data;                   /* JSON chart data (allocated) */
    size_t chart_data_len;              /* Current chart data length */
    bool active;                        /* Slot in use */
} ai_tracked_token_t;

/**
 * @brief AI data collector state
 */
typedef struct {
    ai_tracked_token_t tokens[AI_MAX_TRACKED_TOKENS];
    pthread_mutex_t lock;               /* Thread safety */
    pthread_t thread;                   /* Background collector thread */
    bool running;                       /* Thread running flag */
    char data_dir[256];                 /* Output directory for CSV */
    uint32_t tokens_collected;          /* Total tokens written to CSV */
    uint32_t tokens_active;             /* Currently tracking */
} ai_data_collector_t;

/**
 * @brief Initialize the AI data collector
 *
 * Sets up the collector state and prepares for tracking tokens.
 * Does not start the background thread - call ai_collector_start().
 *
 * @param collector Pointer to collector structure to initialize
 * @param data_dir Directory path for CSV output (will be created if needed)
 *
 * @return 0 on success, -1 on failure
 */
int ai_collector_init(ai_data_collector_t *collector, const char *data_dir);

/**
 * @brief Start the AI data collector background thread
 *
 * Starts the background thread that polls chart data and writes CSV.
 *
 * @param collector Initialized collector structure
 *
 * @return 0 on success, -1 on failure
 */
int ai_collector_start(ai_data_collector_t *collector);

/**
 * @brief Stop the AI data collector
 *
 * Stops the background thread and writes any remaining data to CSV.
 *
 * @param collector Running collector structure
 */
void ai_collector_stop(ai_data_collector_t *collector);

/**
 * @brief Cleanup the AI data collector
 *
 * Frees all resources. Call ai_collector_stop() first.
 *
 * @param collector Collector structure to cleanup
 */
void ai_collector_cleanup(ai_data_collector_t *collector);

/**
 * @brief Add a token for AI data tracking
 *
 * Called when a token passes filters. Starts collecting chart data.
 *
 * @param collector Active collector structure
 * @param address Token mint address
 * @param symbol Token symbol
 * @param discovered_age_sec Age reported by WebSocket at discovery
 *
 * @return 0 on success, -1 if full or error
 */
int ai_collector_add_token(ai_data_collector_t *collector,
                           const char *address,
                           const char *symbol,
                           uint32_t discovered_age_sec);

/**
 * @brief Get count of actively tracked tokens
 *
 * @param collector Collector structure
 *
 * @return Number of tokens currently being tracked
 */
uint32_t ai_collector_get_active_count(ai_data_collector_t *collector);

/**
 * @brief Get total tokens collected to CSV
 *
 * @param collector Collector structure
 *
 * @return Total tokens written to CSV since start
 */
uint32_t ai_collector_get_total_collected(ai_data_collector_t *collector);

/**
 * @brief Convert death reason to string
 *
 * @param reason Death reason enum value
 *
 * @return Human-readable string for the death reason
 */
const char *ai_death_reason_str(ai_death_reason_t reason);

#endif /* AI_DATA_COLLECTOR_H */
