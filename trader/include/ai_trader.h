/**
 * @file ai_trader.h
 * @brief Main AI trader integration module
 *
 * Integrates all components:
 * - logger_c for new coin discovery
 * - Chart fetcher for real-time data
 * - ONNX model for inference
 * - Paper trader / Jito for execution
 *
 * Dependencies: All other ai_trader headers
 *
 * @date 2025-12-24
 */

#ifndef AI_TRADER_H
#define AI_TRADER_H

#include <stdbool.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>

#include "config_loader.h"
#include "chart_fetcher.h"
#include "paper_trader.h"
#include "trade_logger.h"

/* Trader limits */
#define TRADER_MAX_ACTIVE_TOKENS    32      /* Max tokens trading simultaneously */
#define TRADER_MIN_CANDLES          30      /* Minimum candles before trading */
#define TRADER_MODEL_CONFIDENCE     0.70    /* Minimum confidence to trade */

/**
 * @brief Trading signal from model
 */
typedef enum {
    SIGNAL_HOLD = 0,
    SIGNAL_BUY = 1,
    SIGNAL_SELL = 2
} trade_signal_t;

/**
 * @brief Model inference result
 */
typedef struct {
    trade_signal_t signal;          /* Predicted action */
    float confidence;               /* Confidence (0.0-1.0) */
    float probabilities[3];         /* [hold, buy, sell] probs */
    int64_t inference_time_us;      /* Inference time in microseconds */
} model_result_t;

/**
 * @brief Active token state
 */
typedef struct {
    char address[64];               /* Token mint address */
    char symbol[32];                /* Token symbol */
    chart_buffer_t chart;           /* Chart data */
    time_t discovered_at;           /* When added to trading */
    time_t last_action_at;          /* Last trade time */
    int trades_on_token;            /* Trades executed */
    double token_pnl;               /* PnL for this token */
    bool active;                    /* Slot in use */
} active_token_t;

/**
 * @brief AI trader state
 */
typedef struct {
    /* Configuration */
    trader_config_t config;

    /* Paper trading (if enabled) */
    paper_trader_t paper_trader;

    /* Active tokens */
    active_token_t tokens[TRADER_MAX_ACTIVE_TOKENS];
    int active_token_count;

    /* Threading */
    pthread_t polling_thread;       /* Chart polling thread */
    pthread_mutex_t lock;
    bool running;

    /* Statistics */
    int total_signals;              /* Total model predictions */
    int total_buys;                 /* Total buy signals */
    int total_sells;                /* Total sell signals */
    time_t start_time;              /* When trading started */

    /* ONNX model handle (opaque pointer) */
    void *onnx_session;
} ai_trader_t;

/**
 * @brief Initialize AI trader
 *
 * Loads configuration, initializes paper trader (if enabled),
 * and loads ONNX model.
 *
 * @param trader Trader instance to initialize
 * @param config_path Path to .env configuration file
 *
 * @return 0 on success, -1 on failure
 */
int ai_trader_init(ai_trader_t *trader, const char *config_path);

/**
 * @brief Start AI trader
 *
 * Starts background polling thread and begins trading.
 *
 * @param trader Initialized trader instance
 *
 * @return 0 on success, -1 on failure
 */
int ai_trader_start(ai_trader_t *trader);

/**
 * @brief Add token for trading
 *
 * Called when logger_c discovers a new token that passes filters.
 * Starts chart data collection and model inference.
 *
 * @param trader Running trader instance
 * @param address Token mint address
 * @param symbol Token symbol
 *
 * @return 0 on success, -1 if full or error
 */
int ai_trader_add_token(ai_trader_t *trader, const char *address,
                        const char *symbol);

/**
 * @brief Check if token is being traded
 *
 * @param trader Trader instance
 * @param address Token mint address
 *
 * @return true if token is active
 */
bool ai_trader_has_token(ai_trader_t *trader, const char *address);

/**
 * @brief Remove token from trading
 *
 * Closes any open positions and stops tracking.
 *
 * @param trader Trader instance
 * @param address Token mint address
 *
 * @return 0 on success, -1 if not found
 */
int ai_trader_remove_token(ai_trader_t *trader, const char *address);

/**
 * @brief Run model inference on token
 *
 * Extracts features from chart and runs ONNX model.
 * Inference target: < 0.5 seconds (user requirement).
 *
 * @param trader Trader instance
 * @param token Active token with chart data
 * @param result Output: model prediction
 *
 * @return 0 on success, -1 on inference error
 */
int ai_trader_infer(ai_trader_t *trader, active_token_t *token,
                    model_result_t *result);

/**
 * @brief Execute trade based on signal
 *
 * Routes to paper trader or Jito bundle based on config.
 *
 * @param trader Trader instance
 * @param token Active token
 * @param result Model inference result
 *
 * @return 0 on success, -1 on failure
 */
int ai_trader_execute(ai_trader_t *trader, active_token_t *token,
                      const model_result_t *result);

/**
 * @brief Stop AI trader
 *
 * Stops polling thread and closes positions.
 *
 * @param trader Running trader instance
 */
void ai_trader_stop(ai_trader_t *trader);

/**
 * @brief Print trading summary
 *
 * Outputs performance statistics to stdout.
 *
 * @param trader Trader instance
 */
void ai_trader_print_summary(ai_trader_t *trader);

/**
 * @brief Cleanup AI trader
 *
 * Releases all resources including ONNX session.
 *
 * @param trader Trader instance
 */
void ai_trader_cleanup(ai_trader_t *trader);

/* Callback for logger_c integration */

/**
 * @brief Callback when new token passes filters
 *
 * Called by logger_c when a token passes filter criteria.
 * Adds token to AI trader for analysis.
 *
 * @param address Token mint address
 * @param symbol Token symbol
 */
void ai_trader_on_new_token(const char *address, const char *symbol);

/**
 * @brief Set global trader instance for callbacks
 *
 * Must be called before starting logger_c.
 *
 * @param trader Trader instance
 */
void ai_trader_set_global(ai_trader_t *trader);

#endif /* AI_TRADER_H */
