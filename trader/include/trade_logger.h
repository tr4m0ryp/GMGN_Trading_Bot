/**
 * @file trade_logger.h
 * @brief Structured logging system for AI trader
 *
 * Provides separate log files for different purposes:
 * - trader.log: General info and debug messages
 * - trades.csv: Transaction records for post-analysis
 * - errors.log: Error messages only
 *
 * Log files are organized by date in the logs directory.
 *
 * Dependencies: <stdbool.h>, <stdint.h>, <time.h>
 *
 * @date 2025-12-24
 */

#ifndef TRADE_LOGGER_H
#define TRADE_LOGGER_H

#include <stdbool.h>
#include <stdint.h>
#include <time.h>

/* Log configuration */
#define LOG_MAX_MSG_LEN         1024
#define LOG_MAX_PATH_LEN        512
#define LOG_FLUSH_INTERVAL_SEC  5

/**
 * @brief Log severity levels
 */
typedef enum {
    LOG_LEVEL_DEBUG = 0,
    LOG_LEVEL_INFO,
    LOG_LEVEL_WARN,
    LOG_LEVEL_ERROR
} log_level_t;

/**
 * @brief Trade record for transaction logging
 *
 * Contains all information needed for post-trade analysis.
 * Written to trades.csv in comma-separated format.
 */
typedef struct {
    time_t timestamp;               /* Unix timestamp of trade */
    char token_address[64];         /* Token mint address */
    char token_symbol[32];          /* Token symbol */
    char action[8];                 /* "BUY", "SELL", or "HOLD" */
    double entry_price;             /* Price at entry (for SELL records) */
    double exit_price;              /* Execution price */
    double amount_sol;              /* SOL amount traded */
    double pnl_sol;                 /* Profit/loss in SOL */
    double pnl_pct;                 /* Profit/loss percentage */
    double confidence;              /* Model confidence (0.0-1.0) */
    double balance_after;           /* Wallet balance after trade */
    bool is_paper;                  /* true if paper trade */
    int sequence_length;            /* Number of candles used for decision */
    char error_msg[256];            /* Error message if trade failed */
} trade_record_t;

/**
 * @brief Initialize logging system
 *
 * Creates log directory structure and opens log files.
 * Directory format: logs/YYYY-MM-DD/
 *
 * @param log_dir Base directory for logs
 *
 * @return 0 on success, -1 on failure
 */
int logger_init(const char *log_dir);

/**
 * @brief Set minimum log level
 *
 * Messages below this level will not be logged.
 *
 * @param level Minimum level to log
 */
void logger_set_level(log_level_t level);

/**
 * @brief Log debug message
 *
 * @param fmt Format string (printf-style)
 * @param ... Format arguments
 */
void log_debug(const char *fmt, ...);

/**
 * @brief Log info message
 *
 * @param fmt Format string (printf-style)
 * @param ... Format arguments
 */
void log_info(const char *fmt, ...);

/**
 * @brief Log warning message
 *
 * @param fmt Format string (printf-style)
 * @param ... Format arguments
 */
void log_warn(const char *fmt, ...);

/**
 * @brief Log error message
 *
 * Also written to errors.log for easy filtering.
 *
 * @param fmt Format string (printf-style)
 * @param ... Format arguments
 */
void log_error(const char *fmt, ...);

/**
 * @brief Log trade record to CSV
 *
 * Appends trade to trades.csv file for analysis.
 * Format: timestamp,address,symbol,action,entry_price,exit_price,
 *         amount,pnl_sol,pnl_pct,confidence,balance,is_paper,seq_len,error
 *
 * @param record Trade record to log
 *
 * @return 0 on success, -1 on failure
 */
int log_trade(const trade_record_t *record);

/**
 * @brief Flush all log buffers to disk
 *
 * Call periodically or before shutdown to ensure
 * all log data is written.
 */
void logger_flush(void);

/**
 * @brief Cleanup logging system
 *
 * Flushes buffers and closes all log files.
 */
void logger_cleanup(void);

/**
 * @brief Get current log directory path
 *
 * Returns the full path to today's log directory.
 *
 * @param buffer Output buffer for path
 * @param buffer_len Size of output buffer
 *
 * @return 0 on success, -1 if buffer too small
 */
int logger_get_current_dir(char *buffer, size_t buffer_len);

#endif /* TRADE_LOGGER_H */
