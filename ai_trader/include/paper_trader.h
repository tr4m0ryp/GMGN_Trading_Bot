/**
 * @file paper_trader.h
 * @brief Paper trading simulation with full fee modeling
 *
 * Simulates trading without real money, accurately modeling:
 * - Jito tips and gas fees (~7% per round trip)
 * - Transaction delays (1 second Jito confirmation)
 * - Virtual balance and position tracking
 *
 * All trades are logged for analysis and model validation.
 *
 * Dependencies: <stdbool.h>, <stdint.h>, <time.h>, <pthread.h>
 *
 * @date 2025-12-24
 */

#ifndef PAPER_TRADER_H
#define PAPER_TRADER_H

#include <stdbool.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>

/* Fee structure (Jito infrastructure) */
#define PAPER_JITO_TIP_SOL      0.00005     /* Average Jito tip */
#define PAPER_GAS_FEE_SOL       0.0002      /* Solana gas fee */
#define PAPER_PRIORITY_FEE_SOL  0.0001      /* Priority fee */
#define PAPER_TOTAL_FEE_PER_TX  0.00035     /* Total per transaction */
#define PAPER_FEE_ROUND_TRIP    0.0007      /* Total for buy + sell (~7%) */

/* Limits */
#define PAPER_MAX_POSITIONS     32          /* Max concurrent positions */

/**
 * @brief Open position tracking
 */
typedef struct {
    char token_address[64];         /* Token mint address */
    char token_symbol[32];          /* Token symbol */
    double entry_price;             /* Price at buy */
    double amount_sol;              /* SOL invested */
    double tokens_held;             /* Tokens received (amount / price) */
    time_t entry_time;              /* When position opened */
    int entry_seq_length;           /* Candles seen at entry */
    double confidence;              /* Model confidence at entry */
    bool active;                    /* Slot in use */
} paper_position_t;

/**
 * @brief Paper trading state
 */
typedef struct {
    /* Balance tracking */
    double initial_balance;         /* Starting balance */
    double current_balance;         /* Current available SOL */
    double total_invested;          /* SOL in open positions */

    /* Position tracking */
    paper_position_t positions[PAPER_MAX_POSITIONS];
    int position_count;             /* Active positions */

    /* Statistics */
    int total_trades;               /* Total completed trades */
    int winning_trades;             /* Trades with profit > 0 */
    double total_pnl;               /* Cumulative profit/loss */
    double total_fees_paid;         /* Total fees simulated */
    double best_trade_pnl;          /* Best single trade */
    double worst_trade_pnl;         /* Worst single trade */

    /* Thread safety */
    pthread_mutex_t lock;
} paper_trader_t;

/**
 * @brief Trade result from paper execution
 */
typedef struct {
    bool success;                   /* true if trade executed */
    double executed_price;          /* Actual execution price */
    double fee_paid;                /* Transaction fee */
    double pnl;                     /* Profit/loss (SELL only) */
    double pnl_pct;                 /* PnL percentage (SELL only) */
    double new_balance;             /* Balance after trade */
    char error[128];                /* Error message if failed */
} paper_trade_result_t;

/**
 * @brief Initialize paper trader
 *
 * @param pt Paper trader instance
 * @param initial_balance Starting SOL balance
 *
 * @return 0 on success, -1 on failure
 */
int paper_trader_init(paper_trader_t *pt, double initial_balance);

/**
 * @brief Execute paper buy
 *
 * Simulates buying tokens with fee deduction.
 * Fails if insufficient balance.
 *
 * @param pt Paper trader instance
 * @param address Token mint address
 * @param symbol Token symbol
 * @param price Current token price
 * @param amount_sol SOL to invest
 * @param confidence Model confidence (0.0-1.0)
 * @param seq_length Candles used for decision
 * @param result Output: trade result
 *
 * @return 0 on success, -1 on failure
 */
int paper_buy(paper_trader_t *pt, const char *address, const char *symbol,
              double price, double amount_sol, double confidence,
              int seq_length, paper_trade_result_t *result);

/**
 * @brief Execute paper sell
 *
 * Closes position for token with fee deduction.
 * Fails if no position exists.
 *
 * @param pt Paper trader instance
 * @param address Token mint address
 * @param price Current token price
 * @param result Output: trade result with PnL
 *
 * @return 0 on success, -1 on failure
 */
int paper_sell(paper_trader_t *pt, const char *address, double price,
               paper_trade_result_t *result);

/**
 * @brief Get position for token
 *
 * @param pt Paper trader instance
 * @param address Token mint address
 *
 * @return Pointer to position, or NULL if none
 */
paper_position_t *paper_get_position(paper_trader_t *pt, const char *address);

/**
 * @brief Check if has position for token
 *
 * @param pt Paper trader instance
 * @param address Token mint address
 *
 * @return true if position exists
 */
bool paper_has_position(paper_trader_t *pt, const char *address);

/**
 * @brief Get paper trading statistics
 *
 * @param pt Paper trader instance
 * @param current_balance Output: current balance
 * @param total_trades Output: completed trades
 * @param winning_trades Output: profitable trades
 * @param total_pnl Output: cumulative PnL
 * @param win_rate Output: win percentage (0.0-1.0)
 */
void paper_get_stats(paper_trader_t *pt, double *current_balance,
                     int *total_trades, int *winning_trades,
                     double *total_pnl, double *win_rate);

/**
 * @brief Calculate unrealized PnL for all positions
 *
 * @param pt Paper trader instance
 * @param current_prices Array of current prices (parallel to positions)
 *
 * @return Total unrealized PnL in SOL
 */
double paper_get_unrealized_pnl(paper_trader_t *pt, double *current_prices);

/**
 * @brief Print paper trading summary
 *
 * Outputs formatted statistics to stdout.
 *
 * @param pt Paper trader instance
 */
void paper_print_summary(paper_trader_t *pt);

/**
 * @brief Cleanup paper trader
 *
 * @param pt Paper trader instance
 */
void paper_trader_cleanup(paper_trader_t *pt);

#endif /* PAPER_TRADER_H */
