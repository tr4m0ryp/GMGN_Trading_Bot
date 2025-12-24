/**
 * @file paper_trader.c
 * @brief Implementation of paper trading simulation
 *
 * Simulates trading with virtual balance, accurately modeling:
 * - Jito tips and gas fees (~7% per round trip)
 * - Position tracking and PnL calculation
 * - Win/loss statistics
 *
 * Dependencies: <stdio.h>, <stdlib.h>, <string.h>, <time.h>, <pthread.h>
 *
 * @date 2025-12-24
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

#include "paper_trader.h"

int paper_trader_init(paper_trader_t *pt, double initial_balance) {
    if (!pt || initial_balance <= 0.0) {
        return -1;
    }

    memset(pt, 0, sizeof(paper_trader_t));

    pt->initial_balance = initial_balance;
    pt->current_balance = initial_balance;
    pt->total_invested = 0.0;
    pt->position_count = 0;
    pt->total_trades = 0;
    pt->winning_trades = 0;
    pt->total_pnl = 0.0;
    pt->total_fees_paid = 0.0;
    pt->best_trade_pnl = 0.0;
    pt->worst_trade_pnl = 0.0;

    if (pthread_mutex_init(&pt->lock, NULL) != 0) {
        return -1;
    }

    return 0;
}

/**
 * @brief Find free position slot
 */
static int find_free_slot(paper_trader_t *pt) {
    for (int i = 0; i < PAPER_MAX_POSITIONS; i++) {
        if (!pt->positions[i].active) {
            return i;
        }
    }
    return -1;
}

/**
 * @brief Find position by address
 */
static int find_position(paper_trader_t *pt, const char *address) {
    for (int i = 0; i < PAPER_MAX_POSITIONS; i++) {
        if (pt->positions[i].active &&
            strcmp(pt->positions[i].token_address, address) == 0) {
            return i;
        }
    }
    return -1;
}

int paper_buy(paper_trader_t *pt, const char *address, const char *symbol,
              double price, double amount_sol, double confidence,
              int seq_length, paper_trade_result_t *result) {

    if (!pt || !address || !symbol || price <= 0.0 || amount_sol <= 0.0) {
        if (result) {
            result->success = false;
            snprintf(result->error, sizeof(result->error), "Invalid parameters");
        }
        return -1;
    }

    pthread_mutex_lock(&pt->lock);

    /* Clear result */
    if (result) {
        memset(result, 0, sizeof(paper_trade_result_t));
    }

    /* Calculate total cost including fee */
    double fee = PAPER_TOTAL_FEE_PER_TX;
    double total_cost = amount_sol + fee;

    /* Check balance */
    if (total_cost > pt->current_balance) {
        pthread_mutex_unlock(&pt->lock);
        if (result) {
            result->success = false;
            snprintf(result->error, sizeof(result->error),
                     "Insufficient balance: %.6f SOL needed, %.6f available",
                     total_cost, pt->current_balance);
        }
        return -1;
    }

    /* Check if already have position */
    if (find_position(pt, address) >= 0) {
        pthread_mutex_unlock(&pt->lock);
        if (result) {
            result->success = false;
            snprintf(result->error, sizeof(result->error),
                     "Already have position for %s", symbol);
        }
        return -1;
    }

    /* Find free slot */
    int slot = find_free_slot(pt);
    if (slot < 0) {
        pthread_mutex_unlock(&pt->lock);
        if (result) {
            result->success = false;
            snprintf(result->error, sizeof(result->error),
                     "No free position slots");
        }
        return -1;
    }

    /* Create position */
    paper_position_t *pos = &pt->positions[slot];
    strncpy(pos->token_address, address, sizeof(pos->token_address) - 1);
    strncpy(pos->token_symbol, symbol, sizeof(pos->token_symbol) - 1);
    pos->entry_price = price;
    pos->amount_sol = amount_sol;
    pos->tokens_held = amount_sol / price;
    pos->entry_time = time(NULL);
    pos->entry_seq_length = seq_length;
    pos->confidence = confidence;
    pos->active = true;

    /* Update balances */
    pt->current_balance -= total_cost;
    pt->total_invested += amount_sol;
    pt->total_fees_paid += fee;
    pt->position_count++;

    pthread_mutex_unlock(&pt->lock);

    /* Fill result */
    if (result) {
        result->success = true;
        result->executed_price = price;
        result->fee_paid = fee;
        result->pnl = 0.0;
        result->pnl_pct = 0.0;
        result->new_balance = pt->current_balance;
    }

    return 0;
}

int paper_sell(paper_trader_t *pt, const char *address, double price,
               paper_trade_result_t *result) {

    if (!pt || !address || price <= 0.0) {
        if (result) {
            result->success = false;
            snprintf(result->error, sizeof(result->error), "Invalid parameters");
        }
        return -1;
    }

    pthread_mutex_lock(&pt->lock);

    /* Clear result */
    if (result) {
        memset(result, 0, sizeof(paper_trade_result_t));
    }

    /* Find position */
    int slot = find_position(pt, address);
    if (slot < 0) {
        pthread_mutex_unlock(&pt->lock);
        if (result) {
            result->success = false;
            snprintf(result->error, sizeof(result->error),
                     "No position for address %s", address);
        }
        return -1;
    }

    paper_position_t *pos = &pt->positions[slot];

    /* Calculate sale value */
    double sell_value = pos->tokens_held * price;
    double fee = PAPER_TOTAL_FEE_PER_TX;
    double net_value = sell_value - fee;

    /* Calculate PnL */
    double pnl = net_value - pos->amount_sol;
    double pnl_pct = pnl / pos->amount_sol;

    /* Update balance */
    pt->current_balance += net_value;
    pt->total_invested -= pos->amount_sol;
    pt->total_fees_paid += fee;
    pt->total_pnl += pnl;
    pt->total_trades++;

    if (pnl > 0) {
        pt->winning_trades++;
    }

    /* Track best/worst */
    if (pnl > pt->best_trade_pnl) {
        pt->best_trade_pnl = pnl;
    }
    if (pnl < pt->worst_trade_pnl) {
        pt->worst_trade_pnl = pnl;
    }

    /* Fill result before clearing position */
    if (result) {
        result->success = true;
        result->executed_price = price;
        result->fee_paid = fee;
        result->pnl = pnl;
        result->pnl_pct = pnl_pct;
        result->new_balance = pt->current_balance;
    }

    /* Clear position */
    memset(pos, 0, sizeof(paper_position_t));
    pt->position_count--;

    pthread_mutex_unlock(&pt->lock);

    return 0;
}

paper_position_t *paper_get_position(paper_trader_t *pt, const char *address) {
    if (!pt || !address) {
        return NULL;
    }

    pthread_mutex_lock(&pt->lock);

    int slot = find_position(pt, address);
    paper_position_t *pos = (slot >= 0) ? &pt->positions[slot] : NULL;

    pthread_mutex_unlock(&pt->lock);

    return pos;
}

bool paper_has_position(paper_trader_t *pt, const char *address) {
    if (!pt || !address) {
        return false;
    }

    pthread_mutex_lock(&pt->lock);
    int slot = find_position(pt, address);
    pthread_mutex_unlock(&pt->lock);

    return slot >= 0;
}

void paper_get_stats(paper_trader_t *pt, double *current_balance,
                     int *total_trades, int *winning_trades,
                     double *total_pnl, double *win_rate) {

    if (!pt) {
        return;
    }

    pthread_mutex_lock(&pt->lock);

    if (current_balance) {
        *current_balance = pt->current_balance;
    }
    if (total_trades) {
        *total_trades = pt->total_trades;
    }
    if (winning_trades) {
        *winning_trades = pt->winning_trades;
    }
    if (total_pnl) {
        *total_pnl = pt->total_pnl;
    }
    if (win_rate) {
        *win_rate = pt->total_trades > 0 ?
                    (double)pt->winning_trades / pt->total_trades : 0.0;
    }

    pthread_mutex_unlock(&pt->lock);
}

double paper_get_unrealized_pnl(paper_trader_t *pt, double *current_prices) {
    double unrealized = 0.0;

    if (!pt || !current_prices) {
        return 0.0;
    }

    pthread_mutex_lock(&pt->lock);

    int price_idx = 0;
    for (int i = 0; i < PAPER_MAX_POSITIONS; i++) {
        if (!pt->positions[i].active) {
            continue;
        }

        paper_position_t *pos = &pt->positions[i];
        double current_value = pos->tokens_held * current_prices[price_idx];
        double cost = pos->amount_sol + PAPER_FEE_ROUND_TRIP;  /* Include expected sell fee */
        unrealized += (current_value - cost);

        price_idx++;
    }

    pthread_mutex_unlock(&pt->lock);

    return unrealized;
}

void paper_print_summary(paper_trader_t *pt) {
    if (!pt) {
        return;
    }

    pthread_mutex_lock(&pt->lock);

    double roi = (pt->current_balance - pt->initial_balance) / pt->initial_balance * 100;
    double win_rate = pt->total_trades > 0 ?
                      (double)pt->winning_trades / pt->total_trades * 100 : 0.0;

    printf("\n========== Paper Trading Summary ==========\n");
    printf("Initial Balance:    %.6f SOL\n", pt->initial_balance);
    printf("Current Balance:    %.6f SOL\n", pt->current_balance);
    printf("Total Invested:     %.6f SOL (in open positions)\n", pt->total_invested);
    printf("-------------------------------------------\n");
    printf("Total Trades:       %d\n", pt->total_trades);
    printf("Winning Trades:     %d\n", pt->winning_trades);
    printf("Win Rate:           %.1f%%\n", win_rate);
    printf("-------------------------------------------\n");
    printf("Total PnL:          %.6f SOL\n", pt->total_pnl);
    printf("ROI:                %.2f%%\n", roi);
    printf("Best Trade:         %.6f SOL\n", pt->best_trade_pnl);
    printf("Worst Trade:        %.6f SOL\n", pt->worst_trade_pnl);
    printf("Total Fees Paid:    %.6f SOL\n", pt->total_fees_paid);
    printf("-------------------------------------------\n");
    printf("Open Positions:     %d\n", pt->position_count);
    printf("==========================================\n\n");

    pthread_mutex_unlock(&pt->lock);
}

void paper_trader_cleanup(paper_trader_t *pt) {
    if (!pt) {
        return;
    }

    paper_print_summary(pt);

    pthread_mutex_destroy(&pt->lock);
    memset(pt, 0, sizeof(paper_trader_t));
}
