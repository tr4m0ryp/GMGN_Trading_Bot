/**
 * @file ai_trader.c
 * @brief Main AI trading implementation
 *
 * Integrates all components into a working trading loop:
 * - Model inference (stable-baselines3 via Python subprocess)
 * - Chart data polling (GMGN API)
 * - Paper trading with fee simulation
 * - Structured logging
 *
 * Dependencies: All ai_trader headers
 *
 * @date 2025-12-24
 */

/* Disable truncation warnings - we handle string sizes manually */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-truncation"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>
#include <math.h>
#include <sys/time.h>

#include "config_loader.h"
#include "trade_logger.h"
#include "paper_trader.h"
#include "chart_fetcher.h"
#include "model_loader.h"
#include "logger_integration.h"
#include "telegram_bot.h"

/* Trading configuration */
#define TRADER_POLL_INTERVAL_MS     1000    /* Poll charts every 1 second */
#define TRADER_MIN_CANDLES          30      /* Minimum candles before trading */
#define TRADER_MAX_TOKENS           16      /* Max concurrent tokens */
#define TRADER_MIN_CONFIDENCE       0.60    /* Minimum confidence to act */
#define TRADER_TP_CHECK_INTERVAL    5       /* Check TP/SL every N seconds */

/**
 * @brief Active token state
 */
typedef struct {
    char address[64];               /* Token mint address */
    char symbol[32];                /* Token symbol */
    chart_buffer_t chart;           /* Chart data */
    time_t added_at;                /* When added to trading */
    time_t last_trade_at;           /* Last trade time */
    int total_trades;               /* Trades on this token */
    double token_pnl;               /* PnL for this token */
    bool active;                    /* Slot in use */
    bool has_position;              /* Currently holding */
} active_token_t;

/**
 * @brief AI trader state
 */
typedef struct {
    /* Configuration */
    trader_config_t config;
    
    /* Model */
    model_state_t model;
    
    /* Paper trading */
    paper_trader_t paper_trader;
    
    /* Active tokens */
    active_token_t tokens[TRADER_MAX_TOKENS];
    int active_count;
    
    /* Threading */
    pthread_t trading_thread;
    pthread_mutex_t lock;
    volatile bool running;
    
    /* Statistics */
    int total_predictions;
    int total_buys;
    int total_sells;
    int64_t total_inference_time_us;
    time_t start_time;
    
    /* Logger integration */
    logger_integration_t logger;
    
    /* Telegram bot */
    telegram_bot_t telegram;
    pthread_t telegram_thread;
    volatile bool telegram_running;
} ai_trader_t;

/* Global trader instance for signal handling */
static ai_trader_t *g_trader = NULL;

/**
 * @brief Signal handler for graceful shutdown
 */
static void signal_handler(int sig) {
    (void)sig;
    if (g_trader) {
        g_trader->running = false;
    }
}

/**
 * @brief Handle Telegram commands
 */
static void telegram_command_handler(const char *command, const char *args, void *user_data) {
    ai_trader_t *trader = (ai_trader_t *)user_data;
    (void)args;
    
    if (!trader) return;
    
    if (strcmp(command, "/status") == 0) {
        time_t uptime = time(NULL) - trader->start_time;
        telegram_send_fmt(&trader->telegram,
            "📊 *Status*\n"
            "Mode: `%s`\n"
            "Uptime: `%ld min`\n"
            "Tokens: `%d active`\n"
            "Predictions: `%d`",
            trader->config.paper_trade ? "PAPER" : "LIVE",
            uptime / 60,
            trader->active_count,
            trader->total_predictions);
    }
    else if (strcmp(command, "/pnl") == 0) {
        if (trader->config.paper_trade) {
            double balance, total_pnl, win_rate;
            int total_trades, winning_trades;
            paper_get_stats(&trader->paper_trader, &balance, &total_trades,
                            &winning_trades, &total_pnl, &win_rate);
            telegram_send_fmt(&trader->telegram,
                "💰 *Profit/Loss*\n"
                "Total PnL: `%.6f SOL`\n"
                "Win Rate: `%.1f%%`\n"
                "Wins: `%d` | Losses: `%d`\n"
                "Trades: `%d`",
                total_pnl,
                win_rate * 100,
                winning_trades,
                total_trades - winning_trades,
                total_trades);
        } else {
            telegram_send_message(&trader->telegram, "📊 *PnL*: Live mode - check wallet");
        }
    }
    else if (strcmp(command, "/balance") == 0) {
        if (trader->config.paper_trade) {
            double balance, total_pnl, win_rate;
            int total_trades, winning_trades;
            paper_get_stats(&trader->paper_trader, &balance, &total_trades,
                            &winning_trades, &total_pnl, &win_rate);
            double initial = trader->paper_trader.initial_balance;
            telegram_send_fmt(&trader->telegram,
                "💵 *Balance*\n"
                "Current: `%.6f SOL`\n"
                "Initial: `%.6f SOL`\n"
                "Change: `%+.2f%%`",
                balance,
                initial,
                (initial > 0) ? ((balance - initial) / initial) * 100 : 0);
        } else {
            telegram_send_message(&trader->telegram, "💵 *Balance*: Check Solana wallet");
        }
    }
    else if (strcmp(command, "/trades") == 0) {
        telegram_send_fmt(&trader->telegram,
            "📈 *Recent Activity*\n"
            "Buys: `%d`\n"
            "Sells: `%d`\n"
            "Predictions: `%d`",
            trader->total_buys,
            trader->total_sells,
            trader->total_predictions);
    }
    else if (strcmp(command, "/help") == 0) {
        telegram_send_message(&trader->telegram,
            "🤖 *AI Trader Commands*\n"
            "/status - Bot status\n"
            "/pnl - Profit/Loss\n"
            "/balance - Wallet balance\n"
            "/trades - Recent trades\n"
            "/help - This message");
    }
}

/**
 * @brief Telegram polling thread
 */
static void *telegram_poll_thread(void *arg) {
    ai_trader_t *trader = (ai_trader_t *)arg;
    
    while (trader->telegram_running) {
        telegram_poll_updates(&trader->telegram);
    }
    
    return NULL;
}

/**
 * @brief Build observation vector from chart data and position state
 */
static int build_observation(const chart_buffer_t *chart, bool in_position,
                             double entry_price, int entry_idx,
                             float *obs) {
    int num_candles = chart->candle_count;
    
    if (num_candles < TRADER_MIN_CANDLES) {
        return -1;
    }
    
    /* Get current candle index (most recent) */
    int current_idx = num_candles - 1;
    const candle_t *current = &chart->candles[current_idx];
    double current_price = current->close;
    
    /* Feature extraction similar to Python environment */
    /* Features: normalized prices, returns, volume, etc. */
    
    /* Base price for normalization */
    double base_price = chart->candles[0].close;
    if (base_price <= 0.0) base_price = current_price;
    
    /* Calculate log returns over different windows */
    int idx = 0;
    
    /* Price features (log-normalized) */
    for (int i = 0; i < 5 && i < num_candles; i++) {
        int candle_idx = current_idx - i;
        if (candle_idx >= 0) {
            double c = chart->candles[candle_idx].close;
            obs[idx++] = (float)log(c / base_price);
        } else {
            obs[idx++] = 0.0f;
        }
    }
    
    /* Volume features */
    for (int i = 0; i < 4 && i < num_candles; i++) {
        int candle_idx = current_idx - i;
        if (candle_idx >= 0) {
            obs[idx++] = (float)(chart->candles[candle_idx].volume);
        } else {
            obs[idx++] = 0.0f;
        }
    }
    
    /* Price change over windows */
    double changes[] = {1, 5, 10, 20, 30};
    for (int w = 0; w < 5 && idx < MODEL_NUM_FEATURES; w++) {
        int lookback = (int)changes[w];
        if (current_idx >= lookback && current_idx - lookback >= 0) {
            double old_price = chart->candles[current_idx - lookback].close;
            if (old_price > 0) {
                obs[idx++] = (float)((current_price - old_price) / old_price);
            } else {
                obs[idx++] = 0.0f;
            }
        } else {
            obs[idx++] = 0.0f;
        }
    }
    
    /* Pad remaining price features with zeros */
    while (idx < MODEL_NUM_FEATURES) {
        obs[idx++] = 0.0f;
    }
    
    /* Position features (5 total) */
    obs[idx++] = in_position ? 1.0f : 0.0f;     /* in_position */
    
    if (in_position && entry_price > 0) {
        obs[idx++] = (float)((entry_price - current_price) / current_price);  /* entry_price_norm */
        obs[idx++] = (float)((current_price - entry_price) / entry_price);    /* unrealized_pnl */
        int time_in = current_idx - entry_idx;
        obs[idx++] = (float)(time_in / 100.0);   /* time_in_position */
    } else {
        obs[idx++] = 0.0f;
        obs[idx++] = 0.0f;
        obs[idx++] = 0.0f;
    }
    
    /* Momentum hint */
    if (current_idx >= 5) {
        double old_price = chart->candles[current_idx - 5].close;
        obs[idx++] = (float)((current_price - old_price) / old_price);
    } else {
        obs[idx++] = 0.0f;
    }
    
    return 0;
}

/**
 * @brief Execute trade based on model prediction
 */
static int execute_trade(ai_trader_t *trader, active_token_t *token,
                         const model_result_t *result, double current_price) {
    paper_trade_result_t trade_result;
    trade_record_t record;
    int ret = 0;
    
    memset(&trade_result, 0, sizeof(trade_result));
    memset(&record, 0, sizeof(record));
    
    if (result->action == SIGNAL_BUY && !token->has_position) {
        /* Execute BUY */
        ret = paper_buy(&trader->paper_trader,
                        token->address, token->symbol,
                        current_price,
                        trader->config.trade_amount_sol,
                        result->confidence,
                        token->chart.candle_count,
                        &trade_result);
        
        if (ret == 0 && trade_result.success) {
            token->has_position = true;
            token->last_trade_at = time(NULL);
            token->total_trades++;
            trader->total_buys++;
            
            log_info("[BUY] %s @ %.10f (conf: %.2f%%, balance: %.6f SOL)",
                     token->symbol, current_price,
                     result->confidence * 100, trade_result.new_balance);
            
            /* Log trade record */
            record.timestamp = time(NULL);
            strncpy(record.token_address, token->address, sizeof(record.token_address) - 1);
            strncpy(record.token_symbol, token->symbol, sizeof(record.token_symbol) - 1);
            strncpy(record.action, "BUY", sizeof(record.action) - 1);
            record.exit_price = current_price;
            record.amount_sol = trader->config.trade_amount_sol;
            record.confidence = result->confidence;
            record.balance_after = trade_result.new_balance;
            record.is_paper = trader->config.paper_trade;
            record.sequence_length = token->chart.candle_count;
            log_trade(&record);
        }
    }
    else if (result->action == SIGNAL_SELL && token->has_position) {
        /* Get entry price for record */
        paper_position_t *pos = paper_get_position(&trader->paper_trader, token->address);
        double entry_price = pos ? pos->entry_price : 0.0;
        
        /* Execute SELL */
        ret = paper_sell(&trader->paper_trader,
                         token->address,
                         current_price,
                         &trade_result);
        
        if (ret == 0 && trade_result.success) {
            token->has_position = false;
            token->last_trade_at = time(NULL);
            token->total_trades++;
            token->token_pnl += trade_result.pnl;
            trader->total_sells++;
            
            log_info("[SELL] %s @ %.10f, PnL: %.6f SOL (%.2f%%), balance: %.6f",
                     token->symbol, current_price,
                     trade_result.pnl, trade_result.pnl_pct * 100,
                     trade_result.new_balance);
            
            /* Log trade record */
            record.timestamp = time(NULL);
            strncpy(record.token_address, token->address, sizeof(record.token_address) - 1);
            strncpy(record.token_symbol, token->symbol, sizeof(record.token_symbol) - 1);
            strncpy(record.action, "SELL", sizeof(record.action) - 1);
            record.entry_price = entry_price;
            record.exit_price = current_price;
            record.amount_sol = trader->config.trade_amount_sol;
            record.pnl_sol = trade_result.pnl;
            record.pnl_pct = trade_result.pnl_pct;
            record.confidence = result->confidence;
            record.balance_after = trade_result.new_balance;
            record.is_paper = trader->config.paper_trade;
            record.sequence_length = token->chart.candle_count;
            log_trade(&record);
        }
    }
    
    return ret;
}

/**
 * @brief Process single token: fetch chart, run inference, execute trade
 */
static void process_token(ai_trader_t *trader, active_token_t *token) {
    fetch_status_t fetch_status;
    model_result_t result;
    float obs[MODEL_TOTAL_OBS_DIM];
    
    /* Fetch latest chart data */
    fetch_status = chart_fetch(token->address, &token->chart);
    
    if (fetch_status != FETCH_SUCCESS) {
        if (token->chart.fetch_failures >= 5) {
            log_warn("[CHART] %s: too many failures (%d), removing",
                     token->symbol, token->chart.fetch_failures);
            token->active = false;
        }
        return;
    }
    
    /* Need minimum candles before trading */
    if (token->chart.candle_count < TRADER_MIN_CANDLES) {
        return;
    }
    
    double current_price = chart_get_price(&token->chart);
    if (current_price <= 0.0) {
        return;
    }
    
    /* Model handles all trading decisions - no manual TP/SL override */
    
    /* Get entry info for observation */
    bool in_position = token->has_position;
    double entry_price = 0.0;
    int entry_idx = 0;
    
    if (in_position) {
        paper_position_t *pos = paper_get_position(&trader->paper_trader, token->address);
        if (pos) {
            entry_price = pos->entry_price;
            entry_idx = pos->entry_seq_length;
        }
    }
    
    /* Build observation */
    if (build_observation(&token->chart, in_position, entry_price, entry_idx, obs) != 0) {
        return;
    }
    
    /* Run model inference */
    if (model_predict(&trader->model, obs, &result) != 0) {
        log_debug("[MODEL] Inference failed for %s: %s", token->symbol, result.error);
        return;
    }
    
    trader->total_predictions++;
    trader->total_inference_time_us += result.inference_time_us;
    
    /* Log prediction */
    log_debug("[PRED] %s: %s (conf: %.2f%%, probs: H=%.2f B=%.2f S=%.2f, time: %ldus)",
              token->symbol, model_action_str(result.action),
              result.confidence * 100,
              result.probabilities[0], result.probabilities[1], result.probabilities[2],
              result.inference_time_us);
    
    /* Execute trade if confidence is high enough */
    if (result.confidence >= TRADER_MIN_CONFIDENCE && result.action != SIGNAL_HOLD) {
        execute_trade(trader, token, &result, current_price);
    }
}

/**
 * @brief Main trading loop
 */
static void *trading_loop(void *arg) {
    ai_trader_t *trader = (ai_trader_t *)arg;
    struct timespec sleep_time;
    
    sleep_time.tv_sec = 0;
    sleep_time.tv_nsec = TRADER_POLL_INTERVAL_MS * 1000000L;
    
    log_info("Trading loop started");
    
    while (trader->running) {
        pthread_mutex_lock(&trader->lock);
        
        /* Process each active token */
        for (int i = 0; i < TRADER_MAX_TOKENS; i++) {
            if (!trader->tokens[i].active) continue;
            
            process_token(trader, &trader->tokens[i]);
        }
        
        pthread_mutex_unlock(&trader->lock);
        
        /* Sleep for poll interval */
        nanosleep(&sleep_time, NULL);
    }
    
    log_info("Trading loop stopped");
    return NULL;
}

/**
 * @brief Add token to trader
 */
int trader_add_token(ai_trader_t *trader, const char *address, const char *symbol) {
    if (!trader || !address || !symbol) {
        return -1;
    }
    
    pthread_mutex_lock(&trader->lock);
    
    /* Check if already tracking */
    for (int i = 0; i < TRADER_MAX_TOKENS; i++) {
        if (trader->tokens[i].active &&
            strcmp(trader->tokens[i].address, address) == 0) {
            pthread_mutex_unlock(&trader->lock);
            return 0;  /* Already tracking */
        }
    }
    
    /* Find free slot */
    int slot = -1;
    for (int i = 0; i < TRADER_MAX_TOKENS; i++) {
        if (!trader->tokens[i].active) {
            slot = i;
            break;
        }
    }
    
    if (slot < 0) {
        pthread_mutex_unlock(&trader->lock);
        log_warn("No free slots for token %s", symbol);
        return -1;
    }
    
    /* Initialize token */
    active_token_t *token = &trader->tokens[slot];
    memset(token, 0, sizeof(active_token_t));
    strncpy(token->address, address, sizeof(token->address) - 1);
    strncpy(token->symbol, symbol, sizeof(token->symbol) - 1);
    chart_buffer_init(&token->chart, address, symbol);
    token->added_at = time(NULL);
    token->active = true;
    
    trader->active_count++;
    
    pthread_mutex_unlock(&trader->lock);
    
    log_info("[ADD] Token %s (%s) added for trading", symbol, address);
    return 0;
}

/**
 * @brief Print trading statistics
 */
void trader_print_stats(ai_trader_t *trader) {
    double avg_inference = 0.0;
    
    if (trader->total_predictions > 0) {
        avg_inference = (double)trader->total_inference_time_us / trader->total_predictions;
    }
    
    printf("\n======== AI Trader Statistics ========\n");
    printf("Mode:                %s\n", trader->config.paper_trade ? "PAPER" : "LIVE");
    printf("Active Tokens:       %d\n", trader->active_count);
    printf("Total Predictions:   %d\n", trader->total_predictions);
    printf("Total Buys:          %d\n", trader->total_buys);
    printf("Total Sells:         %d\n", trader->total_sells);
    printf("Avg Inference Time:  %.2f ms\n", avg_inference / 1000.0);
    printf("======================================\n\n");
    
    if (trader->config.paper_trade) {
        paper_print_summary(&trader->paper_trader);
    }
}

/**
 * @brief Initialize trader
 */
int trader_init(ai_trader_t *trader, const char *config_path, const char *model_dir) {
    if (!trader) {
        return -1;
    }
    
    memset(trader, 0, sizeof(ai_trader_t));
    
    /* Load configuration */
    if (config_load(&trader->config, config_path) != 0) {
        log_warn("Using default configuration");
    }
    
    if (config_validate(&trader->config) != 0) {
        return -1;
    }
    
    config_print(&trader->config);
    
    /* Initialize paper trader */
    if (trader->config.paper_trade) {
        if (paper_trader_init(&trader->paper_trader, trader->config.paper_wallet_balance) != 0) {
            log_error("Failed to initialize paper trader");
            return -1;
        }
    }
    
    /* Initialize model */
    if (model_init(&trader->model, model_dir) != 0) {
        log_error("Failed to load model from %s", model_dir);
        log_error("Please place model.zip (stable-baselines3) in the models/ directory");
        return -1;
    }
    
    log_info("Model loaded: backend=%s, path=%s",
             model_backend_name(&trader->model), trader->model.model_path);
    
    /* Initialize chart fetcher */
    if (chart_fetcher_init(&trader->config) != 0) {
        log_error("Failed to initialize chart fetcher");
        return -1;
    }
    
    /* Initialize mutex */
    if (pthread_mutex_init(&trader->lock, NULL) != 0) {
        return -1;
    }
    
    trader->start_time = time(NULL);
    
    /* Initialize logger integration for auto-discovery */
    logger_integration_init(&trader->logger);
    
    /* Initialize Telegram bot */
    if (trader->config.telegram_bot_token[0] != '\0' &&
        trader->config.telegram_chat_id[0] != '\0') {
        if (telegram_init(&trader->telegram,
                          trader->config.telegram_bot_token,
                          trader->config.telegram_chat_id) == 0) {
            telegram_set_command_callback(&trader->telegram, telegram_command_handler, trader);
            log_info("Telegram bot initialized");
        } else {
            log_warn("Failed to initialize Telegram bot");
        }
    } else {
        log_info("Telegram bot disabled (no token/chat_id configured)");
    }
    
    return 0;
}

/**
 * @brief Start trading
 */
int trader_start(ai_trader_t *trader) {
    if (!trader) {
        return -1;
    }
    
    trader->running = true;
    
    if (pthread_create(&trader->trading_thread, NULL, trading_loop, trader) != 0) {
        log_error("Failed to create trading thread");
        return -1;
    }
    
    /* Start Telegram polling if configured */
    if (trader->telegram.initialized) {
        trader->telegram_running = true;
        if (pthread_create(&trader->telegram_thread, NULL, telegram_poll_thread, trader) != 0) {
            log_warn("Failed to start Telegram polling");
        } else {
            log_info("Telegram bot listening for commands");
            telegram_send_message(&trader->telegram, "🤖 *AI Trader Started*\nSend /help for commands");
        }
    }
    
    return 0;
}

/**
 * @brief Stop trading
 */
void trader_stop(ai_trader_t *trader) {
    if (!trader || !trader->running) {
        return;
    }
    
    trader->running = false;
    pthread_join(trader->trading_thread, NULL);
}

/**
 * @brief Cleanup trader
 */
void trader_cleanup(ai_trader_t *trader) {
    if (!trader) {
        return;
    }
    
    trader_stop(trader);
    trader_print_stats(trader);
    
    /* Stop and cleanup Telegram */
    if (trader->telegram.initialized) {
        trader->telegram_running = false;
        telegram_send_message(&trader->telegram, "🛑 *AI Trader Stopped*");
        pthread_join(trader->telegram_thread, NULL);
        telegram_cleanup(&trader->telegram);
    }
    
    /* Cleanup integrations */
    logger_integration_cleanup(&trader->logger);
    chart_fetcher_cleanup();
    model_cleanup(&trader->model);
    
    if (trader->config.paper_trade) {
        paper_trader_cleanup(&trader->paper_trader);
    }
    
    pthread_mutex_destroy(&trader->lock);
}

/* ============================================== */
/*                  MAIN ENTRY                    */
/* ============================================== */

static void print_usage(const char *prog) {
    printf("AI Trader - Automated Trading Bot\n\n");
    printf("Usage: %s [options]\n\n", prog);
    printf("Options:\n");
    printf("  -c, --config PATH     Path to .env config file (default: .env)\n");
    printf("  -m, --models PATH     Path to models directory (default: models)\n");
    printf("  -t, --token ADDR:SYM  Add token to trade (can specify multiple)\n");
    printf("  -d, --demo            Run demo with test token\n");
    printf("  -h, --help            Show this help\n");
    printf("\n");
    printf("Model Setup:\n");
    printf("  Place your trained model in the models/ directory as model.zip\n");
    printf("  The model should be a stable-baselines3 PPO model.\n");
    printf("\n");
    printf("Example:\n");
    printf("  %s --demo                     # Run with demo token\n", prog);
    printf("  %s -t ABC123:PEPE -t DEF456:DOGE\n", prog);
}

/**
 * @brief Callback when logger_c discovers a new token
 */
static void on_new_token(const char *address, const char *symbol, void *user_data) {
    ai_trader_t *trader = (ai_trader_t *)user_data;
    if (trader) {
        trader_add_token(trader, address, symbol);
    }
}

int main(int argc, char *argv[]) {
    ai_trader_t trader;
    const char *config_path = ".env";
    const char *model_dir = "models";
    bool run_demo = false;
    char demo_tokens[10][128];
    int demo_token_count = 0;
    
    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        else if ((strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--config") == 0) && i + 1 < argc) {
            config_path = argv[++i];
        }
        else if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--models") == 0) && i + 1 < argc) {
            model_dir = argv[++i];
        }
        else if ((strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--token") == 0) && i + 1 < argc) {
            if (demo_token_count < 10) {
                strncpy(demo_tokens[demo_token_count++], argv[++i], 127);
            }
        }
        else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--demo") == 0) {
            run_demo = true;
        }
    }
    
    /* Setup signal handlers */
    g_trader = &trader;
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    printf("=========================================\n");
    printf("       AI Trader - Starting Up          \n");
    printf("=========================================\n\n");
    
    /* Initialize logger */
    if (logger_init("logs") != 0) {
        fprintf(stderr, "Failed to initialize logger\n");
        return 1;
    }
    
    /* Initialize trader */
    if (trader_init(&trader, config_path, model_dir) != 0) {
        fprintf(stderr, "Failed to initialize trader\n");
        logger_cleanup();
        return 1;
    }
    
    /* Add demo token if requested */
    if (run_demo) {
        /* Use a known active token for testing - this should be replaced */
        trader_add_token(&trader, "So11111111111111111111111111111111111111112", "SOL");
    }
    
    /* Add tokens from command line */
    for (int i = 0; i < demo_token_count; i++) {
        char *colon = strchr(demo_tokens[i], ':');
        if (colon) {
            *colon = '\0';
            trader_add_token(&trader, demo_tokens[i], colon + 1);
        }
    }
    
    /* Connect to logger_c for auto-discovery of new tokens */
    logger_integration_set_callback(&trader.logger, on_new_token, &trader);
    if (logger_integration_connect(&trader.logger) == 0) {
        log_info("Logger connected - auto-discovering new tokens");
    } else {
        log_warn("Logger connection failed - run tokens manually or start logger_c");
    }
    
    /* Start trading */
    if (trader_start(&trader) != 0) {
        log_error("Failed to start trading");
        trader_cleanup(&trader);
        logger_cleanup();
        return 1;
    }
    
    log_info("AI Trader running. Press Ctrl+C to stop.");
    
    /* Main loop - just wait for signals */
    while (trader.running) {
        sleep(1);
    }
    
    /* Cleanup */
    log_info("Shutting down...");
    trader_cleanup(&trader);
    logger_flush();
    logger_cleanup();
    
    printf("\nAI Trader stopped.\n");
    return 0;
}
