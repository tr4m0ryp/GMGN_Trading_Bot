/**
 * @file config_loader.c
 * @brief Implementation of .env configuration parser
 *
 * Parses environment configuration files in KEY=VALUE format.
 * Handles quoted values, comments, and missing values with defaults.
 *
 * Dependencies: <stdio.h>, <stdlib.h>, <string.h>, <ctype.h>
 *
 * @date 2025-12-24
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "config_loader.h"

/* Line buffer size */
#define LINE_BUFFER_SIZE    1024

/**
 * @brief Safe string copy that avoids truncation warnings
 */
static void safe_strcpy(char *dest, size_t dest_size, const char *src) {
    if (!dest || !src || dest_size == 0) {
        return;
    }
    size_t src_len = strlen(src);
    size_t copy_len = (src_len < dest_size - 1) ? src_len : dest_size - 1;
    memcpy(dest, src, copy_len);
    dest[copy_len] = '\0';
}

/**
 * @brief Trim whitespace from string (in-place)
 */
static char *trim_whitespace(char *str) {
    char *end;

    /* Trim leading space */
    while (isspace((unsigned char)*str)) {
        str++;
    }

    if (*str == 0) {
        return str;
    }

    /* Trim trailing space */
    end = str + strlen(str) - 1;
    while (end > str && isspace((unsigned char)*end)) {
        end--;
    }

    end[1] = '\0';
    return str;
}

/**
 * @brief Parse boolean value from string
 */
static bool parse_bool(const char *value) {
    if (strcasecmp(value, "true") == 0 ||
        strcasecmp(value, "1") == 0 ||
        strcasecmp(value, "yes") == 0) {
        return true;
    }
    return false;
}

/**
 * @brief Parse single config line
 */
static int parse_line(char *line, char *key, char *value) {
    char *eq_pos;
    char *val_start;
    char *val_end;

    /* Skip empty lines and comments */
    line = trim_whitespace(line);
    if (*line == '\0' || *line == '#') {
        return 0;
    }

    /* Find equals sign */
    eq_pos = strchr(line, '=');
    if (!eq_pos) {
        return 0;
    }

    /* Extract key */
    *eq_pos = '\0';
    safe_strcpy(key, CONFIG_MAX_KEY_LEN, trim_whitespace(line));

    /* Extract value */
    val_start = trim_whitespace(eq_pos + 1);

    /* Handle quoted values */
    if (*val_start == '"' || *val_start == '\'') {
        char quote = *val_start;
        val_start++;
        val_end = strrchr(val_start, quote);
        if (val_end) {
            *val_end = '\0';
        }
    }

    safe_strcpy(value, CONFIG_MAX_VALUE_LEN, val_start);

    return 1;
}

void config_set_defaults(trader_config_t *config) {
    if (!config) {
        return;
    }

    memset(config, 0, sizeof(trader_config_t));

    /* Default values */
    config->paper_trade = true;                 /* Safe default: paper trading */
    config->trade_amount_sol = 0.01;            /* 0.01 SOL per trade */
    config->max_positions = 5;                  /* Max 5 concurrent */
    config->paper_wallet_balance = 1.0;         /* 1 SOL starting balance */
    config->take_profit_pct = 0.30;             /* 30% take profit */
    config->stop_loss_pct = 0.08;               /* 8% stop loss */
    config->jito_tip_lamports = 50000;          /* 0.00005 SOL tip */

    safe_strcpy(config->jito_endpoint, sizeof(config->jito_endpoint),
                "https://amsterdam.mainnet.block-engine.jito.wtf");
}

int config_load(trader_config_t *config, const char *env_path) {
    FILE *fp;
    char line[LINE_BUFFER_SIZE];
    char key[CONFIG_MAX_KEY_LEN];
    char value[CONFIG_MAX_VALUE_LEN];

    if (!config || !env_path) {
        return -1;
    }

    /* Start with defaults */
    config_set_defaults(config);

    /* Open config file */
    fp = fopen(env_path, "r");
    if (!fp) {
        fprintf(stderr, "[CONFIG] Cannot open %s: using defaults\n", env_path);
        return -1;
    }

    /* Parse each line */
    while (fgets(line, sizeof(line), fp)) {
        if (!parse_line(line, key, value)) {
            continue;
        }

        /* Trading mode */
        if (strcmp(key, "PAPERTRADE") == 0) {
            config->paper_trade = parse_bool(value);
        }
        /* Wallet */
        else if (strcmp(key, "WALLET_PRIVATE_KEY") == 0) {
            safe_strcpy(config->wallet_private_key, sizeof(config->wallet_private_key), value);
        }
        /* Position settings */
        else if (strcmp(key, "TRADE_AMOUNT_SOL") == 0) {
            config->trade_amount_sol = atof(value);
        }
        else if (strcmp(key, "MAX_POSITIONS") == 0) {
            config->max_positions = atoi(value);
        }
        /* Paper trading */
        else if (strcmp(key, "PAPER_WALLET_BALANCE") == 0) {
            config->paper_wallet_balance = atof(value);
        }
        /* Risk management */
        else if (strcmp(key, "TAKE_PROFIT_PCT") == 0) {
            config->take_profit_pct = atof(value);
        }
        else if (strcmp(key, "STOP_LOSS_PCT") == 0) {
            config->stop_loss_pct = atof(value);
        }
        /* API cookies */
        else if (strcmp(key, "GMGN_CF_CLEARANCE") == 0) {
            safe_strcpy(config->gmgn_cf_clearance, sizeof(config->gmgn_cf_clearance), value);
        }
        else if (strcmp(key, "GMGN_CF_BM") == 0) {
            safe_strcpy(config->gmgn_cf_bm, sizeof(config->gmgn_cf_bm), value);
        }
        else if (strcmp(key, "GMGN_GA") == 0) {
            safe_strcpy(config->gmgn_ga, sizeof(config->gmgn_ga), value);
        }
        else if (strcmp(key, "GMGN_GA_SESSION") == 0) {
            safe_strcpy(config->gmgn_ga_session, sizeof(config->gmgn_ga_session), value);
        }
        /* Jito settings */
        else if (strcmp(key, "JITO_ENDPOINT") == 0) {
            safe_strcpy(config->jito_endpoint, sizeof(config->jito_endpoint), value);
        }
        else if (strcmp(key, "JITO_TIP_LAMPORTS") == 0) {
            config->jito_tip_lamports = (uint64_t)atoll(value);
        }
    }

    fclose(fp);
    return 0;
}

int config_validate(const trader_config_t *config) {
    if (!config) {
        return -1;
    }

    /* Validate trade amount */
    if (config->trade_amount_sol <= 0.0 || config->trade_amount_sol > 1.0) {
        fprintf(stderr, "[CONFIG] Invalid TRADE_AMOUNT_SOL: %.4f (must be 0-1)\n",
                config->trade_amount_sol);
        return -1;
    }

    /* Validate max positions */
    if (config->max_positions <= 0 || config->max_positions > 32) {
        fprintf(stderr, "[CONFIG] Invalid MAX_POSITIONS: %d (must be 1-32)\n",
                config->max_positions);
        return -1;
    }

    /* Validate take profit */
    if (config->take_profit_pct <= 0.05 || config->take_profit_pct > 1.0) {
        fprintf(stderr, "[CONFIG] Invalid TAKE_PROFIT_PCT: %.2f (must be 0.05-1.0)\n",
                config->take_profit_pct);
        return -1;
    }

    /* Validate stop loss */
    if (config->stop_loss_pct <= 0.0 || config->stop_loss_pct > 0.50) {
        fprintf(stderr, "[CONFIG] Invalid STOP_LOSS_PCT: %.2f (must be 0.01-0.50)\n",
                config->stop_loss_pct);
        return -1;
    }

    /* Live mode requires wallet */
    if (!config->paper_trade && strlen(config->wallet_private_key) == 0) {
        fprintf(stderr, "[CONFIG] Live trading requires WALLET_PRIVATE_KEY\n");
        return -1;
    }

    return 0;
}

void config_print(const trader_config_t *config) {
    if (!config) {
        return;
    }

    printf("\n=== AI Trader Configuration ===\n");
    printf("Mode:               %s\n", config->paper_trade ? "PAPER TRADING" : "LIVE TRADING");
    printf("Trade Amount:       %.4f SOL\n", config->trade_amount_sol);
    printf("Max Positions:      %d\n", config->max_positions);

    if (config->paper_trade) {
        printf("Paper Balance:      %.4f SOL\n", config->paper_wallet_balance);
    } else {
        printf("Wallet Key:         %s****\n",
               strlen(config->wallet_private_key) > 4 ?
               "****" : "(not set)");
    }

    printf("Take Profit:        %.1f%%\n", config->take_profit_pct * 100);
    printf("Stop Loss:          %.1f%%\n", config->stop_loss_pct * 100);
    printf("Jito Endpoint:      %s\n", config->jito_endpoint);
    printf("Jito Tip:           %lu lamports\n", (unsigned long)config->jito_tip_lamports);
    printf("GMGN Cookies:       %s\n",
           strlen(config->gmgn_cf_clearance) > 0 ? "configured" : "NOT SET");
    printf("================================\n\n");
}
