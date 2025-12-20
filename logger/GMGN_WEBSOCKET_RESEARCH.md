# GMGN Trenches WebSocket Research and C Logger Implementation Guide

## Executive Summary

GMGN Trenches uses **WebSocket** connections (not simple HTTP requests) to stream real-time token data. This document outlines the WebSocket architecture, data structures, and implementation plan for a C-based logger that tracks newly minted Solana tokens with customizable filters.

---

## Table of Contents

1. [WebSocket Architecture Overview](#websocket-architecture-overview)
2. [GMGN API Connection Details](#gmgn-api-connection-details)
3. [Available Data Channels](#available-data-channels)
4. [Data Models and Structures](#data-models-and-structures)
5. [Filter Parameters](#filter-parameters)
6. [C Logger Implementation Plan](#c-logger-implementation-plan)
7. [Code Structure](#code-structure)
8. [References and Sources](#references-and-sources)

---

## WebSocket Architecture Overview

### Why WebSocket?

GMGN Trenches delivers real-time token data through persistent WebSocket connections rather than polling HTTP endpoints. This provides:

- **Low latency**: Immediate data delivery when new tokens are created
- **Reduced overhead**: No repeated HTTP handshakes
- **Bidirectional communication**: Subscribe/unsubscribe to specific channels
- **Persistent connection**: Automatic reconnection with exponential backoff

### Connection Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     GMGN WebSocket Architecture                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌──────────────┐    WebSocket     ┌─────────────────────────────┐ │
│   │   C Logger   │◄──────────────►  │  wss://gmgn.ai/ws           │ │
│   └──────────────┘     (TLS 1.3)    └─────────────────────────────┘ │
│         │                                        │                   │
│         ▼                                        ▼                   │
│   ┌──────────────┐                  ┌─────────────────────────────┐ │
│   │  Filter &    │                  │   Data Channels:            │ │
│   │  Process     │                  │   - new_pools               │ │
│   │  Messages    │                  │   - pair_update             │ │
│   └──────────────┘                  │   - token_launch            │ │
│         │                           │   - wallet_trades (auth)    │ │
│         ▼                           └─────────────────────────────┘ │
│   ┌──────────────┐                                                   │
│   │  Log/Export  │                                                   │
│   │  Results     │                                                   │
│   └──────────────┘                                                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## GMGN API Connection Details

### WebSocket Endpoint

```
Primary:     wss://gmgn.ai/ws
Chain Param: wss://gmgn.ai/ws?chain=sol
```

### Connection Parameters

| Parameter                | Value                | Description                           |
|--------------------------|----------------------|---------------------------------------|
| websocket_url            | wss://gmgn.ai/ws     | Primary WebSocket endpoint            |
| access_token             | Optional             | Required for authenticated channels   |
| max_reconnect_attempts   | 10                   | Maximum reconnection attempts         |
| reconnect_delay          | 1.0s                 | Initial delay (exponential backoff)   |
| heartbeat_interval       | 30.0s                | Keep-alive ping interval              |

### Authentication

Some channels require authentication via GMGN access token:

```c
/* Authentication header format */
{
    "access_token": "your_gmgn_access_token_here"
}
```

**Obtaining Access Token:**
1. Create GMGN account at https://gmgn.ai
2. Navigate to Account Settings
3. Generate API token
4. Store securely (environment variable recommended)

---

## Available Data Channels

### Public Channels (No Authentication Required)

| Channel          | Description                              | Data Type          |
|------------------|------------------------------------------|--------------------|
| `new_pools`      | New liquidity pool creation events       | NewPoolInfo        |
| `pair_update`    | Trading pair price and volume updates    | PairUpdateData     |
| `token_launch`   | New token launch notifications           | TokenLaunchData    |
| `chain_stats`    | Blockchain statistics and metrics        | ChainStatsData     |

### Authenticated Channels (Require Access Token)

| Channel          | Description                              | Data Type          |
|------------------|------------------------------------------|--------------------|
| `token_social`   | Token social media and community info    | TokenSocialData    |
| `wallet_trades`  | Wallet trading activity                  | WalletTradeData    |
| `limit_orders`   | Limit order updates and fills            | LimitOrderData     |

### Subscription Messages

**Subscribe to new_pools:**
```json
{
    "action": "subscribe",
    "channel": "new_pools",
    "chain": "sol"
}
```

**Subscribe to token updates:**
```json
{
    "action": "subscribe",
    "channel": "pair_update",
    "tokens": ["EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"],
    "chain": "sol"
}
```

---

## Data Models and Structures

### NewPoolInfo (Primary Data for New Token Detection)

```json
{
    "c": "sol",                    /* Chain identifier */
    "rg": "us-east",               /* Region */
    "p": [                         /* Pools array */
        {
            "a": "pool_address",   /* Pool address */
            "ex": "raydium",       /* Exchange name */
            "ba": "base_token_addr",    /* Base token address */
            "qa": "quote_token_addr",   /* Quote token address */
            "il": 50000.00,        /* Initial liquidity (USD) */
            "bti": {               /* Base Token Info */
                "s": "TOKEN",      /* Symbol */
                "n": "Token Name", /* Name */
                "mc": 150000,      /* Market cap (USD) */
                "v24h": 50000.0,   /* 24h volume (USD) */
                "p": 0.00015,      /* Price (USD) */
                "hc": 500,         /* Holder count */
                "t10hr": 0.45,     /* Top 10 holders ratio (0-1) */
                "cbr": 0.1         /* Creator balance ratio */
            }
        }
    ]
}
```

### C Structure Definitions

```c
/**
 * @brief Token information structure
 */
typedef struct {
    char symbol[16];               /* Token symbol (e.g., "SOL") */
    char name[64];                 /* Token full name */
    uint64_t market_cap;           /* Market cap in USD cents */
    uint64_t volume_24h;           /* 24h volume in USD cents */
    uint64_t price;                /* Price in fixed-point (1e-8) */
    uint32_t holder_count;         /* Number of token holders */
    uint16_t top_10_ratio;         /* Top 10 holders % (0-10000) */
    uint16_t creator_balance_ratio;/* Creator balance % (0-10000) */
    uint32_t age_seconds;          /* Token age in seconds */
    uint8_t kol_count;             /* KOL/influencer count */
} token_info_t;

/**
 * @brief Pool data structure
 */
typedef struct {
    char pool_address[64];         /* Pool contract address */
    char base_token_addr[64];      /* Base token address */
    char quote_token_addr[64];     /* Quote token address */
    char exchange[32];             /* Exchange name */
    uint64_t initial_liquidity;    /* Initial liquidity (USD cents) */
    token_info_t base_token;       /* Base token info */
    time_t created_at;             /* Creation timestamp */
} pool_data_t;

/**
 * @brief Filter configuration structure
 */
typedef struct {
    uint64_t min_market_cap;       /* Minimum market cap (USD cents) */
    uint64_t max_market_cap;       /* Maximum market cap (USD cents) */
    uint64_t min_liquidity;        /* Minimum liquidity (USD cents) */
    uint64_t min_volume_24h;       /* Minimum 24h volume (USD cents) */
    uint32_t min_holder_count;     /* Minimum holder count */
    uint32_t max_age_seconds;      /* Maximum token age */
    uint8_t min_kol_count;         /* Minimum KOL count */
    uint16_t max_top_10_ratio;     /* Max top 10 holder % (0-10000) */
    uint16_t max_creator_ratio;    /* Max creator balance % (0-10000) */
    char exchanges[8][32];         /* Allowed exchanges (empty = all) */
    uint8_t exchange_count;        /* Number of allowed exchanges */
    char exclude_symbols[16][16];  /* Symbols to exclude */
    uint8_t exclude_count;         /* Number of excluded symbols */
} filter_config_t;
```

---

## Filter Parameters

### Requested Filter Criteria

Based on the user requirements, here are the configurable filter parameters:

| Filter           | Example Value       | Description                          |
|------------------|---------------------|--------------------------------------|
| `min_market_cap` | $5,500 (5500 USD)   | Minimum market cap                   |
| `max_market_cap` | $10,000 (10000 USD) | Maximum market cap                   |
| `min_kol_count`  | 1                   | Minimum KOL (Key Opinion Leader)     |
| `max_age`        | 10 minutes          | Maximum token age                    |
| `min_liquidity`  | $5,000              | Minimum liquidity requirement        |
| `min_holders`    | 10                  | Minimum holder count                 |
| `max_top_10`     | 70%                 | Max concentration in top 10 holders  |
| `exchanges`      | raydium, orca       | Allowed exchanges                    |

### Filter Configuration Example

```c
/**
 * @brief Initialize filter with user requirements
 */
filter_config_t create_default_filter(void) {
    filter_config_t filter = {0};
    
    /* Market cap range: $5.5K - $10K */
    filter.min_market_cap = 550000;     /* $5,500 in cents */
    filter.max_market_cap = 1000000;    /* $10,000 in cents */
    
    /* Minimum 1 KOL */
    filter.min_kol_count = 1;
    
    /* Maximum age: 10 minutes */
    filter.max_age_seconds = 600;
    
    /* Minimum liquidity: $5K */
    filter.min_liquidity = 500000;
    
    /* Minimum 10 holders */
    filter.min_holder_count = 10;
    
    /* Max top 10 concentration: 70% */
    filter.max_top_10_ratio = 7000;     /* 70.00% */
    
    /* Trusted exchanges */
    strncpy(filter.exchanges[0], "raydium", 31);
    strncpy(filter.exchanges[1], "orca", 31);
    strncpy(filter.exchanges[2], "meteora", 31);
    filter.exchange_count = 3;
    
    /* Exclude suspicious symbols */
    strncpy(filter.exclude_symbols[0], "SCAM", 15);
    strncpy(filter.exclude_symbols[1], "TEST", 15);
    strncpy(filter.exclude_symbols[2], "FAKE", 15);
    filter.exclude_count = 3;
    
    return filter;
}
```

---

## C Logger Implementation Plan

### Module Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    C Logger Module Architecture                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌───────────────────────────────────────────────────────────────┐ │
│   │                      Main Application                          │ │
│   │                     (gmgn_logger.c)                            │ │
│   └───────────────────────────────────────────────────────────────┘ │
│                              │                                       │
│         ┌────────────────────┼────────────────────┐                 │
│         ▼                    ▼                    ▼                 │
│   ┌───────────┐       ┌───────────┐       ┌───────────┐            │
│   │ WebSocket │       │  Filter   │       │   Output  │            │
│   │  Client   │       │  Engine   │       │  Handler  │            │
│   │(websocket │       │(filter.c) │       │(output.c) │            │
│   │  _client  │       └───────────┘       └───────────┘            │
│   │   .c)     │                                                     │
│   └───────────┘                                                     │
│         │                                                           │
│         ▼                                                           │
│   ┌───────────┐       ┌───────────┐       ┌───────────┐            │
│   │   JSON    │       │  Config   │       │  Logger   │            │
│   │  Parser   │       │  Parser   │       │ (Syslog/  │            │
│   │(json.c)   │       │(config.c) │       │  File)    │            │
│   └───────────┘       └───────────┘       └───────────┘            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Required Libraries

| Library          | Purpose                          | License     |
|------------------|----------------------------------|-------------|
| libwebsockets    | WebSocket client implementation  | MIT         |
| cJSON            | JSON parsing                     | MIT         |
| libssl/openssl   | TLS/SSL for secure connections   | Apache 2.0  |
| pthread          | Threading support                | LGPL        |

### File Structure

```
gmgn_trading/
├── src/
│   ├── gmgn_logger.c        # Main application entry point
│   ├── websocket_client.c   # WebSocket connection management
│   ├── filter.c             # Token filtering logic
│   ├── json_parser.c        # JSON message parsing
│   ├── config.c             # Configuration file parsing
│   ├── output.c             # Output formatting and logging
│   └── utils.c              # Utility functions
├── include/
│   ├── gmgn_logger.h        # Main header with type definitions
│   ├── websocket_client.h   # WebSocket client interface
│   ├── filter.h             # Filter function declarations
│   ├── json_parser.h        # JSON parser interface
│   ├── config.h             # Configuration structures
│   ├── output.h             # Output handler interface
│   └── utils.h              # Utility declarations
├── config/
│   └── gmgn_logger.conf     # Runtime configuration file
├── scripts/
│   ├── Makefile             # Build configuration
│   └── install.sh           # Installation script
└── tests/
    ├── test_filter.c        # Filter unit tests
    └── test_json_parser.c   # JSON parser tests
```

---

## Code Structure

### Main Application (gmgn_logger.c)

```c
/**
 * @file gmgn_logger.c
 * @brief GMGN Trenches token logger main application
 *
 * This application connects to GMGN.ai WebSocket API to monitor
 * newly created Solana tokens in real-time. It applies configurable
 * filters and logs matching tokens for further analysis.
 *
 * Dependencies: libwebsockets, cJSON, openssl, pthread
 *
 * @date 2025-12-20
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <time.h>

#include "gmgn_logger.h"
#include "websocket_client.h"
#include "filter.h"
#include "config.h"
#include "output.h"

static volatile sig_atomic_t g_running = 1;

/**
 * @brief Signal handler for graceful shutdown
 *
 * @param signum Signal number received
 */
static void signal_handler(int signum) {
    (void)signum;
    g_running = 0;
}

/**
 * @brief Callback for incoming pool data
 *
 * @param pool Pointer to pool data structure
 * @param filter Pointer to active filter configuration
 *
 * @return 0 on success, -1 on error
 */
static int on_new_pool(const pool_data_t *pool, const filter_config_t *filter) {
    if (!pool || !filter) {
        return -1;
    }
    
    /* Apply filters */
    if (!filter_check_pool(pool, filter)) {
        return 0;  /* Filtered out, not an error */
    }
    
    /* Log matching token */
    output_log_token(&pool->base_token, pool);
    
    return 0;
}

/**
 * @brief Main application entry point
 *
 * @param argc Argument count
 * @param argv Argument vector
 *
 * @return EXIT_SUCCESS on normal termination, EXIT_FAILURE on error
 */
int main(int argc, char *argv[]) {
    ws_client_t *client = NULL;
    filter_config_t filter = {0};
    app_config_t config = {0};
    int ret = EXIT_SUCCESS;
    
    /* Setup signal handlers */
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    /* Parse command line and config file */
    if (config_parse(argc, argv, &config) != 0) {
        fprintf(stderr, "Failed to parse configuration\n");
        return EXIT_FAILURE;
    }
    
    /* Initialize filter from config */
    filter = config_create_filter(&config);
    
    /* Initialize WebSocket client */
    client = ws_client_create(config.websocket_url, config.access_token);
    if (!client) {
        fprintf(stderr, "Failed to create WebSocket client\n");
        return EXIT_FAILURE;
    }
    
    /* Set callbacks */
    ws_client_set_pool_callback(client, on_new_pool, &filter);
    
    /* Connect and subscribe */
    if (ws_client_connect(client) != 0) {
        fprintf(stderr, "Failed to connect to GMGN WebSocket\n");
        ret = EXIT_FAILURE;
        goto cleanup;
    }
    
    /* Subscribe to new pools channel */
    if (ws_client_subscribe(client, "new_pools", "sol") != 0) {
        fprintf(stderr, "Failed to subscribe to new_pools\n");
        ret = EXIT_FAILURE;
        goto cleanup;
    }
    
    /* Main event loop */
    while (g_running) {
        if (ws_client_service(client, 1000) < 0) {
            fprintf(stderr, "WebSocket service error\n");
            ret = EXIT_FAILURE;
            break;
        }
    }
    
cleanup:
    ws_client_destroy(client);
    config_cleanup(&config);
    
    return ret;
}
```

### Filter Implementation (filter.c)

```c
/**
 * @file filter.c
 * @brief Token filtering implementation
 *
 * Implements filtering logic for new token pools based on
 * configurable criteria including market cap, liquidity,
 * holder count, KOL presence, and token age.
 *
 * @date 2025-12-20
 */

#include <string.h>
#include <time.h>
#include <stdbool.h>

#include "filter.h"

/**
 * @brief Check if symbol is in exclude list
 *
 * @param symbol Token symbol to check
 * @param filter Filter configuration
 *
 * @return true if symbol should be excluded
 */
static bool is_excluded_symbol(const char *symbol, const filter_config_t *filter) {
    for (uint8_t i = 0; i < filter->exclude_count; i++) {
        if (strstr(symbol, filter->exclude_symbols[i]) != NULL) {
            return true;
        }
    }
    return false;
}

/**
 * @brief Check if exchange is allowed
 *
 * @param exchange Exchange name
 * @param filter Filter configuration
 *
 * @return true if exchange is allowed (or no filter set)
 */
static bool is_allowed_exchange(const char *exchange, const filter_config_t *filter) {
    if (filter->exchange_count == 0) {
        return true;  /* No exchange filter, allow all */
    }
    
    for (uint8_t i = 0; i < filter->exchange_count; i++) {
        if (strcmp(exchange, filter->exchanges[i]) == 0) {
            return true;
        }
    }
    return false;
}

/**
 * @brief Check if pool passes all filter criteria
 *
 * @param pool Pool data to check
 * @param filter Filter configuration
 *
 * @return true if pool passes all filters, false otherwise
 */
bool filter_check_pool(const pool_data_t *pool, const filter_config_t *filter) {
    const token_info_t *token = &pool->base_token;
    
    /* Check excluded symbols */
    if (is_excluded_symbol(token->symbol, filter)) {
        return false;
    }
    
    /* Check exchange whitelist */
    if (!is_allowed_exchange(pool->exchange, filter)) {
        return false;
    }
    
    /* Market cap range */
    if (filter->min_market_cap > 0 && token->market_cap < filter->min_market_cap) {
        return false;
    }
    if (filter->max_market_cap > 0 && token->market_cap > filter->max_market_cap) {
        return false;
    }
    
    /* Liquidity minimum */
    if (filter->min_liquidity > 0 && pool->initial_liquidity < filter->min_liquidity) {
        return false;
    }
    
    /* 24h volume minimum */
    if (filter->min_volume_24h > 0 && token->volume_24h < filter->min_volume_24h) {
        return false;
    }
    
    /* Holder count minimum */
    if (filter->min_holder_count > 0 && token->holder_count < filter->min_holder_count) {
        return false;
    }
    
    /* Token age maximum */
    if (filter->max_age_seconds > 0 && token->age_seconds > filter->max_age_seconds) {
        return false;
    }
    
    /* KOL count minimum */
    if (filter->min_kol_count > 0 && token->kol_count < filter->min_kol_count) {
        return false;
    }
    
    /* Top 10 holders concentration */
    if (filter->max_top_10_ratio > 0 && token->top_10_ratio > filter->max_top_10_ratio) {
        return false;
    }
    
    /* Creator balance ratio */
    if (filter->max_creator_ratio > 0 && 
        token->creator_balance_ratio > filter->max_creator_ratio) {
        return false;
    }
    
    return true;
}
```

### Configuration File Format (gmgn_logger.conf)

```ini
# GMGN Logger Configuration
# All values use USD for monetary amounts

[connection]
websocket_url = wss://gmgn.ai/ws
chain = sol
access_token = 
reconnect_attempts = 10
reconnect_delay_ms = 1000
heartbeat_interval_ms = 30000

[filter]
# Market cap range (USD)
min_market_cap = 5500
max_market_cap = 10000

# Liquidity minimum (USD)
min_liquidity = 5000

# 24h volume minimum (USD)
min_volume_24h = 1000

# Holder requirements
min_holder_count = 10
max_top_10_ratio = 70

# Token age (seconds)
max_age_seconds = 600

# KOL/influencer count
min_kol_count = 1

# Maximum creator balance percentage
max_creator_ratio = 50

# Allowed exchanges (comma-separated, empty = all)
exchanges = raydium,orca,meteora

# Excluded symbols (comma-separated)
exclude_symbols = SCAM,TEST,FAKE,RUG

[output]
# Output format: console, json, csv, sqlite
format = console

# Log file path (for file-based output)
log_file = ./logs/gmgn_tokens.log

# Enable timestamps
timestamps = true

# Verbosity: 0=minimal, 1=normal, 2=verbose
verbosity = 1
```

---

## Build Instructions

### Makefile

```makefile
# GMGN Logger Makefile

CC = gcc
CFLAGS = -Wall -Wextra -Werror -O2 -g
LDFLAGS = -lwebsockets -lcjson -lssl -lcrypto -lpthread

SRC_DIR = src
INC_DIR = include
BUILD_DIR = build

SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRCS))
TARGET = $(BUILD_DIR)/gmgn_logger

.PHONY: all clean install

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(INC_DIR) -c $< -o $@

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)

install: $(TARGET)
	install -m 755 $(TARGET) /usr/local/bin/

debug: CFLAGS += -DDEBUG -fsanitize=address
debug: clean $(TARGET)
```

### Building

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt-get install libwebsockets-dev libcjson-dev libssl-dev

# Build
make

# Build with debug/sanitizers
make debug

# Install system-wide
sudo make install
```

---

## References and Sources

### Primary Sources

1. **ChipaDevTeam/GmGnAPI** - Official Python WebSocket client
   - URL: https://github.com/ChipaDevTeam/GmGnAPI
   - Documentation: https://chipadevteam.github.io/GmGnAPI/
   - Key insight: WebSocket URL `wss://gmgn.ai/ws`, channel subscriptions

2. **1f1n/gmgnai-wrapper** - Python API wrapper (136 stars)
   - URL: https://github.com/1f1n/gmgnai-wrapper
   - Key insight: Undocumented endpoint patterns, Cloudflare protection notes

3. **Malakia-sol/gmgn-api** - JavaScript client with cookie auth
   - URL: https://github.com/Malakia-sol/gmgn-api
   - Key insight: Cloudflare bypass using browser cookies

### Key Technical Details from Sources

| Source          | Key Information                                           |
|-----------------|-----------------------------------------------------------|
| GmGnAPI docs    | WebSocket URL: `wss://gmgn.ai/ws`                        |
| GmGnAPI docs    | Channels: new_pools, pair_update, token_launch            |
| GmGnAPI docs    | Data models: NewPoolInfo, PoolData, TokenInfo            |
| GmGnAPI docs    | Filter: TokenFilter with market_cap, liquidity, etc.     |
| GmGnAPI docs    | Auth: access_token for wallet_trades channel              |
| gmgnai-wrapper  | Cloudflare protection requires proper headers             |
| gmgn-api (JS)   | cf_clearance cookie needed for HTTP endpoints            |

### Additional Notes

- GMGN.ai is protected by Cloudflare; WebSocket connections may require proper headers
- Token authentication is optional for public channels (new_pools)
- Heartbeat/ping must be sent every 30 seconds to maintain connection
- Automatic reconnection with exponential backoff is essential for production use
- Data format uses abbreviated field names (c=chain, p=pools, bti=base_token_info)

---

## Next Steps

1. **Implement websocket_client.c** - Core WebSocket connection using libwebsockets
2. **Implement json_parser.c** - Parse GMGN JSON messages using cJSON
3. **Implement filter.c** - Token filtering with all criteria
4. **Create test suite** - Unit tests for filter and parser modules
5. **Add configuration parsing** - INI file parser for runtime config
6. **Integrate output handlers** - Console, JSON, CSV, SQLite output options

---

*Document created: 2025-12-20*
*Based on research of public GMGN API implementations*
