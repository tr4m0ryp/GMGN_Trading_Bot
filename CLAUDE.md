# Claude Code Development Guidelines

## Primary Language: C

This project focuses on **C programming** as the primary language due to its superior performance for computational operations. C provides the fastest execution speed for calculation-intensive tasks, making it ideal for trading algorithms and financial computations.

## Project Overview

This is a **GMGN Trading Bot** project with multiple components:

1. **logger_c/** - Real-time token logger that connects to GMGN.ai WebSocket API
2. **trading_algorithm/** - (Future) Mathematical trading algorithm based on logger analysis
3. **logger/** - (Legacy) JavaScript-based network capture utilities

## Project Structure

```
gmgn_trading/
├── CLAUDE.md              # This file - development guidelines
├── README.md              # Project overview
├── config/                # Configuration files
├── logger_c/              # C-based real-time token logger
│   ├── src/               # Source files (.c)
│   ├── include/           # Header files (.h)
│   ├── build/             # Compiled binaries
│   └── Makefile           # Build configuration
├── trading_algorithm/     # (Future) Trading algorithm component
├── data_analysis/         # (Future) Token analysis and ML models
└── logger/                # Legacy JS network capture tools
```

### Component Directories

- **logger_c/**: The C-based GMGN token logger - connects to WebSocket, filters tokens
- **trading_algorithm/**: (Future) Trading decision engine using mathematical models
- **data_analysis/**: (Future) Historical data analysis and pattern recognition

---

## GMGN WebSocket API - Lessons Learned

### Critical Information for Future Development

This section documents issues encountered and solutions found when working with the GMGN.ai WebSocket API.

### 1. WebSocket Connection URL and Parameters

**Correct WebSocket URL:**
```
wss://ws.gmgn.ai/quotation
```

**Required Query Parameters:**
```
device_id, client_id, from_app, app_ver, tz_name, tz_offset, app_lang, fp_did, os, uuid
```

Example path:
```
/quotation?device_id=xxx&client_id=gmgn_python_xxx&from_app=gmgn&app_ver=20250729-1647-ffac485&tz_name=UTC&tz_offset=0&app_lang=en-US&fp_did=xxx&os=linux&uuid=xxx
```

### 2. Subscription Message Format (CRITICAL)

**WRONG format (will not receive data):**
```json
{"channel":"new_pool_info","data":[{"chain":"sol"}]}
```

**CORRECT format (required fields):**
```json
{
  "action": "subscribe",
  "channel": "new_pool_info",
  "f": "w",
  "id": "gmgn_00000001",
  "data": [{"chain": "sol"}]
}
```

Required fields:
- `action`: Must be `"subscribe"`
- `channel`: Channel name (e.g., `"new_pool_info"`)
- `f`: Must be `"w"` (unknown purpose, but required)
- `id`: Unique message ID for acknowledgment tracking
- `data`: Array with chain specification

### 3. Channel Names

| Description | Correct Channel Name |
|-------------|---------------------|
| New pools | `new_pool_info` |
| Pair updates | `new_pair_update` |
| Token launches | `new_launched_info` |
| Chain stats | `chain_stat` |
| Wallet trades | `wallet_trade_data` |

### 4. Data Structure for new_pool_info

The response structure:
```json
{
  "channel": "new_pool_info",
  "data": [
    {
      "c": "sol",
      "rg": "3",
      "p": [
        {
          "a": "pool_address",
          "ex": "pump",
          "ba": "base_token_address",
          "bti": {
            "s": "SYMBOL",
            "n": "Token Name",
            "mc": 379140
          }
        }
      ]
    }
  ]
}
```

**Key insight:** The `data` field is an **array**, not an object. Access pools via `data[0].p`.

### 5. Required HTTP Headers

```
Origin: https://gmgn.ai
User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36
```

### 6. libwebsockets Specifics (C Implementation)

- Use `lws_create_context()` not `lws_context_create()` (API renamed in v4.x)
- Set `pt_serv_buf_size = 8192` or higher for long query strings
- Pass `client` pointer directly to `userdata`, not `&client` (stack address issue)
- Use `LWS_CALLBACK_CLIENT_APPEND_HANDSHAKE_HEADER` to add custom headers

---

## Code Quality Standards

### Professional Development Practices
- Act as a professional high-end developer
- Follow strict code rules and best practices
- Thoroughly check all code before submission
- Ensure code is production-ready and maintainable
- Verify memory management (no leaks, proper allocation/deallocation)
- Check for buffer overflows and security vulnerabilities
- Ensure thread safety where applicable

## C Coding Standards

### Naming Conventions
- **Functions**: lowercase with underscores (snake_case): `calculate_profit_margin()`
- **Variables**: lowercase with underscores: `total_price`, `order_count`
- **Constants**: uppercase with underscores: `MAX_BUFFER_SIZE`, `PI`
- **Macros**: uppercase with underscores: `#define ARRAY_SIZE(x)`
- **Struct/Typedef**: PascalCase or suffix with _t: `OrderBook`, `trade_data_t`
- **Global variables**: prefix with g_: `g_connection_pool`
- **Static variables**: prefix with s_: `s_instance_count`

### Code Style Rules
- **Indentation**: 4 spaces (no tabs)
- **Braces**: K&R style (opening brace on same line for functions, control structures)
- **Line length**: Maximum 100 characters
- **NO emojis** in code, comments, or commit messages
- One statement per line
- Space after keywords: `if (condition)`, `while (condition)`
- No space between function name and parenthesis: `function_name(args)`
- Pointer asterisk with type: `char *ptr` not `char* ptr`

### Memory Management
- Always check return values of `malloc()`, `calloc()`, `realloc()`
- Always free allocated memory when done
- Set pointers to NULL after freeing
- Avoid memory leaks and dangling pointers
- Use `valgrind` or similar tools to verify

### Error Handling
- Check all return values from functions
- Use errno for system call errors
- Return meaningful error codes
- Log errors appropriately
- Clean up resources in error paths

### Header Guards
All header files must use include guards:
```c
#ifndef PROJECT_MODULE_H
#define PROJECT_MODULE_H

/* Header content */

#endif /* PROJECT_MODULE_H */
```

## Code Documentation Requirements

### File Header Comments

Every file must include a header comment block at the top:

**Sample Header File Comment:**
```c
/**
 * @file order_book.h
 * @brief Order book data structures and management functions
 *
 * This header defines the order book structure for managing buy/sell orders
 * in the trading system. It provides functions for order insertion, removal,
 * matching, and price level management.
 *
 * Dependencies: <stdint.h>, <stdbool.h>, "common_types.h"
 *
 * @date 2025-12-20
 */
```

**Sample Source File Comment:**
```c
/**
 * @file price_calculator.c
 * @brief Implementation of price calculation algorithms
 *
 * This module implements high-performance price calculation functions
 * including VWAP, TWAP, moving averages, and profit margin calculations.
 * Optimized for speed using bitwise operations and lookup tables where
 * applicable.
 *
 * Dependencies: <math.h>, <stdint.h>, "price_calculator.h"
 *
 * @date 2025-12-20
 */
```

### Function Comments

Every function must include a comment block explaining its purpose, parameters, and return value:

**Sample Function Comment:**
```c
/**
 * @brief Calculate the volume-weighted average price (VWAP)
 *
 * Computes the VWAP for a given array of trades over a specified time period.
 * The calculation is optimized using fixed-point arithmetic to avoid floating
 * point precision issues in financial calculations.
 *
 * @param trades Array of trade structures containing price and volume data
 * @param count Number of trades in the array (must be > 0)
 * @param start_time Unix timestamp for calculation window start
 * @param end_time Unix timestamp for calculation window end
 *
 * @return VWAP as a 64-bit integer (multiply by 1e-8 for actual price)
 *         Returns 0 if no trades in time window or on error
 *
 * @note This function assumes trades are sorted by timestamp
 * @warning Returns 0 on error - check errno for details
 */
uint64_t calculate_vwap(const trade_t *trades, size_t count,
                        time_t start_time, time_t end_time);
```

**Sample Simple Function Comment:**
```c
/**
 * @brief Initialize an order book structure
 *
 * Allocates and initializes a new order book with default parameters.
 * Caller is responsible for calling destroy_order_book() when done.
 *
 * @param symbol Trading symbol (e.g., "BTC/USD"), must not be NULL
 *
 * @return Pointer to initialized order book, or NULL on allocation failure
 */
order_book_t *create_order_book(const char *symbol);
```

### Inline Comments

Use inline comments sparingly, only when the code logic is not self-evident:

```c
/* Fast bit-count using Brian Kernighan's algorithm */
while (n) {
    n &= (n - 1);  /* Clear the lowest set bit */
    count++;
}
```

## File Creation Policy

- **AVOID** creating markdown (.md) files unless explicitly requested
- Minimize unnecessary file creation to reduce token usage
- Prefer editing existing files over creating new ones
- Only create new .c/.h files when adding genuinely new modules

## Compilation and Build

- Use `-Wall -Wextra -Werror` for strict error checking
- Use `-O2` or `-O3` for optimized builds
- Use `-g` for debug builds
- Use appropriate sanitizers: `-fsanitize=address` for memory issues

## Goal

All code should be easily understandable by developers unfamiliar with the project. Documentation and code clarity are paramount. Every function should be self-documenting through clear naming, and supported by concise comments that explain the "why" rather than the "what".
