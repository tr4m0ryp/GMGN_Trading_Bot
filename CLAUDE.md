# Claude Code Development Guidelines

## Primary Language: C

This project focuses on **C programming** as the primary language due to its superior performance for computational operations. C provides the fastest execution speed for calculation-intensive tasks, making it ideal for trading algorithms and financial computations.

## Project Structure

The project follows a standard organized C project structure:

```
gmgn_trading/
├── src/           # Source files (.c)
├── include/       # Header files (.h)
├── build/         # Compiled binaries and object files
├── lib/           # External libraries
├── tests/         # Test files
├── scripts/       # Build and utility scripts
└── docs/          # Documentation (minimal, avoid creating unless necessary)
```

### Directory Guidelines
- **src/**: All implementation files (.c) go here
- **include/**: All header files (.h) go here
- **build/**: Output directory for compiled artifacts (do not commit to git)
- **lib/**: Third-party libraries and dependencies
- **tests/**: Unit tests and integration tests
- **scripts/**: Makefile, build scripts, deployment scripts

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
