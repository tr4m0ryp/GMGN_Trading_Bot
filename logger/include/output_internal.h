/**
 * @file output_internal.h
 * @brief Internal shared state for output module
 *
 * Exposes ANSI color codes, static state variables, and helper
 * functions shared across the split output source files. Not part
 * of the public API.
 *
 * Dependencies: "output.h", "filter.h"
 *
 * @date 2025-12-20
 */

#ifndef OUTPUT_INTERNAL_H
#define OUTPUT_INTERNAL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <inttypes.h>

#include "output.h"
#include "filter.h"

/* ANSI color codes */
#define ANSI_RESET      "\033[0m"
#define ANSI_BOLD       "\033[1m"
#define ANSI_DIM        "\033[2m"
#define ANSI_RED        "\033[31m"
#define ANSI_GREEN      "\033[32m"
#define ANSI_YELLOW     "\033[33m"
#define ANSI_BLUE       "\033[34m"
#define ANSI_MAGENTA    "\033[35m"
#define ANSI_CYAN       "\033[36m"
#define ANSI_WHITE      "\033[37m"
#define ANSI_BG_RED     "\033[41m"
#define ANSI_BG_GREEN   "\033[42m"
#define ANSI_BG_BLUE    "\033[44m"

/* Shared output module state */
extern output_verbosity_t g_output_verbosity;
extern int g_output_use_colors;
extern uint64_t g_output_tokens_logged;

/**
 * @brief Check if terminal supports colors
 *
 * @return 1 if colors supported, 0 otherwise
 */
int output_check_color_support(void);

/**
 * @brief Get current timestamp string formatted as HH:MM:SS
 *
 * @param buffer Output buffer for timestamp
 * @param size Size of output buffer
 */
void output_get_timestamp(char *buffer, size_t size);

/**
 * @brief Print colored text (respects color support flag)
 *
 * @param color ANSI color escape sequence
 * @param text Text to print in color
 */
void output_print_color(const char *color, const char *text);

#endif /* OUTPUT_INTERNAL_H */
