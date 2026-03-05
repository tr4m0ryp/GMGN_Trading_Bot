/**
 * @file output_init.c
 * @brief Output system initialization, color support, and shared state
 *
 * Implements the output module's global state definitions, initialization,
 * cleanup, color detection, and shared helper functions.
 *
 * Dependencies: "output_internal.h"
 *
 * @date 2025-12-20
 */

#include "output_internal.h"

/* Shared output module state definitions */
output_verbosity_t g_output_verbosity = OUTPUT_NORMAL;
int g_output_use_colors = 1;
uint64_t g_output_tokens_logged = 0;

int output_check_color_support(void) {
    const char *term = getenv("TERM");
    if (!term) {
        return 0;
    }

    if (strstr(term, "color") || strstr(term, "xterm") ||
        strstr(term, "256") || strstr(term, "screen") ||
        strstr(term, "tmux")) {
        return 1;
    }

    return isatty(STDOUT_FILENO);
}

void output_get_timestamp(char *buffer, size_t size) {
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    strftime(buffer, size, "%H:%M:%S", tm_info);
}

void output_print_color(const char *color, const char *text) {
    if (g_output_use_colors) {
        printf("%s%s%s", color, text, ANSI_RESET);
    } else {
        printf("%s", text);
    }
}

int output_init(output_verbosity_t verbosity) {
    g_output_verbosity = verbosity;
    g_output_use_colors = output_check_color_support();
    g_output_tokens_logged = 0;

    /* Disable buffering for real-time output */
    setvbuf(stdout, NULL, _IONBF, 0);

    return 0;
}

void output_cleanup(void) {
    /* Nothing to clean up currently */
}

void output_set_verbosity(output_verbosity_t verbosity) {
    g_output_verbosity = verbosity;
}

output_verbosity_t output_get_verbosity(void) {
    return g_output_verbosity;
}
