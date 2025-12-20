/**
 * @file compat.h
 * @brief Platform compatibility definitions
 *
 * Provides compatibility macros and functions for cross-platform builds.
 * Include this header in files that use GNU-specific functions.
 *
 * @date 2025-12-20
 */

#ifndef COMPAT_H
#define COMPAT_H

/* macOS does not have strcasestr - provide fallback */
#ifdef __APPLE__
#include <string.h>
#include <ctype.h>

static inline char *strcasestr(const char *haystack, const char *needle) {
    if (!needle[0]) {
        return (char *)haystack;
    }
    for (; *haystack; haystack++) {
        const char *h = haystack;
        const char *n = needle;
        while (*h && *n && tolower((unsigned char)*h) == tolower((unsigned char)*n)) {
            h++;
            n++;
        }
        if (!*n) {
            return (char *)haystack;
        }
    }
    return NULL;
}
#endif /* __APPLE__ */

#endif /* COMPAT_H */
