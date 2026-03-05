/**
 * @file chart_fetcher_internal.h
 * @brief Internal shared state for chart fetcher split files
 *
 * Exposes CURL handle, headers, and helper types shared between
 * chart_fetch.c (network + parsing) and chart_access.c (data access).
 *
 * Dependencies: <curl/curl.h>, <stdbool.h>, "chart_fetcher.h"
 *
 * @date 2026-03-05
 */

#ifndef CHART_FETCHER_INTERNAL_H
#define CHART_FETCHER_INTERNAL_H

#include <stdbool.h>
#include <curl/curl.h>

#include "chart_fetcher.h"

/* CURL response buffer */
typedef struct {
    char *data;
    size_t size;
} chart_curl_buffer_t;

/* Global CURL handle (persistent connection) */
extern CURL *g_chart_curl;
extern struct curl_slist *g_chart_headers;
extern bool g_chart_initialized;

/**
 * @brief CURL write callback for chart fetcher
 */
size_t chart_curl_write_cb(void *contents, size_t size, size_t nmemb, void *userp);

#endif /* CHART_FETCHER_INTERNAL_H */
