/**
 * @file jito_client_internal.h
 * @brief Internal shared state for Jito client split files
 *
 * Exposes CURL buffer type, write callback, and time utilities shared
 * between jito_submit.c (bundle submission) and jito_status.c (status checking).
 *
 * Dependencies: <curl/curl.h>, <stdint.h>, "jito_client.h"
 *
 * @date 2026-03-05
 */

#ifndef JITO_CLIENT_INTERNAL_H
#define JITO_CLIENT_INTERNAL_H

#include <stdint.h>
#include <curl/curl.h>

#include "jito_client.h"

/* CURL response buffer */
typedef struct {
    char *data;
    size_t size;
} jito_curl_buffer_t;

/**
 * @brief CURL write callback for Jito client
 */
size_t jito_curl_write_cb(void *contents, size_t size, size_t nmemb, void *userp);

/**
 * @brief Get current time in milliseconds
 */
int64_t jito_get_time_ms(void);

#endif /* JITO_CLIENT_INTERNAL_H */
