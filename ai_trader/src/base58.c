/**
 * @file base58.c
 * @brief Base58 encoding/decoding implementation
 *
 * Uses the Bitcoin/Solana alphabet.
 *
 * @date 2025-12-24
 */

#include <string.h>
#include <stdlib.h>

#include "base58.h"

/* Bitcoin/Solana base58 alphabet */
static const char BASE58_ALPHABET[] = 
    "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

/* Reverse lookup table */
static const int8_t BASE58_MAP[128] = {
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1, 0, 1, 2, 3, 4, 5, 6, 7, 8,-1,-1,-1,-1,-1,-1, /* 0-9 */
    -1, 9,10,11,12,13,14,15,16,-1,17,18,19,20,21,-1, /* A-O */
    22,23,24,25,26,27,28,29,30,31,32,-1,-1,-1,-1,-1, /* P-Z */
    -1,33,34,35,36,37,38,39,40,41,42,43,-1,44,45,46, /* a-m */
    47,48,49,50,51,52,53,54,55,56,57,-1,-1,-1,-1,-1  /* n-z */
};

int base58_encode(const uint8_t *data, size_t data_len, char *out, size_t out_len) {
    if (!data || !out || out_len == 0) {
        return -1;
    }

    /* Count leading zeros */
    size_t leading_zeros = 0;
    while (leading_zeros < data_len && data[leading_zeros] == 0) {
        leading_zeros++;
    }

    /* Allocate enough space for base58 representation */
    /* base58 expands by roughly 138/100 */
    size_t size = (data_len - leading_zeros) * 138 / 100 + 1;
    uint8_t *buf = calloc(size, 1);
    if (!buf) return -1;

    /* Process each byte */
    for (size_t i = leading_zeros; i < data_len; i++) {
        int carry = data[i];
        for (size_t j = 0; j < size; j++) {
            carry += 256 * buf[size - 1 - j];
            buf[size - 1 - j] = carry % 58;
            carry /= 58;
        }
    }

    /* Skip leading zeros in buffer */
    size_t buf_start = 0;
    while (buf_start < size && buf[buf_start] == 0) {
        buf_start++;
    }

    /* Calculate output length */
    size_t result_len = leading_zeros + (size - buf_start);
    if (result_len >= out_len) {
        free(buf);
        return -1;  /* Buffer too small */
    }

    /* Build output string */
    size_t pos = 0;
    for (size_t i = 0; i < leading_zeros; i++) {
        out[pos++] = '1';
    }
    for (size_t i = buf_start; i < size; i++) {
        out[pos++] = BASE58_ALPHABET[buf[i]];
    }
    out[pos] = '\0';

    free(buf);
    return (int)pos;
}

int base58_decode(const char *str, uint8_t *out, size_t out_len) {
    if (!str || !out || out_len == 0) {
        return -1;
    }

    size_t str_len = strlen(str);
    if (str_len == 0) {
        return 0;
    }

    /* Count leading '1's (zeros in decoded) */
    size_t leading_ones = 0;
    while (str[leading_ones] == '1') {
        leading_ones++;
    }

    /* Allocate buffer for decoded bytes */
    size_t size = str_len * 733 / 1000 + 1;  /* log(58) / log(256) ≈ 0.733 */
    uint8_t *buf = calloc(size, 1);
    if (!buf) return -1;

    /* Decode each character */
    for (size_t i = leading_ones; i < str_len; i++) {
        unsigned char c = (unsigned char)str[i];
        if (c >= 128 || BASE58_MAP[c] < 0) {
            free(buf);
            return -1;  /* Invalid character */
        }
        int carry = BASE58_MAP[c];
        for (size_t j = 0; j < size; j++) {
            carry += 58 * buf[size - 1 - j];
            buf[size - 1 - j] = carry & 0xff;
            carry >>= 8;
        }
    }

    /* Skip leading zeros in buffer */
    size_t buf_start = 0;
    while (buf_start < size && buf[buf_start] == 0) {
        buf_start++;
    }

    /* Calculate result length */
    size_t result_len = leading_ones + (size - buf_start);
    if (result_len > out_len) {
        free(buf);
        return -1;  /* Output buffer too small */
    }

    /* Copy result */
    memset(out, 0, leading_ones);
    memcpy(out + leading_ones, buf + buf_start, size - buf_start);

    free(buf);
    return (int)result_len;
}

int base58_validate(const char *str) {
    if (!str || *str == '\0') {
        return 0;
    }

    while (*str) {
        unsigned char c = (unsigned char)*str;
        if (c >= 128 || BASE58_MAP[c] < 0) {
            return 0;
        }
        str++;
    }

    return 1;
}
