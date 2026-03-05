/**
 * @file base58.h
 * @brief Base58 encoding/decoding for Solana addresses
 *
 * Implements Base58 encoding used by Solana for keypairs and addresses.
 * Uses the Bitcoin alphabet: 123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz
 *
 * @date 2025-12-24
 */

#ifndef BASE58_H
#define BASE58_H

#include <stddef.h>
#include <stdint.h>

/* Solana public key is 32 bytes, base58 encoded ~44 chars */
#define BASE58_PUBKEY_LEN       44
#define BASE58_PRIVKEY_LEN      88       /* 64 bytes -> ~88 chars */
#define SOLANA_PUBKEY_SIZE      32
#define SOLANA_PRIVKEY_SIZE     64       /* 32 private + 32 public */
#define SOLANA_SEED_SIZE        32       /* Just the seed */

/**
 * @brief Encode binary data to base58 string
 *
 * @param data Input binary data
 * @param data_len Length of input data
 * @param out Output string buffer (must be large enough)
 * @param out_len Size of output buffer
 *
 * @return Number of chars written (excluding null), or -1 on error
 */
int base58_encode(const uint8_t *data, size_t data_len, char *out, size_t out_len);

/**
 * @brief Decode base58 string to binary data
 *
 * @param str Input base58 string
 * @param out Output binary buffer
 * @param out_len Size of output buffer
 *
 * @return Number of bytes written, or -1 on error
 */
int base58_decode(const char *str, uint8_t *out, size_t out_len);

/**
 * @brief Validate a base58 string
 *
 * @param str String to validate
 *
 * @return 1 if valid, 0 if invalid
 */
int base58_validate(const char *str);

#endif /* BASE58_H */
