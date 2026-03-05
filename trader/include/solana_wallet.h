/**
 * @file solana_wallet.h
 * @brief Solana wallet management with ed25519 signing
 *
 * Loads wallet from base58-encoded private key and provides signing.
 * Uses libsodium for cryptographic operations.
 *
 * Dependencies: libsodium, base58.h
 *
 * @date 2025-12-24
 */

#ifndef SOLANA_WALLET_H
#define SOLANA_WALLET_H

#include <stdbool.h>
#include <stdint.h>

#include "base58.h"

/* Signature size for ed25519 */
#define SOLANA_SIGNATURE_SIZE   64

/**
 * @brief Solana wallet keypair
 */
typedef struct {
    uint8_t secret_key[SOLANA_PRIVKEY_SIZE];    /* 64-byte secret key (seed + pubkey) */
    uint8_t public_key[SOLANA_PUBKEY_SIZE];     /* 32-byte public key */
    char public_key_b58[BASE58_PUBKEY_LEN + 1]; /* Base58-encoded public key */
    bool loaded;                                 /* Wallet loaded successfully */
} solana_wallet_t;

/**
 * @brief Initialize and load wallet from base58 private key
 *
 * Accepts either:
 * - 64-byte keypair base58 (~88 chars) - full keypair
 * - 32-byte seed base58 (~44 chars) - derives public key
 *
 * @param wallet Wallet to initialize
 * @param base58_key Base58-encoded private key
 *
 * @return 0 on success, -1 on error
 */
int wallet_load(solana_wallet_t *wallet, const char *base58_key);

/**
 * @brief Sign a message with the wallet's private key
 *
 * @param wallet Initialized wallet
 * @param message Message bytes to sign
 * @param message_len Length of message
 * @param signature Output signature buffer (64 bytes)
 *
 * @return 0 on success, -1 on error
 */
int wallet_sign(const solana_wallet_t *wallet, const uint8_t *message,
                size_t message_len, uint8_t *signature);

/**
 * @brief Verify a signature
 *
 * @param public_key 32-byte public key
 * @param message Message that was signed
 * @param message_len Length of message
 * @param signature 64-byte signature
 *
 * @return 1 if valid, 0 if invalid
 */
int wallet_verify(const uint8_t *public_key, const uint8_t *message,
                  size_t message_len, const uint8_t *signature);

/**
 * @brief Get wallet's public key as base58 string
 *
 * @param wallet Initialized wallet
 *
 * @return Base58-encoded public key (do not free)
 */
const char *wallet_get_pubkey_b58(const solana_wallet_t *wallet);

/**
 * @brief Securely clear wallet memory
 *
 * @param wallet Wallet to clear
 */
void wallet_cleanup(solana_wallet_t *wallet);

/**
 * @brief Check if libsodium is available
 *
 * @return 1 if available, 0 if not
 */
int wallet_sodium_available(void);

#endif /* SOLANA_WALLET_H */
