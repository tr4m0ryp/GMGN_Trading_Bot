/**
 * @file solana_wallet.c
 * @brief Solana wallet implementation with libsodium
 *
 * Uses libsodium for:
 * - Ed25519 key pair generation from seed
 * - Message signing
 * - Signature verification
 *
 * @date 2025-12-24
 */

#include <stdio.h>
#include <string.h>
#include <sodium.h>

#include "solana_wallet.h"
#include "base58.h"

/* Track if sodium is initialized */
static int g_sodium_initialized = 0;

/**
 * @brief Initialize libsodium (thread-safe)
 */
static int init_sodium(void) {
    if (g_sodium_initialized) {
        return 0;
    }
    
    if (sodium_init() < 0) {
        return -1;
    }
    
    g_sodium_initialized = 1;
    return 0;
}

int wallet_sodium_available(void) {
    return init_sodium() == 0;
}

int wallet_load(solana_wallet_t *wallet, const char *base58_key) {
    uint8_t decoded[SOLANA_PRIVKEY_SIZE];
    int decoded_len;
    
    if (!wallet || !base58_key) {
        return -1;
    }
    
    /* Initialize sodium */
    if (init_sodium() != 0) {
        fprintf(stderr, "[WALLET] Failed to initialize libsodium\n");
        return -1;
    }
    
    /* Clear wallet */
    sodium_memzero(wallet, sizeof(solana_wallet_t));
    
    /* Validate base58 */
    if (!base58_validate(base58_key)) {
        fprintf(stderr, "[WALLET] Invalid base58 key format\n");
        return -1;
    }
    
    /* Decode base58 key */
    decoded_len = base58_decode(base58_key, decoded, sizeof(decoded));
    
    if (decoded_len == SOLANA_PRIVKEY_SIZE) {
        /* Full 64-byte keypair: first 32 = seed, second 32 = public key */
        memcpy(wallet->secret_key, decoded, SOLANA_PRIVKEY_SIZE);
        memcpy(wallet->public_key, decoded + 32, SOLANA_PUBKEY_SIZE);
    }
    else if (decoded_len == SOLANA_SEED_SIZE) {
        /* 32-byte seed: derive public key */
        uint8_t pk[crypto_sign_PUBLICKEYBYTES];
        uint8_t sk[crypto_sign_SECRETKEYBYTES];
        
        crypto_sign_seed_keypair(pk, sk, decoded);
        
        memcpy(wallet->secret_key, sk, SOLANA_PRIVKEY_SIZE);
        memcpy(wallet->public_key, pk, SOLANA_PUBKEY_SIZE);
        
        /* Clear temporary keys */
        sodium_memzero(pk, sizeof(pk));
        sodium_memzero(sk, sizeof(sk));
    }
    else {
        fprintf(stderr, "[WALLET] Invalid key length: %d (expected 32 or 64)\n", decoded_len);
        sodium_memzero(decoded, sizeof(decoded));
        return -1;
    }
    
    /* Clear decoded buffer */
    sodium_memzero(decoded, sizeof(decoded));
    
    /* Encode public key as base58 */
    if (base58_encode(wallet->public_key, SOLANA_PUBKEY_SIZE,
                      wallet->public_key_b58, sizeof(wallet->public_key_b58)) < 0) {
        fprintf(stderr, "[WALLET] Failed to encode public key\n");
        wallet_cleanup(wallet);
        return -1;
    }
    
    wallet->loaded = true;
    return 0;
}

int wallet_sign(const solana_wallet_t *wallet, const uint8_t *message,
                size_t message_len, uint8_t *signature) {
    unsigned long long sig_len;
    
    if (!wallet || !wallet->loaded || !message || !signature) {
        return -1;
    }
    
    /* Sign message with detached signature */
    if (crypto_sign_detached(signature, &sig_len,
                              message, message_len,
                              wallet->secret_key) != 0) {
        return -1;
    }
    
    return 0;
}

int wallet_verify(const uint8_t *public_key, const uint8_t *message,
                  size_t message_len, const uint8_t *signature) {
    if (!public_key || !message || !signature) {
        return 0;
    }
    
    if (init_sodium() != 0) {
        return 0;
    }
    
    return crypto_sign_verify_detached(signature, message, message_len, public_key) == 0;
}

const char *wallet_get_pubkey_b58(const solana_wallet_t *wallet) {
    if (!wallet || !wallet->loaded) {
        return NULL;
    }
    return wallet->public_key_b58;
}

void wallet_cleanup(solana_wallet_t *wallet) {
    if (wallet) {
        sodium_memzero(wallet, sizeof(solana_wallet_t));
    }
}
