/**
 * @file solana_tx.h
 * @brief Solana transaction building and signing
 *
 * Builds Solana transactions for token swaps using Jito bundles.
 * Uses RPC to get recent blockhash.
 *
 * Dependencies: solana_wallet.h, base58.h, libcurl
 *
 * @date 2025-12-24
 */

#ifndef SOLANA_TX_H
#define SOLANA_TX_H

#include <stdint.h>
#include <stdbool.h>

#include "solana_wallet.h"

/* Transaction limits */
#define SOLANA_MAX_TX_SIZE          1232
#define SOLANA_MAX_INSTRUCTIONS     16
#define SOLANA_MAX_ACCOUNTS         32
#define SOLANA_BLOCKHASH_SIZE       32
#define SOLANA_SIG_SIZE             64

/* Program IDs (Base58) */
#define SOLANA_SYSTEM_PROGRAM       "11111111111111111111111111111111"
#define SOLANA_TOKEN_PROGRAM        "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
#define SOLANA_ASSOCIATED_TOKEN     "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL"

/* Raydium AMM (for swaps) */
#define RAYDIUM_AMM_PROGRAM         "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"

/**
 * @brief Account metadata for instruction
 */
typedef struct {
    uint8_t pubkey[32];         /* Account public key */
    bool is_signer;             /* Requires signature */
    bool is_writable;           /* Will be modified */
} solana_account_meta_t;

/**
 * @brief Instruction data
 */
typedef struct {
    uint8_t program_id[32];                     /* Program to call */
    solana_account_meta_t accounts[SOLANA_MAX_ACCOUNTS];
    int account_count;
    uint8_t data[256];                          /* Instruction data */
    int data_len;
} solana_instruction_t;

/**
 * @brief Complete transaction
 */
typedef struct {
    uint8_t recent_blockhash[SOLANA_BLOCKHASH_SIZE];
    solana_instruction_t instructions[SOLANA_MAX_INSTRUCTIONS];
    int instruction_count;
    
    /* Output */
    uint8_t signature[SOLANA_SIG_SIZE];         /* Signature after signing */
    uint8_t serialized[SOLANA_MAX_TX_SIZE];     /* Serialized transaction */
    int serialized_len;
    char base64_tx[2048];                       /* Base64-encoded for Jito */
    bool signed_flag;
} solana_tx_t;

/**
 * @brief RPC client state
 */
typedef struct {
    char endpoint[256];         /* RPC URL */
    char api_key[128];          /* Optional API key */
} solana_rpc_t;

/**
 * @brief Initialize RPC client
 *
 * @param rpc RPC client
 * @param endpoint RPC URL (e.g., "https://api.mainnet-beta.solana.com")
 * @param api_key Optional API key
 *
 * @return 0 on success
 */
int solana_rpc_init(solana_rpc_t *rpc, const char *endpoint, const char *api_key);

/**
 * @brief Get recent blockhash
 *
 * @param rpc RPC client
 * @param blockhash Output: 32-byte blockhash
 *
 * @return 0 on success, -1 on error
 */
int solana_get_recent_blockhash(solana_rpc_t *rpc, uint8_t *blockhash);

/**
 * @brief Initialize transaction
 *
 * @param tx Transaction to initialize
 * @param blockhash 32-byte recent blockhash
 *
 * @return 0 on success
 */
int solana_tx_init(solana_tx_t *tx, const uint8_t *blockhash);

/**
 * @brief Add instruction to transaction
 *
 * @param tx Transaction
 * @param instruction Instruction to add
 *
 * @return 0 on success, -1 if at limit
 */
int solana_tx_add_instruction(solana_tx_t *tx, const solana_instruction_t *instruction);

/**
 * @brief Sign transaction
 *
 * @param tx Transaction to sign
 * @param wallet Wallet with private key
 *
 * @return 0 on success, -1 on error
 */
int solana_tx_sign(solana_tx_t *tx, const solana_wallet_t *wallet);

/**
 * @brief Get Base64-encoded transaction for Jito
 *
 * @param tx Signed transaction
 *
 * @return Base64 string (do not free)
 */
const char *solana_tx_get_base64(const solana_tx_t *tx);

/**
 * @brief Build SOL transfer instruction
 *
 * @param instruction Output instruction
 * @param from Sender public key (32 bytes)
 * @param to Recipient public key (32 bytes)
 * @param lamports Amount in lamports
 *
 * @return 0 on success
 */
int solana_build_sol_transfer(solana_instruction_t *instruction,
                               const uint8_t *from, const uint8_t *to,
                               uint64_t lamports);

#endif /* SOLANA_TX_H */
