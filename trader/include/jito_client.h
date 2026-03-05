/**
 * @file jito_client.h
 * @brief Jito Bundle client for fast Solana transactions
 *
 * Implements Jito Bundle submission for MEV-protected, fast execution.
 * Uses JSON-RPC to submit bundles to Jito block engine.
 *
 * Endpoints:
 * - Amsterdam: https://amsterdam.mainnet.block-engine.jito.wtf
 * - Frankfurt: https://frankfurt.mainnet.block-engine.jito.wtf
 * - New York:  https://ny.mainnet.block-engine.jito.wtf
 * - Tokyo:     https://tokyo.mainnet.block-engine.jito.wtf
 *
 * Dependencies: <stdbool.h>, <stdint.h>, libcurl
 *
 * @date 2025-12-24
 */

#ifndef JITO_CLIENT_H
#define JITO_CLIENT_H

#include <stdbool.h>
#include <stdint.h>

/* Jito configuration */
#define JITO_API_PATH           "/api/v1/bundles"
#define JITO_MAX_BUNDLE_SIZE    5       /* Max transactions per bundle */
#define JITO_DEFAULT_TIP        50000   /* Default tip: 50000 lamports = 0.00005 SOL */
#define JITO_MIN_TIP            1000    /* Minimum tip: 1000 lamports */
#define JITO_MAX_TIP            500000  /* Max tip: 0.0005 SOL */

/* Number of tip accounts (for load balancing) */
#define JITO_TIP_ACCOUNT_COUNT  8

/**
 * @brief Jito tip accounts (pick one randomly to reduce contention)
 */
extern const char *JITO_TIP_ACCOUNTS[JITO_TIP_ACCOUNT_COUNT];

/**
 * @brief Bundle submission result
 */
typedef struct {
    bool success;                   /* Submission accepted */
    char bundle_id[128];            /* Bundle ID from Jito */
    int64_t submit_time_ms;         /* Time to submit */
    char error[256];                /* Error message if failed */
} jito_result_t;

/**
 * @brief Bundle status
 */
typedef enum {
    JITO_STATUS_UNKNOWN = 0,
    JITO_STATUS_PENDING,            /* Submitted, awaiting processing */
    JITO_STATUS_LANDED,             /* Successfully included in block */
    JITO_STATUS_FAILED,             /* Bundle failed */
    JITO_STATUS_DROPPED             /* Bundle dropped/expired */
} jito_bundle_status_t;

/**
 * @brief Jito client state
 */
typedef struct {
    char endpoint[256];             /* Jito API endpoint */
    uint64_t tip_lamports;          /* Tip amount in lamports */
    int current_tip_account;        /* Current tip account index */
    bool initialized;
} jito_client_t;

/**
 * @brief Initialize Jito client
 *
 * @param client Client instance
 * @param endpoint Jito block engine endpoint (e.g., "https://amsterdam.mainnet.block-engine.jito.wtf")
 * @param tip_lamports Tip amount in lamports (recommended: 50000)
 *
 * @return 0 on success, -1 on error
 */
int jito_init(jito_client_t *client, const char *endpoint, uint64_t tip_lamports);

/**
 * @brief Submit a single transaction via Jito Bundle
 *
 * Wraps the transaction in a bundle with a tip for priority.
 *
 * @param client Jito client
 * @param signed_tx Base64-encoded signed transaction
 * @param result Output: submission result
 *
 * @return 0 on success, -1 on error
 */
int jito_submit_tx(jito_client_t *client, const char *signed_tx, jito_result_t *result);

/**
 * @brief Submit multiple transactions as a bundle
 *
 * All transactions execute atomically (all or nothing).
 *
 * @param client Jito client
 * @param signed_txs Array of base64-encoded signed transactions
 * @param tx_count Number of transactions (max JITO_MAX_BUNDLE_SIZE)
 * @param result Output: submission result
 *
 * @return 0 on success, -1 on error
 */
int jito_submit_bundle(jito_client_t *client, const char **signed_txs, int tx_count,
                       jito_result_t *result);

/**
 * @brief Check bundle status
 *
 * @param client Jito client
 * @param bundle_id Bundle ID returned from submission
 * @param status Output: current bundle status
 *
 * @return 0 on success, -1 on error
 */
int jito_get_bundle_status(jito_client_t *client, const char *bundle_id,
                           jito_bundle_status_t *status);

/**
 * @brief Get a random tip account address
 *
 * Using a random tip account helps distribute load.
 *
 * @param client Jito client
 *
 * @return Tip account address (do not free)
 */
const char *jito_get_tip_account(jito_client_t *client);

/**
 * @brief Cleanup Jito client
 *
 * @param client Client instance
 */
void jito_cleanup(jito_client_t *client);

/**
 * @brief Convert bundle status to string
 *
 * @param status Status enum
 *
 * @return Human-readable string
 */
const char *jito_status_str(jito_bundle_status_t status);

#endif /* JITO_CLIENT_H */
