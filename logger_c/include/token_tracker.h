/**
 * @file token_tracker.h
 * @brief Token tracking for periodic re-checking
 *
 * Tracks newly discovered tokens and periodically re-checks them
 * against filters until they either pass or exceed max age.
 *
 * @date 2025-12-20
 */

#ifndef TOKEN_TRACKER_H
#define TOKEN_TRACKER_H

#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include <pthread.h>

#include "gmgn_types.h"
#include "filter.h"

/* Configuration */
#define TRACKER_MAX_TOKENS      256     /* Maximum tracked tokens */
#define TRACKER_CHECK_INTERVAL  5       /* Seconds between API checks */
#define TRACKER_API_TIMEOUT     10      /* API request timeout seconds */

/**
 * @brief Tracked token state
 */
typedef enum {
    TOKEN_STATE_TRACKING = 0,   /* Being tracked, not yet passed */
    TOKEN_STATE_PASSED,         /* Passed filters */
    TOKEN_STATE_EXPIRED,        /* Exceeded max age */
    TOKEN_STATE_REMOVED         /* Slot available */
} token_state_t;

/**
 * @brief Tracked token entry
 */
typedef struct {
    char address[GMGN_MAX_ADDRESS_LEN];     /* Token address */
    char symbol[GMGN_MAX_SYMBOL_LEN];       /* Token symbol */
    char exchange[GMGN_MAX_EXCHANGE_LEN];   /* Exchange/launchpad */
    time_t discovered_at;                    /* When we first saw it */
    time_t last_check;                       /* Last API check time */
    uint64_t last_market_cap;                /* Last known MC (cents) */
    uint8_t last_kol_count;                  /* Last known KOL count */
    uint32_t check_count;                    /* Number of checks performed */
    token_state_t state;                     /* Current state */
} tracked_token_t;

/**
 * @brief Token tracker instance
 */
typedef struct {
    tracked_token_t tokens[TRACKER_MAX_TOKENS];
    uint32_t active_count;                   /* Number of active tokens */
    uint32_t passed_count;                   /* Total tokens that passed */
    uint32_t expired_count;                  /* Total tokens that expired */
    filter_config_t *filter;                 /* Filter configuration */
    pthread_mutex_t lock;                    /* Thread safety */
    bool running;                            /* Tracker thread running */
    pthread_t thread;                        /* Background thread */
    
    /* Callback for when token passes filter */
    void (*on_token_passed)(const tracked_token_t *token, 
                            const token_info_t *info, void *user_data);
    void *callback_user_data;
} token_tracker_t;

/**
 * @brief Initialize token tracker
 *
 * @param tracker Tracker instance to initialize
 * @param filter Filter configuration to use
 * @return 0 on success, -1 on error
 */
int tracker_init(token_tracker_t *tracker, filter_config_t *filter);

/**
 * @brief Start background tracking thread
 *
 * @param tracker Tracker instance
 * @return 0 on success, -1 on error
 */
int tracker_start(token_tracker_t *tracker);

/**
 * @brief Stop tracking thread
 *
 * @param tracker Tracker instance
 */
void tracker_stop(token_tracker_t *tracker);

/**
 * @brief Cleanup tracker resources
 *
 * @param tracker Tracker instance
 */
void tracker_cleanup(token_tracker_t *tracker);

/**
 * @brief Add token to tracking
 *
 * @param tracker Tracker instance
 * @param pool Pool data containing token info
 * @return 0 if added, 1 if already tracking, -1 on error/full
 */
int tracker_add_token(token_tracker_t *tracker, const pool_data_t *pool);

/**
 * @brief Set callback for when token passes filters
 *
 * @param tracker Tracker instance
 * @param callback Callback function
 * @param user_data User data passed to callback
 */
void tracker_set_callback(token_tracker_t *tracker,
                          void (*callback)(const tracked_token_t *, 
                                          const token_info_t *, void *),
                          void *user_data);

/**
 * @brief Get tracker statistics
 *
 * @param tracker Tracker instance
 * @param active Output: active tokens being tracked
 * @param passed Output: total tokens that passed
 * @param expired Output: total tokens that expired
 */
void tracker_get_stats(const token_tracker_t *tracker,
                       uint32_t *active, uint32_t *passed, uint32_t *expired);

/**
 * @brief Manually check a single token (for testing)
 *
 * @param tracker Tracker instance
 * @param address Token address to check
 * @return true if token passed filters
 */
bool tracker_check_token(token_tracker_t *tracker, const char *address);

#endif /* TOKEN_TRACKER_H */
