/**
 * @file json_parser.c
 * @brief JSON parsing implementation for GMGN messages
 *
 * Implements parsing of GMGN WebSocket JSON messages using cJSON library.
 * Handles new_pools, pair_update, and other message types.
 *
 * Dependencies: cJSON, "json_parser.h", "gmgn_types.h"
 *
 * @date 2025-12-20
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cjson/cJSON.h>

#include "json_parser.h"

/* Thread-local error message storage */
static __thread char s_last_error[256] = {0};

/**
 * @brief Set error message
 */
static void set_error(const char *msg) {
    if (msg) {
        strncpy(s_last_error, msg, sizeof(s_last_error) - 1);
        s_last_error[sizeof(s_last_error) - 1] = '\0';
    }
}

/**
 * @brief Safe string copy from cJSON
 */
static void json_get_string(const cJSON *json, const char *key, 
                            char *dest, size_t dest_size) {
    if (!json || !key || !dest || dest_size == 0) {
        return;
    }
    
    const cJSON *item = cJSON_GetObjectItemCaseSensitive(json, key);
    if (cJSON_IsString(item) && item->valuestring) {
        strncpy(dest, item->valuestring, dest_size - 1);
        dest[dest_size - 1] = '\0';
    }
}

/**
 * @brief Get integer value from cJSON
 */
static int64_t json_get_int(const cJSON *json, const char *key, int64_t default_val) {
    if (!json || !key) {
        return default_val;
    }
    
    const cJSON *item = cJSON_GetObjectItemCaseSensitive(json, key);
    if (cJSON_IsNumber(item)) {
        return (int64_t)item->valuedouble;
    }
    return default_val;
}

/**
 * @brief Get double value from cJSON
 */
static double json_get_double(const cJSON *json, const char *key, double default_val) {
    if (!json || !key) {
        return default_val;
    }
    
    const cJSON *item = cJSON_GetObjectItemCaseSensitive(json, key);
    if (cJSON_IsNumber(item)) {
        return item->valuedouble;
    }
    return default_val;
}

/**
 * @brief Parse token info (bti object)
 */
static int parse_token_info_obj(const cJSON *bti, token_info_t *token) {
    if (!bti || !token) {
        return -1;
    }
    
    memset(token, 0, sizeof(token_info_t));
    
    /* Symbol: "s" field */
    json_get_string(bti, "s", token->symbol, sizeof(token->symbol));
    
    /* Name: "n" field */
    json_get_string(bti, "n", token->name, sizeof(token->name));
    
    /* Address: "a" field */
    json_get_string(bti, "a", token->address, sizeof(token->address));
    
    /* Market cap: "mc" field (store as cents) */
    double mc = json_get_double(bti, "mc", 0.0);
    token->market_cap = (uint64_t)(mc * 100.0);
    
    /* 24h volume: "v24h" field (store as cents) */
    double v24h = json_get_double(bti, "v24h", 0.0);
    token->volume_24h = (uint64_t)(v24h * 100.0);
    
    /* Price: "p" field (store as fixed-point 1e-8) */
    double price = json_get_double(bti, "p", 0.0);
    token->price = (uint64_t)(price * GMGN_PRICE_MULTIPLIER);
    
    /* Holder count: "hc" field */
    token->holder_count = (uint32_t)json_get_int(bti, "hc", 0);
    
    /* Top 10 holders ratio: "t10hr" field (0-1 to 0-10000) */
    double t10hr = json_get_double(bti, "t10hr", 0.0);
    token->top_10_ratio = (uint16_t)(t10hr * GMGN_PERCENT_MULTIPLIER);
    
    /* Creator balance ratio: "cbr" field */
    double cbr = json_get_double(bti, "cbr", 0.0);
    token->creator_balance_ratio = (uint16_t)(cbr * GMGN_PERCENT_MULTIPLIER);
    
    /* KOL count: "kol" or "kolCount" field */
    token->kol_count = (uint8_t)json_get_int(bti, "kol", 0);
    if (token->kol_count == 0) {
        token->kol_count = (uint8_t)json_get_int(bti, "kolCount", 0);
    }
    
    /* Token age: "age" field in seconds, or calculate from creation time */
    token->age_seconds = (uint32_t)json_get_int(bti, "age", 0);
    
    /* Created at timestamp */
    token->created_at = (time_t)json_get_int(bti, "createdAt", 0);
    if (token->created_at == 0) {
        token->created_at = (time_t)json_get_int(bti, "ct", 0);
    }
    
    return 0;
}

/**
 * @brief Parse single pool object
 */
static int parse_pool_obj(const cJSON *pool_json, pool_data_t *pool) {
    if (!pool_json || !pool) {
        return -1;
    }
    
    memset(pool, 0, sizeof(pool_data_t));
    
    /* Pool address: "a" field */
    json_get_string(pool_json, "a", pool->pool_address, sizeof(pool->pool_address));
    
    /* Base token address: "ba" field */
    json_get_string(pool_json, "ba", pool->base_token_addr, sizeof(pool->base_token_addr));
    
    /* Quote token address: "qa" field */
    json_get_string(pool_json, "qa", pool->quote_token_addr, sizeof(pool->quote_token_addr));
    
    /* Exchange: "ex" field */
    json_get_string(pool_json, "ex", pool->exchange, sizeof(pool->exchange));
    
    /* Initial liquidity: "il" field (store as cents) */
    double il = json_get_double(pool_json, "il", 0.0);
    pool->initial_liquidity = (uint64_t)(il * 100.0);
    
    /* Created at timestamp: "ct" field (Unix timestamp) */
    pool->created_at = (time_t)json_get_int(pool_json, "ct", 0);
    if (pool->created_at == 0) {
        pool->created_at = (time_t)json_get_int(pool_json, "createdAt", 0);
    }
    if (pool->created_at == 0) {
        pool->created_at = time(NULL); /* fallback to current time */
    }
    
    /* Parse base token info (bti) */
    const cJSON *bti = cJSON_GetObjectItemCaseSensitive(pool_json, "bti");
    if (bti) {
        parse_token_info_obj(bti, &pool->base_token);
        
        /* Copy token address if not set */
        if (pool->base_token.address[0] == '\0' && pool->base_token_addr[0] != '\0') {
            size_t addr_len = sizeof(pool->base_token.address) - 1;
            memcpy(pool->base_token.address, pool->base_token_addr, addr_len);
            pool->base_token.address[addr_len] = '\0';
        }
        
        /* Calculate age from pool creation time if token age not set */
        if (pool->base_token.age_seconds == 0 && pool->created_at > 0) {
            time_t now = time(NULL);
            if (now > pool->created_at) {
                pool->base_token.age_seconds = (uint32_t)(now - pool->created_at);
            }
        }
        
        /* Also set token created_at if not set */
        if (pool->base_token.created_at == 0) {
            pool->base_token.created_at = pool->created_at;
        }
    }
    
    return 0;
}

gmgn_msg_type_t json_parse_message_type(const char *json_str, size_t json_len) {
    if (!json_str || json_len == 0) {
        return GMGN_MSG_UNKNOWN;
    }
    
    cJSON *json = cJSON_ParseWithLength(json_str, json_len);
    if (!json) {
        return GMGN_MSG_UNKNOWN;
    }
    
    gmgn_msg_type_t type = GMGN_MSG_UNKNOWN;
    
    /* Check for channel field */
    const cJSON *channel = cJSON_GetObjectItemCaseSensitive(json, "channel");
    if (cJSON_IsString(channel)) {
        const char *ch = channel->valuestring;
        
        if (strcmp(ch, "new_pool_info") == 0) {
            type = GMGN_MSG_NEW_POOL;
        } else if (strcmp(ch, "new_pair_update") == 0) {
            type = GMGN_MSG_PAIR_UPDATE;
        } else if (strcmp(ch, "new_launched_info") == 0) {
            type = GMGN_MSG_TOKEN_LAUNCH;
        } else if (strcmp(ch, "chain_stat") == 0) {
            type = GMGN_MSG_CHAIN_STATS;
        } else if (strcmp(ch, "wallet_trade_data") == 0) {
            type = GMGN_MSG_WALLET_TRADE;
        } else if (strcmp(ch, "ack") == 0) {
            /* Acknowledgment message, ignore */
            type = GMGN_MSG_UNKNOWN;
        }
    }
    
    /* Check for pong response */
    const cJSON *pong = cJSON_GetObjectItemCaseSensitive(json, "type");
    if (cJSON_IsString(pong) && strcmp(pong->valuestring, "pong") == 0) {
        type = GMGN_MSG_PONG;
    }
    
    /* Check for error */
    const cJSON *error = cJSON_GetObjectItemCaseSensitive(json, "error");
    if (error && !cJSON_IsNull(error)) {
        type = GMGN_MSG_ERROR;
    }
    
    /* Check for pools array directly (alternative format) */
    if (type == GMGN_MSG_UNKNOWN) {
        const cJSON *pools = cJSON_GetObjectItemCaseSensitive(json, "p");
        if (cJSON_IsArray(pools)) {
            type = GMGN_MSG_NEW_POOL;
        }
    }
    
    cJSON_Delete(json);
    return type;
}

int json_parse_new_pools(const char *json_str, size_t json_len,
                         pool_data_t *pools, size_t max_pools) {
    if (!json_str || json_len == 0 || !pools || max_pools == 0) {
        set_error("Invalid parameters");
        return -1;
    }
    
    cJSON *json = cJSON_ParseWithLength(json_str, json_len);
    if (!json) {
        set_error(cJSON_GetErrorPtr());
        return -1;
    }
    
    int count = 0;
    
    /* Look for pools array in GMGN format:
     * {"channel":"new_pool_info","data":[{"c":"sol","p":[...]}]}
     * data is an array of objects, each with "p" pools array
     */
    const cJSON *data = cJSON_GetObjectItemCaseSensitive(json, "data");
    const cJSON *pools_arr = NULL;
    const char *chain_str = NULL;
    
    if (cJSON_IsArray(data)) {
        /* data is array: [{"c":"sol","p":[...]}] - take first element */
        const cJSON *first_data = cJSON_GetArrayItem(data, 0);
        if (first_data) {
            pools_arr = cJSON_GetObjectItemCaseSensitive(first_data, "p");
            const cJSON *chain_obj = cJSON_GetObjectItemCaseSensitive(first_data, "c");
            if (cJSON_IsString(chain_obj)) {
                chain_str = chain_obj->valuestring;
            }
        }
    } else if (cJSON_IsObject(data)) {
        /* data is object: {"c":"sol","p":[...]} */
        pools_arr = cJSON_GetObjectItemCaseSensitive(data, "p");
        const cJSON *chain_obj = cJSON_GetObjectItemCaseSensitive(data, "c");
        if (cJSON_IsString(chain_obj)) {
            chain_str = chain_obj->valuestring;
        }
    }
    
    /* Fallback: look for "p" at top level */
    if (!pools_arr) {
        pools_arr = cJSON_GetObjectItemCaseSensitive(json, "p");
    }
    
    if (cJSON_IsArray(pools_arr)) {
        const cJSON *pool_json;
        cJSON_ArrayForEach(pool_json, pools_arr) {
            if ((size_t)count >= max_pools) {
                break;
            }
            
            if (parse_pool_obj(pool_json, &pools[count]) == 0) {
                /* Set chain from parent message */
                if (chain_str) {
                    strncpy(pools[count].chain, chain_str, 
                            sizeof(pools[count].chain) - 1);
                    pools[count].chain[sizeof(pools[count].chain) - 1] = '\0';
                }
                count++;
            }
        }
    }
    
    cJSON_Delete(json);
    return count;
}

int json_parse_pool(const char *json_str, size_t json_len, pool_data_t *pool) {
    if (!json_str || json_len == 0 || !pool) {
        set_error("Invalid parameters");
        return -1;
    }
    
    cJSON *json = cJSON_ParseWithLength(json_str, json_len);
    if (!json) {
        set_error(cJSON_GetErrorPtr());
        return -1;
    }
    
    int result = parse_pool_obj(json, pool);
    
    cJSON_Delete(json);
    return result;
}

int json_parse_token_info(const char *json_str, size_t json_len, 
                          token_info_t *token) {
    if (!json_str || json_len == 0 || !token) {
        set_error("Invalid parameters");
        return -1;
    }
    
    cJSON *json = cJSON_ParseWithLength(json_str, json_len);
    if (!json) {
        set_error(cJSON_GetErrorPtr());
        return -1;
    }
    
    int result = parse_token_info_obj(json, token);
    
    cJSON_Delete(json);
    return result;
}

int json_create_subscribe_msg(const char *channel, const char *chain,
                              char *buffer, size_t buffer_size) {
    if (!channel || !buffer || buffer_size == 0) {
        return -1;
    }
    
    cJSON *msg = cJSON_CreateObject();
    if (!msg) {
        return -1;
    }
    
    /* Required subscription fields per GMGN API */
    cJSON_AddStringToObject(msg, "action", "subscribe");
    cJSON_AddStringToObject(msg, "channel", channel);
    cJSON_AddStringToObject(msg, "f", "w");
    
    /* Generate a simple unique ID */
    static int msg_counter = 0;
    char id_buf[32];
    snprintf(id_buf, sizeof(id_buf), "gmgn_%08x", ++msg_counter);
    cJSON_AddStringToObject(msg, "id", id_buf);
    
    /* Create data array with chain object: [{"chain": "sol"}] */
    if (chain) {
        cJSON *data_array = cJSON_CreateArray();
        if (data_array) {
            cJSON *chain_obj = cJSON_CreateObject();
            if (chain_obj) {
                cJSON_AddStringToObject(chain_obj, "chain", chain);
                cJSON_AddItemToArray(data_array, chain_obj);
            }
            cJSON_AddItemToObject(msg, "data", data_array);
        }
    }
    
    char *json_str = cJSON_PrintUnformatted(msg);
    cJSON_Delete(msg);
    
    if (!json_str) {
        return -1;
    }
    
    size_t len = strlen(json_str);
    if (len >= buffer_size) {
        free(json_str);
        return -1;
    }
    
    strcpy(buffer, json_str);
    free(json_str);
    
    return (int)len;
}

int json_create_unsubscribe_msg(const char *channel, char *buffer, 
                                size_t buffer_size) {
    if (!channel || !buffer || buffer_size == 0) {
        return -1;
    }
    
    cJSON *msg = cJSON_CreateObject();
    if (!msg) {
        return -1;
    }
    
    cJSON_AddStringToObject(msg, "action", "unsubscribe");
    cJSON_AddStringToObject(msg, "channel", channel);
    
    char *json_str = cJSON_PrintUnformatted(msg);
    cJSON_Delete(msg);
    
    if (!json_str) {
        return -1;
    }
    
    size_t len = strlen(json_str);
    if (len >= buffer_size) {
        free(json_str);
        return -1;
    }
    
    strcpy(buffer, json_str);
    free(json_str);
    
    return (int)len;
}

int json_create_ping_msg(char *buffer, size_t buffer_size) {
    if (!buffer || buffer_size == 0) {
        return -1;
    }
    
    cJSON *msg = cJSON_CreateObject();
    if (!msg) {
        return -1;
    }
    
    cJSON_AddStringToObject(msg, "type", "ping");
    
    char *json_str = cJSON_PrintUnformatted(msg);
    cJSON_Delete(msg);
    
    if (!json_str) {
        return -1;
    }
    
    size_t len = strlen(json_str);
    if (len >= buffer_size) {
        free(json_str);
        return -1;
    }
    
    strcpy(buffer, json_str);
    free(json_str);
    
    return (int)len;
}

bool json_validate(const char *json_str, size_t json_len) {
    if (!json_str || json_len == 0) {
        return false;
    }
    
    cJSON *json = cJSON_ParseWithLength(json_str, json_len);
    if (json) {
        cJSON_Delete(json);
        return true;
    }
    
    return false;
}

const char *json_get_last_error(void) {
    return s_last_error[0] ? s_last_error : NULL;
}
