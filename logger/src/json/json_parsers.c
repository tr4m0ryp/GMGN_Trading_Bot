/**
 * @file json_parsers.c
 * @brief JSON parsing for pool, token, and message type data
 *
 * Implements parsing of GMGN WebSocket messages including message
 * type detection, pool array extraction, and token info parsing.
 *
 * Dependencies: cJSON, "json_parser_internal.h"
 *
 * @date 2025-12-20
 */

#include "json_parser_internal.h"

int json_parse_token_info_obj(const cJSON *bti, token_info_t *token) {
    if (!bti || !token) {
        return -1;
    }

    memset(token, 0, sizeof(token_info_t));

    json_safe_get_string(bti, "s", token->symbol, sizeof(token->symbol));
    json_safe_get_string(bti, "n", token->name, sizeof(token->name));
    json_safe_get_string(bti, "a", token->address, sizeof(token->address));

    double mc = json_safe_get_double(bti, "mc", 0.0);
    token->market_cap = (uint64_t)(mc * 100.0);

    double v24h = json_safe_get_double(bti, "v24h", 0.0);
    token->volume_24h = (uint64_t)(v24h * 100.0);

    double price = json_safe_get_double(bti, "p", 0.0);
    token->price = (uint64_t)(price * GMGN_PRICE_MULTIPLIER);

    token->holder_count = (uint32_t)json_safe_get_int(bti, "hc", 0);

    double t10hr = json_safe_get_double(bti, "t10hr", 0.0);
    token->top_10_ratio = (uint16_t)(t10hr * GMGN_PERCENT_MULTIPLIER);

    double cbr = json_safe_get_double(bti, "cbr", 0.0);
    token->creator_balance_ratio = (uint16_t)(cbr * GMGN_PERCENT_MULTIPLIER);

    token->kol_count = (uint8_t)json_safe_get_int(bti, "kol", 0);
    if (token->kol_count == 0) {
        token->kol_count = (uint8_t)json_safe_get_int(bti, "kolCount", 0);
    }

    token->age_seconds = (uint32_t)json_safe_get_int(bti, "age", 0);

    token->created_at = (time_t)json_safe_get_int(bti, "createdAt", 0);
    if (token->created_at == 0) {
        token->created_at = (time_t)json_safe_get_int(bti, "ct", 0);
    }

    return 0;
}

int json_parse_pool_obj(const cJSON *pool_json, pool_data_t *pool) {
    if (!pool_json || !pool) {
        return -1;
    }

    memset(pool, 0, sizeof(pool_data_t));

    json_safe_get_string(pool_json, "a", pool->pool_address, sizeof(pool->pool_address));
    json_safe_get_string(pool_json, "ba", pool->base_token_addr, sizeof(pool->base_token_addr));
    json_safe_get_string(pool_json, "qa", pool->quote_token_addr, sizeof(pool->quote_token_addr));
    json_safe_get_string(pool_json, "ex", pool->exchange, sizeof(pool->exchange));

    double il = json_safe_get_double(pool_json, "il", 0.0);
    pool->initial_liquidity = (uint64_t)(il * 100.0);

    pool->created_at = (time_t)json_safe_get_int(pool_json, "ct", 0);
    if (pool->created_at == 0) {
        pool->created_at = (time_t)json_safe_get_int(pool_json, "createdAt", 0);
    }
    if (pool->created_at == 0) {
        pool->created_at = time(NULL);
    }

    const cJSON *bti = cJSON_GetObjectItemCaseSensitive(pool_json, "bti");
    if (bti) {
        json_parse_token_info_obj(bti, &pool->base_token);

        if (pool->base_token.address[0] == '\0' && pool->base_token_addr[0] != '\0') {
            size_t addr_len = sizeof(pool->base_token.address) - 1;
            memcpy(pool->base_token.address, pool->base_token_addr, addr_len);
            pool->base_token.address[addr_len] = '\0';
        }

        if (pool->base_token.age_seconds == 0 && pool->created_at > 0) {
            time_t now = time(NULL);
            if (now > pool->created_at) {
                pool->base_token.age_seconds = (uint32_t)(now - pool->created_at);
            }
        }

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
        }
    }

    const cJSON *pong = cJSON_GetObjectItemCaseSensitive(json, "type");
    if (cJSON_IsString(pong) && strcmp(pong->valuestring, "pong") == 0) {
        type = GMGN_MSG_PONG;
    }

    const cJSON *error = cJSON_GetObjectItemCaseSensitive(json, "error");
    if (error && !cJSON_IsNull(error)) {
        type = GMGN_MSG_ERROR;
    }

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
        json_set_error("Invalid parameters");
        return -1;
    }

    cJSON *json = cJSON_ParseWithLength(json_str, json_len);
    if (!json) {
        json_set_error(cJSON_GetErrorPtr());
        return -1;
    }

    int count = 0;

    const cJSON *data = cJSON_GetObjectItemCaseSensitive(json, "data");
    const cJSON *pools_arr = NULL;
    const char *chain_str = NULL;

    if (cJSON_IsArray(data)) {
        const cJSON *first_data = cJSON_GetArrayItem(data, 0);
        if (first_data) {
            pools_arr = cJSON_GetObjectItemCaseSensitive(first_data, "p");
            const cJSON *chain_obj = cJSON_GetObjectItemCaseSensitive(first_data, "c");
            if (cJSON_IsString(chain_obj)) {
                chain_str = chain_obj->valuestring;
            }
        }
    } else if (cJSON_IsObject(data)) {
        pools_arr = cJSON_GetObjectItemCaseSensitive(data, "p");
        const cJSON *chain_obj = cJSON_GetObjectItemCaseSensitive(data, "c");
        if (cJSON_IsString(chain_obj)) {
            chain_str = chain_obj->valuestring;
        }
    }

    if (!pools_arr) {
        pools_arr = cJSON_GetObjectItemCaseSensitive(json, "p");
    }

    if (cJSON_IsArray(pools_arr)) {
        const cJSON *pool_json;
        cJSON_ArrayForEach(pool_json, pools_arr) {
            if ((size_t)count >= max_pools) {
                break;
            }

            if (json_parse_pool_obj(pool_json, &pools[count]) == 0) {
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
        json_set_error("Invalid parameters");
        return -1;
    }

    cJSON *json = cJSON_ParseWithLength(json_str, json_len);
    if (!json) {
        json_set_error(cJSON_GetErrorPtr());
        return -1;
    }

    int result = json_parse_pool_obj(json, pool);

    cJSON_Delete(json);
    return result;
}

int json_parse_token_info(const char *json_str, size_t json_len,
                          token_info_t *token) {
    if (!json_str || json_len == 0 || !token) {
        json_set_error("Invalid parameters");
        return -1;
    }

    cJSON *json = cJSON_ParseWithLength(json_str, json_len);
    if (!json) {
        json_set_error(cJSON_GetErrorPtr());
        return -1;
    }

    int result = json_parse_token_info_obj(json, token);

    cJSON_Delete(json);
    return result;
}
