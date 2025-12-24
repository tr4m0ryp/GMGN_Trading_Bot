/**
 * @file solana_tx.c
 * @brief Solana transaction building implementation
 *
 * Implements transaction serialization and RPC calls.
 * Uses Solana's binary format for compact wire representation.
 *
 * @date 2025-12-24
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curl/curl.h>
#include <cjson/cJSON.h>

#include "solana_tx.h"
#include "base58.h"

/* Base64 encoding table */
static const char BASE64_CHARS[] = 
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/**
 * @brief Base64 encode binary data
 */
static int base64_encode(const uint8_t *data, size_t len, char *out, size_t out_size) {
    size_t out_len = 4 * ((len + 2) / 3);
    if (out_len >= out_size) return -1;

    size_t i, j;
    for (i = 0, j = 0; i < len; i += 3, j += 4) {
        uint32_t n = data[i] << 16;
        if (i + 1 < len) n |= data[i + 1] << 8;
        if (i + 2 < len) n |= data[i + 2];

        out[j]     = BASE64_CHARS[(n >> 18) & 0x3F];
        out[j + 1] = BASE64_CHARS[(n >> 12) & 0x3F];
        out[j + 2] = (i + 1 < len) ? BASE64_CHARS[(n >> 6) & 0x3F] : '=';
        out[j + 3] = (i + 2 < len) ? BASE64_CHARS[n & 0x3F] : '=';
    }
    out[j] = '\0';
    return (int)j;
}

/* CURL response buffer */
typedef struct {
    char *data;
    size_t size;
} curl_buffer_t;

static size_t curl_write_cb(void *contents, size_t size, size_t nmemb, void *userp) {
    size_t realsize = size * nmemb;
    curl_buffer_t *buf = (curl_buffer_t *)userp;

    char *ptr = realloc(buf->data, buf->size + realsize + 1);
    if (!ptr) return 0;

    buf->data = ptr;
    memcpy(&buf->data[buf->size], contents, realsize);
    buf->size += realsize;
    buf->data[buf->size] = '\0';

    return realsize;
}

int solana_rpc_init(solana_rpc_t *rpc, const char *endpoint, const char *api_key) {
    if (!rpc || !endpoint) return -1;

    memset(rpc, 0, sizeof(solana_rpc_t));
    strncpy(rpc->endpoint, endpoint, sizeof(rpc->endpoint) - 1);
    if (api_key) {
        strncpy(rpc->api_key, api_key, sizeof(rpc->api_key) - 1);
    }
    return 0;
}

int solana_get_recent_blockhash(solana_rpc_t *rpc, uint8_t *blockhash) {
    CURL *curl;
    CURLcode res;
    curl_buffer_t response = {0};
    struct curl_slist *headers = NULL;
    const char *request = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"getLatestBlockhash\"}";
    int ret = -1;

    if (!rpc || !blockhash) return -1;

    curl = curl_easy_init();
    if (!curl) return -1;

    response.data = malloc(1);
    response.size = 0;

    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, rpc->endpoint);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);

    res = curl_easy_perform(curl);

    if (res == CURLE_OK) {
        cJSON *root = cJSON_Parse(response.data);
        if (root) {
            cJSON *result = cJSON_GetObjectItemCaseSensitive(root, "result");
            if (result) {
                cJSON *value = cJSON_GetObjectItemCaseSensitive(result, "value");
                if (value) {
                    cJSON *hash = cJSON_GetObjectItemCaseSensitive(value, "blockhash");
                    if (hash && cJSON_IsString(hash)) {
                        if (base58_decode(hash->valuestring, blockhash, 32) == 32) {
                            ret = 0;
                        }
                    }
                }
            }
            cJSON_Delete(root);
        }
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    free(response.data);

    return ret;
}

int solana_tx_init(solana_tx_t *tx, const uint8_t *blockhash) {
    if (!tx) return -1;

    memset(tx, 0, sizeof(solana_tx_t));
    if (blockhash) {
        memcpy(tx->recent_blockhash, blockhash, SOLANA_BLOCKHASH_SIZE);
    }
    return 0;
}

int solana_tx_add_instruction(solana_tx_t *tx, const solana_instruction_t *instruction) {
    if (!tx || !instruction) return -1;
    if (tx->instruction_count >= SOLANA_MAX_INSTRUCTIONS) return -1;

    memcpy(&tx->instructions[tx->instruction_count], instruction, sizeof(solana_instruction_t));
    tx->instruction_count++;
    return 0;
}

/**
 * @brief Compact-u16 encoding for Solana serialization
 */
static int encode_compact_u16(uint8_t *buf, uint16_t value) {
    if (value < 0x80) {
        buf[0] = (uint8_t)value;
        return 1;
    } else if (value < 0x4000) {
        buf[0] = (uint8_t)((value & 0x7F) | 0x80);
        buf[1] = (uint8_t)(value >> 7);
        return 2;
    } else {
        buf[0] = (uint8_t)((value & 0x7F) | 0x80);
        buf[1] = (uint8_t)(((value >> 7) & 0x7F) | 0x80);
        buf[2] = (uint8_t)(value >> 14);
        return 3;
    }
}

int solana_tx_sign(solana_tx_t *tx, const solana_wallet_t *wallet) {
    uint8_t message[SOLANA_MAX_TX_SIZE];
    int msg_len = 0;
    uint8_t num_signers = 1;
    uint8_t num_readonly_signed = 0;
    uint8_t num_readonly_unsigned = 0;
    
    if (!tx || !wallet || !wallet->loaded) return -1;

    /* Build message header */
    message[msg_len++] = num_signers;
    message[msg_len++] = num_readonly_signed;
    message[msg_len++] = num_readonly_unsigned;

    /* Account keys (just signer for simple tx) */
    msg_len += encode_compact_u16(&message[msg_len], 1);
    memcpy(&message[msg_len], wallet->public_key, 32);
    msg_len += 32;

    /* Recent blockhash */
    memcpy(&message[msg_len], tx->recent_blockhash, 32);
    msg_len += 32;

    /* Instructions */
    msg_len += encode_compact_u16(&message[msg_len], tx->instruction_count);
    for (int i = 0; i < tx->instruction_count; i++) {
        solana_instruction_t *ix = &tx->instructions[i];
        
        /* Program ID index (0 for now) */
        message[msg_len++] = 0;
        
        /* Account indices */
        msg_len += encode_compact_u16(&message[msg_len], ix->account_count);
        for (int j = 0; j < ix->account_count; j++) {
            message[msg_len++] = (uint8_t)j;
        }
        
        /* Data */
        msg_len += encode_compact_u16(&message[msg_len], ix->data_len);
        memcpy(&message[msg_len], ix->data, ix->data_len);
        msg_len += ix->data_len;
    }

    /* Sign the message */
    if (wallet_sign(wallet, message, msg_len, tx->signature) != 0) {
        return -1;
    }

    /* Build full serialized transaction: signature + message */
    int tx_len = 0;
    tx->serialized[tx_len++] = 1;  /* Number of signatures */
    memcpy(&tx->serialized[tx_len], tx->signature, SOLANA_SIG_SIZE);
    tx_len += SOLANA_SIG_SIZE;
    memcpy(&tx->serialized[tx_len], message, msg_len);
    tx_len += msg_len;

    tx->serialized_len = tx_len;

    /* Base64 encode for Jito */
    base64_encode(tx->serialized, tx_len, tx->base64_tx, sizeof(tx->base64_tx));

    tx->signed_flag = true;
    return 0;
}

const char *solana_tx_get_base64(const solana_tx_t *tx) {
    if (!tx || !tx->signed_flag) return NULL;
    return tx->base64_tx;
}

int solana_build_sol_transfer(solana_instruction_t *instruction,
                               const uint8_t *from, const uint8_t *to,
                               uint64_t lamports) {
    if (!instruction || !from || !to) return -1;

    memset(instruction, 0, sizeof(solana_instruction_t));

    /* System program ID */
    base58_decode(SOLANA_SYSTEM_PROGRAM, instruction->program_id, 32);

    /* Accounts: from (signer, writable), to (writable) */
    memcpy(instruction->accounts[0].pubkey, from, 32);
    instruction->accounts[0].is_signer = true;
    instruction->accounts[0].is_writable = true;

    memcpy(instruction->accounts[1].pubkey, to, 32);
    instruction->accounts[1].is_signer = false;
    instruction->accounts[1].is_writable = true;

    instruction->account_count = 2;

    /* Transfer instruction: 4-byte index + 8-byte lamports */
    instruction->data[0] = 2;  /* Transfer instruction index */
    instruction->data[1] = 0;
    instruction->data[2] = 0;
    instruction->data[3] = 0;
    memcpy(&instruction->data[4], &lamports, 8);
    instruction->data_len = 12;

    return 0;
}
