/**
 * @file model_loader.c
 * @brief Implementation of model inference
 *
 * Uses Python subprocess for stable-baselines3 model inference.
 * ONNX backend can be added later for faster inference.
 *
 * Dependencies: <stdio.h>, <stdlib.h>, <string.h>, <unistd.h>
 *
 * @date 2025-12-24
 */

/* Disable truncation warnings - we handle string sizes manually */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"
#pragma GCC diagnostic ignored "-Wstringop-truncation"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <errno.h>

#include "model_loader.h"

/**
 * @brief Safe string copy
 */
static void safe_strcpy(char *dest, size_t dest_size, const char *src) {
    if (!dest || !src || dest_size == 0) return;
    size_t len = strlen(src);
    if (len >= dest_size) len = dest_size - 1;
    memcpy(dest, src, len);
    dest[len] = '\0';
}

/* JSON parsing helpers */
static int parse_json_int(const char *json, const char *key, int default_val) {
    char search[64];
    snprintf(search, sizeof(search), "\"%s\":", key);
    char *pos = strstr(json, search);
    if (!pos) return default_val;
    pos += strlen(search);
    while (*pos == ' ') pos++;
    return atoi(pos);
}

static double parse_json_float(const char *json, const char *key, double default_val) {
    char search[64];
    snprintf(search, sizeof(search), "\"%s\":", key);
    char *pos = strstr(json, search);
    if (!pos) return default_val;
    pos += strlen(search);
    while (*pos == ' ') pos++;
    return atof(pos);
}

static int parse_json_bool(const char *json, const char *key, int default_val) {
    char search[64];
    snprintf(search, sizeof(search), "\"%s\":", key);
    char *pos = strstr(json, search);
    if (!pos) return default_val;
    pos += strlen(search);
    while (*pos == ' ') pos++;
    return (strncmp(pos, "true", 4) == 0) ? 1 : 0;
}

static int parse_json_array(const char *json, const char *key, float *arr, int max_len) {
    char search[64];
    snprintf(search, sizeof(search), "\"%s\":", key);
    char *pos = strstr(json, search);
    if (!pos) return -1;
    
    pos = strchr(pos, '[');
    if (!pos) return -1;
    pos++;
    
    int count = 0;
    while (*pos && *pos != ']' && count < max_len) {
        while (*pos == ' ' || *pos == ',') pos++;
        if (*pos == ']') break;
        arr[count++] = (float)atof(pos);
        while (*pos && *pos != ',' && *pos != ']') pos++;
    }
    
    return count;
}

static void parse_json_string(const char *json, const char *key, char *out, int max_len) {
    char search[64];
    snprintf(search, sizeof(search), "\"%s\":\"", key);
    char *pos = strstr(json, search);
    if (!pos) {
        out[0] = '\0';
        return;
    }
    pos += strlen(search);
    
    int i = 0;
    while (*pos && *pos != '"' && i < max_len - 1) {
        out[i++] = *pos++;
    }
    out[i] = '\0';
}

/**
 * @brief Check if file exists
 */
static int file_exists(const char *path) {
    struct stat st;
    return (stat(path, &st) == 0);
}

int model_init(model_state_t *state, const char *model_dir) {
    char path[512];
    
    if (!state || !model_dir) {
        return -1;
    }
    
    memset(state, 0, sizeof(model_state_t));
    state->backend = MODEL_BACKEND_NONE;
    state->loaded = false;
    
    /* Check for ONNX model first (faster) */
    snprintf(path, sizeof(path), "%s/%s", model_dir, MODEL_ONNX_FILE);
    if (file_exists(path)) {
        /* TODO: Initialize ONNX Runtime session */
        /* For now, fall through to Python backend */
    }
    
    /* Check for stable-baselines3 model */
    snprintf(path, sizeof(path), "%s/%s", model_dir, MODEL_PPO_FILE);
    if (file_exists(path)) {
        safe_strcpy(state->model_path, sizeof(state->model_path), path);
        state->backend = MODEL_BACKEND_PYTHON;
        state->loaded = true;
        
        /* Find inference script */
        snprintf(state->script_path, sizeof(state->script_path),
                 "%s/../scripts/inference.py", model_dir);
        
        return 0;
    }
    
    /* No model found */
    fprintf(stderr, "[MODEL] No model found in %s (expected %s or %s)\n",
            model_dir, MODEL_ONNX_FILE, MODEL_PPO_FILE);
    return -1;
}

int model_predict(model_state_t *state, const float *observation,
                  model_result_t *result) {
    char cmd[1024];
    char obs_str[2048];
    char output[4096];
    FILE *fp;
    struct timeval start, end;
    int obs_len = 0;
    
    if (!state || !observation || !result) {
        return -1;
    }
    
    memset(result, 0, sizeof(model_result_t));
    result->action = SIGNAL_HOLD;
    result->probabilities[0] = 1.0f;
    
    if (!state->loaded) {
        snprintf(result->error, sizeof(result->error), "Model not loaded");
        result->success = false;
        return -1;
    }
    
    gettimeofday(&start, NULL);
    
    if (state->backend == MODEL_BACKEND_PYTHON) {
        /* Build observation string */
        for (int i = 0; i < MODEL_TOTAL_OBS_DIM; i++) {
            if (i > 0) {
                obs_len += snprintf(obs_str + obs_len, sizeof(obs_str) - obs_len, ",");
            }
            obs_len += snprintf(obs_str + obs_len, sizeof(obs_str) - obs_len, "%.6f", observation[i]);
        }
        
        /* Build command */
        snprintf(cmd, sizeof(cmd),
                 "echo '%s' | python3 '%s' --model '%s'",
                 obs_str, state->script_path, state->model_path);
        
        /* Execute */
        fp = popen(cmd, "r");
        if (!fp) {
            snprintf(result->error, sizeof(result->error), "Failed to execute inference: %s", strerror(errno));
            result->success = false;
            return -1;
        }
        
        /* Read output */
        if (fgets(output, sizeof(output), fp) == NULL) {
            pclose(fp);
            snprintf(result->error, sizeof(result->error), "No output from inference");
            result->success = false;
            return -1;
        }
        
        pclose(fp);
        
        /* Parse JSON result */
        result->success = parse_json_bool(output, "success", 0);
        
        if (result->success) {
            result->action = (trade_signal_t)parse_json_int(output, "action", 0);
            result->confidence = (float)parse_json_float(output, "confidence", 0.0);
            parse_json_array(output, "probabilities", result->probabilities, 3);
            result->inference_time_us = parse_json_int(output, "inference_time_us", 0);
        } else {
            parse_json_string(output, "error", result->error, sizeof(result->error));
        }
    } else {
        snprintf(result->error, sizeof(result->error), "Unsupported backend");
        result->success = false;
        return -1;
    }
    
    gettimeofday(&end, NULL);
    
    /* Update total inference time (includes subprocess overhead) */
    int64_t total_time_us = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
    if (result->inference_time_us == 0) {
        result->inference_time_us = total_time_us;
    }
    
    return result->success ? 0 : -1;
}

bool model_is_loaded(const model_state_t *state) {
    return state && state->loaded;
}

const char *model_backend_name(const model_state_t *state) {
    if (!state) return "none";
    
    switch (state->backend) {
        case MODEL_BACKEND_ONNX:   return "onnx";
        case MODEL_BACKEND_PYTHON: return "python";
        default:                   return "none";
    }
}

void model_cleanup(model_state_t *state) {
    if (!state) return;
    
    /* TODO: Cleanup ONNX session if used */
    
    memset(state, 0, sizeof(model_state_t));
}

const char *model_action_str(trade_signal_t action) {
    switch (action) {
        case SIGNAL_HOLD: return "HOLD";
        case SIGNAL_BUY:  return "BUY";
        case SIGNAL_SELL: return "SELL";
        default:          return "???";
    }
}
