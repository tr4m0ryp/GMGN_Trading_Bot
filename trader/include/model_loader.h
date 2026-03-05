/**
 * @file model_loader.h
 * @brief Model inference interface for AI trading
 *
 * Supports two inference backends:
 * 1. ONNX Runtime - Fast C-native inference
 * 2. Python subprocess - Fallback using stable-baselines3
 *
 * The model directory should contain either:
 * - model.onnx (preferred - faster)
 * - model.zip (stable-baselines3 PPO model)
 *
 * Dependencies: <stdbool.h>, <stdint.h>
 *
 * @date 2025-12-24
 */

#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include <stdbool.h>
#include <stdint.h>

/* Model configuration */
#define MODEL_DIR           "models"
#define MODEL_ONNX_FILE     "model.onnx"
#define MODEL_PPO_FILE      "model.zip"
#define MODEL_INFERENCE_SCRIPT  "scripts/inference.py"

/* Observation dimensions */
#define MODEL_NUM_FEATURES      14      /* Price features from candles */
#define MODEL_POSITION_FEATURES 5       /* Position state features */
#define MODEL_TOTAL_OBS_DIM     19      /* Total observation size */

/* Action space */
#define MODEL_ACTION_HOLD   0
#define MODEL_ACTION_BUY    1
#define MODEL_ACTION_SELL   2
#define MODEL_NUM_ACTIONS   3

/**
 * @brief Trading signal from model
 */
typedef enum {
    SIGNAL_HOLD = 0,
    SIGNAL_BUY = 1,
    SIGNAL_SELL = 2
} trade_signal_t;

/**
 * @brief Model inference result
 */
typedef struct {
    trade_signal_t action;          /* Predicted action */
    float confidence;               /* Action probability (0.0-1.0) */
    float probabilities[3];         /* [hold, buy, sell] probabilities */
    int64_t inference_time_us;      /* Inference time in microseconds */
    bool success;                   /* Inference succeeded */
    char error[256];                /* Error message if failed */
} model_result_t;

/**
 * @brief Model backend type
 */
typedef enum {
    MODEL_BACKEND_NONE = 0,
    MODEL_BACKEND_ONNX,             /* ONNX Runtime (fastest) */
    MODEL_BACKEND_PYTHON            /* Python subprocess (fallback) */
} model_backend_t;

/**
 * @brief Model state
 */
typedef struct {
    model_backend_t backend;        /* Active backend */
    void *onnx_session;             /* ONNX Runtime session (if ONNX) */
    char model_path[512];           /* Path to model file */
    char script_path[512];          /* Path to inference script */
    bool loaded;                    /* Model loaded successfully */
} model_state_t;

/**
 * @brief Initialize model loader
 *
 * Searches for model file in models/ directory in order:
 * 1. model.onnx (ONNX Runtime - fastest)
 * 2. model.zip (Python stable-baselines3 - fallback)
 *
 * @param state Model state to initialize
 * @param model_dir Directory containing model files
 *
 * @return 0 on success, -1 if no model found
 */
int model_init(model_state_t *state, const char *model_dir);

/**
 * @brief Run model inference
 *
 * Predicts action from observation vector.
 * Target: < 500ms inference time for memecoin trading.
 *
 * @param state Initialized model state
 * @param observation Observation array (size MODEL_TOTAL_OBS_DIM)
 * @param result Output: prediction result
 *
 * @return 0 on success, -1 on error
 */
int model_predict(model_state_t *state, const float *observation,
                  model_result_t *result);

/**
 * @brief Check if model is loaded
 *
 * @param state Model state
 *
 * @return true if model is ready for inference
 */
bool model_is_loaded(const model_state_t *state);

/**
 * @brief Get model backend name
 *
 * @param state Model state
 *
 * @return Backend name string ("onnx", "python", "none")
 */
const char *model_backend_name(const model_state_t *state);

/**
 * @brief Cleanup model resources
 *
 * @param state Model state
 */
void model_cleanup(model_state_t *state);

/**
 * @brief Convert action enum to string
 *
 * @param action Action value
 *
 * @return "HOLD", "BUY", or "SELL"
 */
const char *model_action_str(trade_signal_t action);

#endif /* MODEL_LOADER_H */
