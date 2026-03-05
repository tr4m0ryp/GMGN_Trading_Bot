"""
Configuration constants and utilities for Multi-Model Trading Architecture.

Contains all constants, thresholds, feature lists, and IO functions.
Model-specific dataclass configurations are in model_configs.py.

Author: Trading Team
Date: 2025-12-29
"""

import json
from typing import Any
from pathlib import Path


# =============================================================================
# Data Configuration
# =============================================================================

DATASET_SIZE = 956
MEDIAN_LIFESPAN_SEC = 172
MEDIAN_TIME_TO_PEAK_SEC = 76
AVG_LIFESPAN_SEC = 220

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

EARLY_WINDOW_SEC = 30
SCREENER_DECISION_TIME = 30

MIN_CANDLES_FOR_TRAINING = 10
MIN_CANDLES_FOR_SCREENER = 5


# =============================================================================
# Label Definitions
# =============================================================================

# Model 1: Screener Labels
SCREENER_WORTHY_THRESHOLD = 2.0
SCREENER_LOOKAHEAD_SEC = 300

# Model 2: Entry Labels
ENTRY_PROFIT_THRESHOLD = 0.20
ENTRY_MAX_DRAWDOWN = 0.15
ENTRY_LOOKAHEAD_SEC = 60

# Model 3: Exit Labels
EXIT_TRAILING_STOP_PCT = 0.15
EXIT_STOP_LOSS_PCT = 0.25
EXIT_PROFIT_TARGET_PCT = 2.00
EXIT_TIME_LIMIT_SEC = 300
EXIT_GAIN_EXCEPTION_PCT = 1.00


# =============================================================================
# Trading Parameters
# =============================================================================

FIXED_POSITION_SIZE_SOL = 0.1
MAX_POSITION_PCT = 0.05

JITO_TIP_SOL = 0.02
GAS_FEE_SOL = 0.003
PRIORITY_FEE_SOL = 0.001
TOTAL_FEE_PER_TX = JITO_TIP_SOL + GAS_FEE_SOL + PRIORITY_FEE_SOL

DELAY_SECONDS = 1


# =============================================================================
# Feature Configuration
# =============================================================================

SCREENER_STATIC_FEATURES = [
    "mc_bin_sub_5k", "mc_bin_5k_10k", "mc_bin_10k_15k",
    "mc_bin_15k_20k", "mc_bin_above_20k", "kol_count",
    "token_age_at_detection",
]

SCREENER_DYNAMIC_FEATURES = [
    "return_5s", "return_10s", "return_15s", "return_30s",
    "volume_10s", "volume_20s", "volume_30s", "tx_count_30s",
    "buy_sell_ratio_30s", "largest_tx_size",
    "volume_acceleration", "price_acceleration",
]

TIMESERIES_FEATURES = [
    "log_close", "return_1s", "return_3s", "return_5s",
    "range_ratio", "volume_log", "rsi_norm", "macd_norm",
    "bb_upper_dev", "bb_lower_dev", "vwap_dev",
    "momentum_10s", "order_imbalance", "drawdown_from_high",
]


# =============================================================================
# Configuration IO
# =============================================================================

def save_config(config: Any, path: str) -> None:
    """Save configuration to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(config, "__dataclass_fields__"):
        data = {k: getattr(config, k) for k in config.__dataclass_fields__}
    else:
        data = config

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_config(config_class: type, path: str) -> Any:
    """Load configuration from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    return config_class(**data)
