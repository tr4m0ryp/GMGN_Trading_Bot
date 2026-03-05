"""
Model Configuration Classes for Multi-Model Trading Architecture.

Contains dataclass configurations for:
    - Model 1 (Screener): XGBoost classifier
    - Model 2 (Entry): LSTM/Hybrid entry timing
    - Model 3 (Exit): Exit point optimizer
    - Backtesting parameters

Author: Trading Team
Date: 2025-12-29
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from .v2_config import (
    SCREENER_DECISION_TIME,
    EXIT_STOP_LOSS_PCT,
    EXIT_TRAILING_STOP_PCT,
    EXIT_TIME_LIMIT_SEC,
    EXIT_PROFIT_TARGET_PCT,
    EXIT_GAIN_EXCEPTION_PCT,
)


# =============================================================================
# Model 1: Screener Configuration (XGBoost)
# =============================================================================

@dataclass
class ScreenerConfig:
    """Configuration for Model 1: Entry Worthiness Screener (XGBoost)."""

    # Architecture
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.05
    min_child_weight: int = 3
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    gamma: float = 0.1
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0

    # Class imbalance handling (AVOID is minority ~19%)
    scale_pos_weight: float = 4.0

    # Training
    early_stopping_rounds: int = 50
    eval_metric: str = "auc"
    random_state: int = 42

    # Decision
    decision_time_sec: int = SCREENER_DECISION_TIME
    confidence_threshold: float = 0.5

    # Features used (static + early dynamic)
    num_static_features: int = 7
    num_dynamic_features: int = 12

    def to_xgb_params(self) -> Dict[str, Any]:
        """Convert to XGBoost parameter dictionary."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "gamma": self.gamma,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "scale_pos_weight": self.scale_pos_weight,
            "random_state": self.random_state,
            "eval_metric": self.eval_metric,
            "use_label_encoder": False,
            "tree_method": "hist",
            "device": "cuda",
        }


# =============================================================================
# Model 2: Entry Configuration (LSTM/Hybrid)
# =============================================================================

@dataclass
class EntryConfig:
    """Configuration for Model 2: Entry Timing Optimizer (LSTM/Hybrid)."""

    # Encoder architecture
    input_dim: int = 14
    hidden_dim: int = 128
    num_layers: int = 2
    bidirectional: bool = True
    dropout: float = 0.2

    # Attention
    use_attention: bool = True
    attention_heads: int = 4

    # Output head
    head_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    num_classes: int = 3  # ENTER_NOW, WAIT, ABORT

    # Training
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    warmup_epochs: int = 3
    total_epochs: int = 50
    patience: int = 10

    # Sequence
    max_seq_len: int = 120
    min_seq_len: int = 10

    # Decision
    confidence_threshold: float = 0.7
    max_wait_time_sec: int = 180
    inference_interval_sec: int = 5


# =============================================================================
# Model 3: Exit Configuration (Ensemble + Rules)
# =============================================================================

@dataclass
class ExitConfig:
    """Configuration for Model 3: Exit Point Optimizer."""

    # ML model architecture (XGBoost for speed)
    n_estimators: int = 100
    max_depth: int = 5
    learning_rate: float = 0.1

    # Training
    batch_size: int = 256

    # Output
    num_classes: int = 3  # EXIT_NOW, HOLD, PARTIAL_EXIT
    confidence_threshold: float = 0.6

    # Hard-coded risk management rules (override ML)
    stop_loss_pct: float = EXIT_STOP_LOSS_PCT
    trailing_stop_pct: float = EXIT_TRAILING_STOP_PCT
    time_stop_sec: int = EXIT_TIME_LIMIT_SEC
    profit_target_pct: float = EXIT_PROFIT_TARGET_PCT
    time_exception_gain_pct: float = EXIT_GAIN_EXCEPTION_PCT

    # Partial exit strategy
    partial_exit_threshold_pct: float = 0.30
    partial_exit_fraction: float = 0.50


# =============================================================================
# GPU Configuration
# =============================================================================

GPU_CONFIGS: Dict[str, Dict[str, int]] = {
    "A100": {"batch": 512, "hidden": 256, "lstm_layers": 3},
    "H100": {"batch": 512, "hidden": 256, "lstm_layers": 3},
    "L4": {"batch": 384, "hidden": 192, "lstm_layers": 2},
    "T4": {"batch": 256, "hidden": 128, "lstm_layers": 2},
    "CPU": {"batch": 64, "hidden": 64, "lstm_layers": 1},
}


def get_gpu_config(gpu_name: Optional[str] = None) -> Dict[str, int]:
    """
    Get GPU-specific configuration.

    Args:
        gpu_name: GPU name string. If None, auto-detect.

    Returns:
        Dictionary with batch size and architecture settings.
    """
    if gpu_name is None:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0).upper()
            else:
                return GPU_CONFIGS["CPU"]
        except ImportError:
            return GPU_CONFIGS["CPU"]

    for key in GPU_CONFIGS:
        if key in gpu_name.upper():
            return GPU_CONFIGS[key]

    return GPU_CONFIGS["T4"]


# =============================================================================
# Backtesting Configuration
# =============================================================================

@dataclass
class BacktestConfig:
    """Configuration for backtesting and evaluation."""

    initial_capital_sol: float = 10.0
    max_position_pct: float = 0.05
    max_concurrent_trades: int = 3

    target_win_rate: float = 0.60
    target_win_loss_ratio: float = 2.0
    target_max_drawdown: float = 0.30

    slippage_pct: float = 0.01
    include_fees: bool = True


# Default configs
DEFAULT_SCREENER_CONFIG = ScreenerConfig()
DEFAULT_ENTRY_CONFIG = EntryConfig()
DEFAULT_EXIT_CONFIG = ExitConfig()
DEFAULT_BACKTEST_CONFIG = BacktestConfig()
