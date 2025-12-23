"""
Configuration management for trading model.

This module defines the default configuration dictionary for the AI trading
model, including model architecture parameters, training hyperparameters,
data processing settings, and trading-specific constants.

Dependencies:
    None (pure Python configuration)

Author: Trading Team
Date: 2025-12-21
"""

from typing import Dict, Any


# Trading constants (Jito-optimized)
FIXED_POSITION_SIZE: float = 0.01
DELAY_SECONDS: int = 1
JITO_TIP_AVG: float = 0.00005
GAS_FEE_FIXED: float = 0.0002
PRIORITY_FEE: float = 0.0001
TOTAL_FEE_PER_TX: float = 0.00035

# Data/labeling constants
MIN_HISTORY_LENGTH: int = 12  # allow acting quickly after launch
LOOKAHEAD_SECONDS: int = 20
TAKE_PROFIT_PCT: float = 0.05   # net profit required to buy (lowered from 0.08)
STOP_LOSS_PCT: float = -0.06    # drawdown that triggers sell/avoid (more tolerant)
TRAIL_BACKOFF_PCT: float = 0.03 # take profit if price rolls over after rally

# Default configuration dictionary
DEFAULT_CONFIG: Dict[str, Any] = {
    'model': {
        'type': 'lstm',
        'input_size': 14,  # Removed in_position flag
        'hidden_size': 128,
        'num_layers': 2,
        'num_classes': 3,
        'dropout': 0.3,
    },
    'training': {
        'batch_size': 64,
        'learning_rate': 0.0005,
        'epochs': 100,
        'early_stopping_patience': 15,
        'gradient_clip_value': 1.0,
        'weight_decay': 1e-5,
        'use_mixed_precision': True,
        'accumulation_steps': 4,
        'use_focal_loss': True,
        'focal_gamma': 2.0,
        'label_smoothing': 0.0,
        'use_weighted_sampler': True,
    },
    'data': {
        'min_history_length': MIN_HISTORY_LENGTH,
        'train_split': 0.8,
        'val_split': 0.1,
        'test_split': 0.1,
        'lookback_seconds': 20,
        'random_seed': 42,
    },
    'trading': {
        'fixed_position_size': FIXED_POSITION_SIZE,
        'delay_seconds': DELAY_SECONDS,
        'total_fee_per_tx': TOTAL_FEE_PER_TX,
        'take_profit_pct': TAKE_PROFIT_PCT,
        'stop_loss_pct': STOP_LOSS_PCT,
        'trail_backoff_pct': TRAIL_BACKOFF_PCT,
        'confidence_threshold': 0.7,
    },
    'dataloader': {
        'num_workers': 4,
        'pin_memory': True,
        'persistent_workers': True,
        'prefetch_factor': 2,
    },
}


def get_config() -> Dict[str, Any]:
    """
    Get the default configuration dictionary.

    Returns:
        Deep copy of the default configuration to prevent mutations.

    Example:
        >>> config = get_config()
        >>> print(config['model']['hidden_size'])
        128
    """
    import copy
    return copy.deepcopy(DEFAULT_CONFIG)
