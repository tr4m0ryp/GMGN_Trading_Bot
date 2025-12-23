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

# Default configuration dictionary (simple LSTM - legacy)
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

# Advanced Transformer-LSTM configuration for maximum accuracy
ADVANCED_CONFIG: Dict[str, Any] = {
    'model': {
        'type': 'transformer_lstm',
        'input_size': 14,
        'hidden_size': 512,           # 4x larger hidden dimension
        'num_lstm_layers': 3,         # Bidirectional LSTM layers
        'num_transformer_layers': 4,  # Transformer encoder layers
        'num_heads': 8,               # Multi-head attention heads
        'num_classes': 3,
        'dropout': 0.4,               # Higher dropout for regularization
        'ff_mult': 4,                 # Feed-forward multiplier
    },
    'training': {
        'batch_size': 256,            # 4x larger batch size for T4 GPU
        'learning_rate': 1e-4,        # Lower LR for larger model
        'min_lr': 1e-6,               # Minimum learning rate
        'warmup_epochs': 5,           # Warmup epochs for LR
        'epochs': 150,                # More epochs with early stopping
        'early_stopping_patience': 20,
        'gradient_clip_value': 1.0,
        'weight_decay': 0.01,         # Higher weight decay (AdamW style)
        'use_mixed_precision': True,
        'accumulation_steps': 2,      # Effective batch = 512
        'use_focal_loss': True,
        'focal_gamma': 2.5,           # Stronger focus on hard examples
        'label_smoothing': 0.1,       # Label smoothing for regularization
        'use_weighted_sampler': True,
        'use_cosine_schedule': True,  # Cosine annealing with warmup
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
        'confidence_threshold': 0.6,  # Lower threshold with better model
    },
    'dataloader': {
        'num_workers': 2,             # Reduced for Colab
        'pin_memory': True,
        'persistent_workers': False,  # Disabled for Colab stability
        'prefetch_factor': 2,
    },
}

# Lightweight transformer config (faster training, still powerful)
LIGHTWEIGHT_CONFIG: Dict[str, Any] = {
    'model': {
        'type': 'lightweight_transformer',
        'input_size': 14,
        'hidden_size': 256,
        'num_layers': 6,
        'num_heads': 8,
        'num_classes': 3,
        'dropout': 0.3,
    },
    'training': {
        'batch_size': 512,
        'learning_rate': 3e-4,
        'min_lr': 1e-6,
        'warmup_epochs': 3,
        'epochs': 100,
        'early_stopping_patience': 15,
        'gradient_clip_value': 1.0,
        'weight_decay': 0.01,
        'use_mixed_precision': True,
        'accumulation_steps': 1,
        'use_focal_loss': True,
        'focal_gamma': 2.0,
        'label_smoothing': 0.1,
        'use_weighted_sampler': True,
        'use_cosine_schedule': True,
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
        'confidence_threshold': 0.6,
    },
    'dataloader': {
        'num_workers': 2,
        'pin_memory': True,
        'persistent_workers': False,
        'prefetch_factor': 2,
    },
}


def get_config(config_type: str = 'default') -> Dict[str, Any]:
    """
    Get a configuration dictionary by type.

    Args:
        config_type: One of 'default', 'advanced', or 'lightweight'.
            - 'default': Simple LSTM model (~245K params)
            - 'advanced': Transformer-LSTM hybrid (~15M params, highest accuracy)
            - 'lightweight': Transformer-only (~3M params, balanced speed/accuracy)

    Returns:
        Deep copy of the requested configuration to prevent mutations.

    Example:
        >>> config = get_config('advanced')
        >>> print(config['model']['hidden_size'])
        512
        >>> print(config['model']['type'])
        'transformer_lstm'
    """
    import copy

    configs = {
        'default': DEFAULT_CONFIG,
        'advanced': ADVANCED_CONFIG,
        'lightweight': LIGHTWEIGHT_CONFIG,
    }

    if config_type not in configs:
        raise ValueError(f"Unknown config type: {config_type}. Choose from {list(configs.keys())}")

    return copy.deepcopy(configs[config_type])


def get_advanced_config() -> Dict[str, Any]:
    """Get the advanced Transformer-LSTM configuration for maximum accuracy."""
    return get_config('advanced')


def get_lightweight_config() -> Dict[str, Any]:
    """Get the lightweight transformer configuration for balanced performance."""
    return get_config('lightweight')
