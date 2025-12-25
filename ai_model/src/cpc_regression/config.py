"""
Configuration classes for CPC + Return Regression model.

Provides dataclass-based configuration for:
- CPC pretraining (Phase 1)
- Return regression (Phase 2)
- Kelly position sizing (Inference)

Includes GPU-aware auto-scaling for A100/H100/L4/T4.

Dependencies:
    dataclasses, typing

Date: 2025-12-25
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import torch


@dataclass
class CPCConfig:
    """
    Configuration for CPC pretraining (Phase 1).

    Default values are optimized for A100/H100 GPUs.
    Use get_config_for_gpu() for automatic scaling.
    """
    # Encoder architecture
    input_dim: int = 14
    hidden_dim: int = 512
    embed_dim: int = 1024
    lstm_layers: int = 3
    n_heads: int = 16
    ff_dim: int = 4096
    dropout: float = 0.1

    # Autoregressive model
    ar_hidden: int = 512

    # CPC settings
    prediction_steps: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10])
    temperature: float = 0.07

    # Training
    batch_size: int = 512
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    total_epochs: int = 50
    max_seq_len: int = 128
    min_seq_len: int = 20
    grad_clip: float = 1.0

    # Scheduler
    scheduler: str = 'cosine'  # 'cosine', 'linear', 'constant'
    min_lr: float = 1e-6


@dataclass
class RegressionConfig:
    """
    Configuration for return regression (Phase 2).

    Multi-task learning with return and drawdown prediction.
    """
    # Head architecture
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    predict_drawdown: bool = True
    dropout: float = 0.1

    # Training
    batch_size: int = 256
    learning_rate: float = 3e-5
    encoder_lr_mult: float = 0.1  # Encoder learns 10x slower
    weight_decay: float = 0.01
    warmup_epochs: int = 2
    total_epochs: int = 30
    grad_clip: float = 1.0

    # Progressive unfreezing
    freeze_encoder_epochs: int = 5
    unfreeze_layers_per_epoch: int = 1

    # Loss settings
    min_log_var: float = -6.0
    max_log_var: float = 6.0
    var_reg_weight: float = 0.1
    drawdown_weight: float = 0.3

    # CPC regularization (prevents forgetting)
    cpc_regularization: float = 0.1
    cpc_decay: float = 0.95  # Decay per epoch

    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4


@dataclass
class KellyConfig:
    """
    Configuration for Kelly position sizing.

    Quarter Kelly (0.25) is conservative and suitable
    for volatile crypto markets.
    """
    kelly_fraction: float = 0.25
    max_position: float = 0.05
    min_edge: float = 0.02
    max_variance: float = 0.01
    transaction_cost: float = 0.007
    min_sharpe: float = 0.5


# GPU-specific configurations
GPU_CONFIGS = {
    'A100': {
        'cpc': {
            'batch_size': 512,
            'hidden_dim': 512,
            'embed_dim': 1024,
            'lstm_layers': 3,
            'n_heads': 16,
            'ff_dim': 4096,
            'ar_hidden': 512,
        },
        'regression': {
            'batch_size': 256,
        },
    },
    'H100': {
        'cpc': {
            'batch_size': 512,
            'hidden_dim': 512,
            'embed_dim': 1024,
            'lstm_layers': 3,
            'n_heads': 16,
            'ff_dim': 4096,
            'ar_hidden': 512,
        },
        'regression': {
            'batch_size': 256,
        },
    },
    'L4': {
        'cpc': {
            'batch_size': 384,
            'hidden_dim': 384,
            'embed_dim': 768,
            'lstm_layers': 2,
            'n_heads': 12,
            'ff_dim': 3072,
            'ar_hidden': 384,
        },
        'regression': {
            'batch_size': 192,
        },
    },
    'T4': {
        'cpc': {
            'batch_size': 256,
            'hidden_dim': 256,
            'embed_dim': 512,
            'lstm_layers': 2,
            'n_heads': 8,
            'ff_dim': 2048,
            'ar_hidden': 256,
        },
        'regression': {
            'batch_size': 128,
        },
    },
}


def detect_gpu() -> str:
    """
    Detect GPU type from CUDA device name.

    Returns:
        GPU type string: 'A100', 'H100', 'L4', 'T4', or 'unknown'
    """
    if not torch.cuda.is_available():
        return 'cpu'

    device_name = torch.cuda.get_device_name(0).upper()

    if 'A100' in device_name:
        return 'A100'
    elif 'H100' in device_name:
        return 'H100'
    elif 'L4' in device_name:
        return 'L4'
    elif 'T4' in device_name:
        return 'T4'
    elif 'V100' in device_name:
        return 'T4'  # Similar memory to T4
    elif 'P100' in device_name:
        return 'T4'  # Similar memory to T4
    else:
        # Default to T4 config for unknown GPUs
        return 'T4'


def get_config_for_gpu(
    gpu_type: Optional[str] = None,
) -> Dict[str, any]:
    """
    Get configuration optimized for detected/specified GPU.

    Args:
        gpu_type: GPU type string. If None, auto-detect.

    Returns:
        Dictionary with 'cpc', 'regression', and 'kelly' configs

    Example:
        >>> configs = get_config_for_gpu()
        >>> cpc_config = CPCConfig(**configs['cpc'])
    """
    if gpu_type is None:
        gpu_type = detect_gpu()

    print(f"Configuring for GPU: {gpu_type}")

    if gpu_type == 'cpu':
        # Minimal config for CPU testing
        return {
            'cpc': {
                'batch_size': 32,
                'hidden_dim': 128,
                'embed_dim': 256,
                'lstm_layers': 1,
                'n_heads': 4,
                'ff_dim': 512,
                'ar_hidden': 128,
            },
            'regression': {
                'batch_size': 32,
            },
            'kelly': {},
            'gpu_type': 'cpu',
        }

    gpu_config = GPU_CONFIGS.get(gpu_type, GPU_CONFIGS['T4'])

    return {
        'cpc': gpu_config['cpc'],
        'regression': gpu_config['regression'],
        'kelly': {},
        'gpu_type': gpu_type,
    }


def create_configs(
    gpu_type: Optional[str] = None,
) -> Dict[str, any]:
    """
    Create full configuration objects for all components.

    Args:
        gpu_type: GPU type string. If None, auto-detect.

    Returns:
        Dictionary with CPCConfig, RegressionConfig, KellyConfig instances
    """
    gpu_overrides = get_config_for_gpu(gpu_type)

    cpc_config = CPCConfig(**gpu_overrides['cpc'])
    reg_config = RegressionConfig(**gpu_overrides['regression'])
    kelly_config = KellyConfig(**gpu_overrides.get('kelly', {}))

    return {
        'cpc': cpc_config,
        'regression': reg_config,
        'kelly': kelly_config,
        'gpu_type': gpu_overrides['gpu_type'],
    }


# Trading constants (from existing config)
FIXED_POSITION_SIZE = 0.01  # SOL
JITO_TIP = 0.00005
GAS_FEE = 0.0002
TOTAL_FEE_PER_TX = JITO_TIP + GAS_FEE  # 0.00025 SOL
TRANSACTION_COST_PCT = TOTAL_FEE_PER_TX / FIXED_POSITION_SIZE  # ~2.5% per trade
ROUND_TRIP_COST = 2 * TRANSACTION_COST_PCT  # ~5% round trip

# Lookahead settings
LOOKAHEAD_SECONDS = 20
DELAY_SECONDS = 1
MIN_HISTORY_LENGTH = 12

# Labeling thresholds (for reference, not used in regression)
TAKE_PROFIT_PCT = 0.05
STOP_LOSS_PCT = -0.06
