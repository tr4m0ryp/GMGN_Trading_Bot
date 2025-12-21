"""
AI Trading Model Package.

This package contains the complete implementation of the variable-length
LSTM trading model for GMGN token trading.

Modules:
    config: Configuration management
    utils: Utility functions
    data: Data loading and feature extraction
    models: LSTM model architecture
    training: Training logic
    evaluation: Evaluation and backtesting

Author: Trading Team
Date: 2025-12-21
"""

__version__ = '1.0.0'

from .config import get_config, DEFAULT_CONFIG
from .utils import set_seed, get_device, count_parameters
from .models import VariableLengthLSTMTrader
from .data import (
    prepare_datasets,
    load_preprocessed_datasets,
    TradingDataset,
    collate_variable_length,
)
from .training import train_model
from .evaluation import evaluate_model, comprehensive_backtest

__all__ = [
    'get_config',
    'DEFAULT_CONFIG',
    'set_seed',
    'get_device',
    'count_parameters',
    'VariableLengthLSTMTrader',
    'prepare_datasets',
    'load_preprocessed_datasets',
    'TradingDataset',
    'collate_variable_length',
    'train_model',
    'evaluate_model',
    'comprehensive_backtest',
]
