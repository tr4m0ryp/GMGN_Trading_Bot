"""
AI Trading Model Package.

This package contains the complete implementation of the CPC + Return Regression
trading model and RL-based trading environment for GMGN token trading.

Modules:
    config: Configuration management (trading constants, model configs)
    utils: Utility functions (seeding, logging, checkpoints)
    data: Data loading and preprocessing
    cpc_regression: CPC pretraining + Return regression models
    training: Training pipelines (CPC and Regression trainers)
    rl: Reinforcement learning environment and trainer

Date: 2025-12-25
"""

__version__ = '2.0.0'

# Import configuration utilities
from config import get_config, DEFAULT_CONFIG, ADVANCED_CONFIG

# Import utility functions
from utils import set_seed, get_device, count_parameters

# CPC + Regression components (main training approach)
from cpc_regression import (
    CPCEncoder,
    CPCModel,
    ProbabilisticReturnHead,
    KellyPositionSizer,
    CPCConfig,
    RegressionConfig,
    KellyConfig,
)

# Training pipelines
from training import train_cpc, train_regression

__all__ = [
    # Version
    '__version__',
    # Config
    'get_config',
    'DEFAULT_CONFIG',
    'ADVANCED_CONFIG',
    # Utils
    'set_seed',
    'get_device',
    'count_parameters',
    # CPC + Regression
    'CPCEncoder',
    'CPCModel',
    'ProbabilisticReturnHead',
    'KellyPositionSizer',
    'CPCConfig',
    'RegressionConfig',
    'KellyConfig',
    # Training
    'train_cpc',
    'train_regression',
]
