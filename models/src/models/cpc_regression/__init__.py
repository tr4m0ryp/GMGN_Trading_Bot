"""
CPC + Return Regression Trading Model.

This module implements a two-phase training approach:
1. CPC Pretraining: Self-supervised learning on price sequences
2. Return Regression: Predict returns with uncertainty quantification

Components:
    - CPCEncoder: Shared encoder (LSTM + Attention)
    - CPCModel: Contrastive Predictive Coding with InfoNCE loss
    - ProbabilisticReturnHead: Multi-task return + drawdown prediction
    - KellyPositionSizer: Optimal position sizing

Dependencies:
    torch, numpy

Date: 2025-12-25
"""

from .encoder import CPCEncoder
from .cpc_model import CPCModel
from .return_head import ProbabilisticReturnHead
from .kelly_sizer import KellyPositionSizer
from .kelly_utils import KellyBacktester
from .config import CPCConfig, RegressionConfig, KellyConfig

__all__ = [
    'CPCEncoder',
    'CPCModel',
    'ProbabilisticReturnHead',
    'KellyPositionSizer',
    'CPCConfig',
    'RegressionConfig',
    'KellyConfig',
]
