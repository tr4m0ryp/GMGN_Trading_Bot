"""
Training modules for CPC + Return Regression model.

This module provides training pipelines for:
1. Phase 1: CPC Pretraining (self-supervised)
2. Phase 2: Return Regression (supervised fine-tuning)

Components:
    - train_cpc: Phase 1 CPC pretraining function
    - train_regression: Phase 2 regression fine-tuning function

Dependencies:
    torch, numpy, tqdm

Date: 2025-12-25
"""

from .cpc_trainer import train_cpc
from .regression_trainer import train_regression

__all__ = [
    'train_cpc',
    'train_regression',
]
