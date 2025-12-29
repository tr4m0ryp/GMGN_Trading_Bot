"""
Training module for Multi-Model Trading Architecture.

This module provides training functions for:
    - Model 1 (Screener): XGBoost training
    - Model 2 (Entry): LSTM training with PyTorch
    - Model 3 (Exit): XGBoost + rules training

Author: Trading Team
Date: 2025-12-29
"""

from .screener_trainer import train_screener_model
from .entry_trainer import train_entry_model
from .exit_trainer import train_exit_model

__all__ = [
    "train_screener_model",
    "train_entry_model",
    "train_exit_model",
]
