"""
Model implementations for Multi-Model Trading Architecture.

This module contains:
    - Model 1 (Screener): XGBoost classifier
    - Model 2 (Entry): LSTM/Hybrid entry timing model
    - Model 3 (Exit): Exit point optimizer with risk rules

Author: Trading Team
Date: 2025-12-29
"""

from .screener_model import ScreenerModel
from .screener import train_screener, evaluate_screener
from .entry_model import EntryEncoder, EntryModel
from .entry import create_entry_model
from .exit import RiskManager, TradingPipeline
from .exit_model import ExitModel

__all__ = [
    "ScreenerModel",
    "train_screener",
    "evaluate_screener",
    "EntryEncoder",
    "EntryModel",
    "create_entry_model",
    "ExitModel",
    "RiskManager",
    "TradingPipeline",
]
