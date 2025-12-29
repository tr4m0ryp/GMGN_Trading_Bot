"""
Multi-Model Trading Architecture for Solana Memecoins.

This package implements a three-model ensemble for trading:
    - Model 1 (Screener): Entry Worthiness classification
    - Model 2 (Entry): Entry Timing optimization
    - Model 3 (Exit): Exit Point optimization

Author: Trading Team
Date: 2025-12-29
"""

from . import config
from . import data
from . import models
from . import training
from . import utils
from . import backtesting

__version__ = "2.0.0"
__all__ = ["config", "data", "models", "training", "utils", "backtesting"]
