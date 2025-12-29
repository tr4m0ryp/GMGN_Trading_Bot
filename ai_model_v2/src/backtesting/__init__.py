"""
Backtesting and evaluation module for Multi-Model Trading Architecture.

This module provides:
    - Full system backtesting with all three models
    - Performance metrics and analysis
    - Trade-by-trade reporting

Author: Trading Team
Date: 2025-12-29
"""

from .backtester import Backtester, BacktestResult, run_backtest
from .metrics import calculate_metrics, print_metrics_report

__all__ = [
    "Backtester",
    "BacktestResult",
    "run_backtest",
    "calculate_metrics",
    "print_metrics_report",
]
