"""
Feature extraction for Multi-Model Trading Architecture.

This module re-exports from technical_indicators and feature_extractors
for backward compatibility. New code should import directly from
those submodules.

Author: Trading Team
Date: 2025-12-29
"""

# Re-export technical indicators
from .technical_indicators import (
    calculate_ema,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_vwap,
    calculate_returns,
    calculate_momentum,
    calculate_rolling_high,
    calculate_rolling_low,
    calculate_drawdown,
)

# Re-export feature extractors
from .feature_extractors import (
    extract_screener_features,
    extract_timeseries_features,
    extract_exit_features,
    FeatureExtractor,
)

__all__ = [
    # Technical indicators
    "calculate_ema",
    "calculate_rsi",
    "calculate_macd",
    "calculate_bollinger_bands",
    "calculate_vwap",
    "calculate_returns",
    "calculate_momentum",
    "calculate_rolling_high",
    "calculate_rolling_low",
    "calculate_drawdown",
    # Feature extractors
    "extract_screener_features",
    "extract_timeseries_features",
    "extract_exit_features",
    "FeatureExtractor",
]
