"""
Technical indicator calculations for feature extraction.

Provides RSI, MACD, Bollinger Bands, VWAP, momentum, returns,
and drawdown computations used by screener, time-series, and
exit feature extractors.

Dependencies: numpy, pandas
Date: 2025-12-29
"""

from typing import List, Tuple

import numpy as np
import pandas as pd

from .loader import Candle


def calculate_ema(prices: np.ndarray, span: int) -> np.ndarray:
    """Calculate Exponential Moving Average."""
    return pd.Series(prices).ewm(span=span, adjust=False).mean().values


def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index (RSI).

    Args:
        prices: Array of closing prices.
        period: RSI period (default 14).

    Returns:
        Array of RSI values (0-100 range).
    """
    if len(prices) < 2:
        return np.full(len(prices), 50.0)

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    rsi_values = np.zeros(len(prices))
    rsi_values[0] = 50.0

    for i in range(1, len(prices)):
        window_start = max(0, i - period)
        avg_gain = np.mean(gains[window_start:i]) if i > window_start else 0
        avg_loss = np.mean(losses[window_start:i]) if i > window_start else 0

        if avg_loss == 0:
            rsi_values[i] = 100.0 if avg_gain > 0 else 50.0
        else:
            rs = avg_gain / avg_loss
            rsi_values[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi_values


def calculate_macd(prices: np.ndarray, fast: int = 12, slow: int = 26) -> np.ndarray:
    """Calculate MACD (Moving Average Convergence Divergence)."""
    if len(prices) < 2:
        return np.zeros(len(prices))

    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)

    return ema_fast - ema_slow


def calculate_bollinger_bands(
    prices: np.ndarray, period: int = 20, num_std: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Bollinger Bands.

    Returns:
        Tuple of (upper_band, middle_band, lower_band).
    """
    if len(prices) < 2:
        p = prices[0] if len(prices) > 0 else 0
        return np.full(len(prices), p), np.full(len(prices), p), np.full(len(prices), p)

    sma = pd.Series(prices).rolling(window=period, min_periods=1).mean().values
    std = pd.Series(prices).rolling(window=period, min_periods=1).std().values
    std = np.nan_to_num(std, nan=0.0)

    upper = sma + (num_std * std)
    lower = sma - (num_std * std)

    return upper, sma, lower


# Alias for backward compatibility with preprocess.py split
calculate_bollinger = calculate_bollinger_bands


def calculate_vwap(candles: List[Candle]) -> np.ndarray:
    """Calculate Volume Weighted Average Price."""
    typical_prices = np.array([(c.high + c.low + c.close) / 3 for c in candles])
    volumes = np.array([c.volume for c in candles])

    cumulative_tp_vol = np.cumsum(typical_prices * volumes)
    cumulative_vol = np.cumsum(volumes)

    return np.divide(
        cumulative_tp_vol, cumulative_vol,
        where=cumulative_vol != 0, out=typical_prices.copy(),
    )


def calculate_returns(prices: np.ndarray, lag: int) -> np.ndarray:
    """Calculate log returns with given lag."""
    if len(prices) <= lag:
        return np.zeros(len(prices))

    safe = np.clip(prices, 1e-10, None)
    logs = np.log(safe)
    shifted = np.concatenate([np.full(lag, logs[0]), logs[:-lag]])

    return logs - shifted


def calculate_momentum(prices: np.ndarray, period: int = 10) -> np.ndarray:
    """Calculate price momentum (rate of change)."""
    momentum = np.zeros(len(prices))

    for i in range(len(prices)):
        lookback_idx = max(0, i - period)
        if prices[lookback_idx] != 0:
            momentum[i] = (prices[i] - prices[lookback_idx]) / prices[lookback_idx]

    return momentum


def calculate_rolling_high(prices: np.ndarray, window: int = 30) -> np.ndarray:
    """Calculate rolling maximum."""
    return pd.Series(prices).rolling(window=window, min_periods=1).max().values


def calculate_rolling_low(prices: np.ndarray, window: int = 30) -> np.ndarray:
    """Calculate rolling minimum."""
    return pd.Series(prices).rolling(window=window, min_periods=1).min().values


def calculate_drawdown(prices: np.ndarray, window: int = 30) -> np.ndarray:
    """Calculate drawdown from rolling high."""
    rolling_high = calculate_rolling_high(prices, window)
    drawdown = np.where(
        rolling_high != 0, (prices - rolling_high) / rolling_high, 0.0
    )
    return drawdown
