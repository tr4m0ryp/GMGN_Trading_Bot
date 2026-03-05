"""
Feature extraction for Multi-Model Trading Architecture.

Provides screener (Model 1) feature extraction and time-series
feature extraction used by entry (Model 2) and exit (Model 3) models.

Dependencies: numpy, pandas
Date: 2025-12-29
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from .data_loading import Candle, TokenData
from .technical_indicators import (
    calculate_rsi,
    calculate_macd,
    calculate_bollinger,
    calculate_vwap,
)


# =============================================================================
# Screener Feature Extraction (Model 1)
# =============================================================================

def extract_screener_features(token: TokenData, decision_time: int = 30) -> Optional[np.ndarray]:
    """
    Extract features for Model 1 (Screener).

    This is the ORIGINAL working feature extraction that achieved ROC-AUC 0.77.
    Features focus on price changes, volume patterns, and momentum.
    """
    candles = token.candles[:decision_time]
    if len(candles) < 5:
        return None

    prices = np.array([c.close for c in candles])
    highs = np.array([c.high for c in candles])
    lows = np.array([c.low for c in candles])
    volumes = np.array([c.volume for c in candles])

    entry_price = prices[0]
    current_price = prices[-1]

    if entry_price <= 0:
        return None

    # === PRICE CHANGE FEATURES ===
    def safe_return(idx):
        if idx < len(prices) and idx > 0:
            return (current_price - prices[-idx]) / prices[-idx] if prices[-idx] > 0 else 0.0
        return 0.0

    price_chg_5s = safe_return(5)
    price_chg_15s = safe_return(15)
    price_chg_30s = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0

    # === VOLUME CHANGE FEATURES ===
    def safe_vol_change(idx):
        if idx < len(volumes) and idx > 0:
            recent = np.sum(volumes[-idx:])
            earlier = np.sum(volumes[:-idx]) if len(volumes) > idx else recent
            return (recent - earlier) / (earlier + 1e-6) if earlier > 0 else 0.0
        return 0.0

    vol_chg_5s = safe_vol_change(5)
    vol_chg_15s = safe_vol_change(15)
    vol_chg_30s = safe_vol_change(30)

    # === PRICE RANGE FEATURES ===
    period_high = np.max(highs)
    period_low = np.min(lows)

    high_low_ratio = (period_high - period_low) / entry_price if entry_price > 0 else 0.0
    close_to_high = (current_price - period_high) / period_high if period_high > 0 else 0.0
    close_to_low = (current_price - period_low) / period_low if period_low > 0 else 0.0

    # === VOLATILITY FEATURES ===
    def calc_volatility(window):
        if window > len(prices):
            window = len(prices)
        if window < 2:
            return 0.0
        subset = prices[-window:]
        returns = np.diff(subset) / subset[:-1]
        return np.std(returns) if len(returns) > 0 else 0.0

    volatility_5s = calc_volatility(5)
    volatility_15s = calc_volatility(15)
    volatility_30s = calc_volatility(30)

    # === TREND FEATURES ===
    if len(prices) >= 5:
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        trend_strength = slope / (entry_price + 1e-6)
    else:
        trend_strength = 0.0

    # Momentum (rate of recent vs earlier changes)
    if len(prices) >= 10:
        mid = len(prices) // 2
        early_change = (prices[mid] - prices[0]) / (prices[0] + 1e-6)
        late_change = (prices[-1] - prices[mid]) / (prices[mid] + 1e-6)
        momentum = late_change - early_change
    else:
        momentum = 0.0

    # === RSI-LIKE FEATURE ===
    if len(prices) > 1:
        changes = np.diff(prices)
        gains = np.sum(changes[changes > 0])
        losses = abs(np.sum(changes[changes < 0]))
        rsi = gains / (gains + losses + 1e-6)
    else:
        rsi = 0.5

    # === VOLUME TREND ===
    if len(volumes) >= 10:
        first_half = np.mean(volumes[:len(volumes)//2])
        second_half = np.mean(volumes[len(volumes)//2:])
        volume_trend = (second_half - first_half) / (first_half + 1e-6)
    else:
        volume_trend = 0.0

    # === PRICE ACCELERATION ===
    if len(prices) >= 10:
        mid = len(prices) // 2
        first_vel = (prices[mid] - prices[0]) / mid if mid > 0 else 0
        second_vel = (prices[-1] - prices[mid]) / (len(prices) - mid) if len(prices) > mid else 0
        price_acceleration = (second_vel - first_vel) / (entry_price + 1e-6)
    else:
        price_acceleration = 0.0

    # === CANDLE PATTERN FEATURES ===
    bodies = prices - np.array([c.open for c in candles])
    candle_body_ratio = np.mean(bodies) / (entry_price + 1e-6)

    green_candles = np.sum(bodies > 0)
    num_green_candles = green_candles / len(candles)

    # === ASSEMBLE FEATURES ===
    features = np.array([
        price_chg_5s, price_chg_15s, price_chg_30s,
        vol_chg_5s, vol_chg_15s, vol_chg_30s,
        high_low_ratio, close_to_high, close_to_low,
        volatility_5s, volatility_15s, volatility_30s,
        trend_strength, momentum, rsi,
        volume_trend, price_acceleration, candle_body_ratio, num_green_candles,
    ], dtype=np.float32)

    features = np.clip(features, -10.0, 10.0)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


# Feature names for screener (must match extract_screener_features output)
SCREENER_FEATURE_NAMES = [
    "price_chg_5s", "price_chg_15s", "price_chg_30s",
    "vol_chg_5s", "vol_chg_15s", "vol_chg_30s",
    "high_low_ratio", "close_to_high", "close_to_low",
    "volatility_5s", "volatility_15s", "volatility_30s",
    "trend_strength", "momentum", "rsi",
    "volume_trend", "price_acceleration", "candle_body_ratio", "num_green_candles",
]


# =============================================================================
# Time-Series Feature Extraction (Models 2 and 3)
# =============================================================================

def extract_timeseries_features(candles: List[Candle]) -> np.ndarray:
    """Extract time-series features for Models 2 & 3.

    Returns an (N, 14) array where N is the number of candles.
    """
    if not candles:
        return np.array([]).reshape(0, 14)

    prices = np.array([c.close for c in candles], dtype=np.float32)
    highs = np.array([c.high for c in candles], dtype=np.float32)
    lows = np.array([c.low for c in candles], dtype=np.float32)
    volumes = np.array([c.volume for c in candles], dtype=np.float32)

    # Technical indicators
    rsi = calculate_rsi(prices)
    macd = calculate_macd(prices)
    bb_upper, bb_middle, bb_lower = calculate_bollinger(prices)
    vwap = calculate_vwap(candles)

    # Log close relative to first
    log_close = np.log(np.clip(prices, 1e-10, None)) - np.log(np.clip(prices[0], 1e-10, None))

    # Returns
    def calc_returns(arr, lag):
        if len(arr) <= lag:
            return np.zeros(len(arr))
        safe = np.clip(arr, 1e-10, None)
        logs = np.log(safe)
        shifted = np.concatenate([np.full(lag, logs[0]), logs[:-lag]])
        return logs - shifted

    return_1s = calc_returns(prices, 1)
    return_3s = calc_returns(prices, 3)
    return_5s = calc_returns(prices, 5)

    # Other features
    range_ratio = np.where(prices != 0, (highs - lows) / prices, 0.0)
    volume_log = np.log1p(np.clip(volumes, 0, None))
    rsi_norm = rsi / 100.0
    macd_norm = np.where(prices != 0, macd / prices, 0.0)
    bb_upper_dev = np.where(prices != 0, (bb_upper - prices) / prices, 0.0)
    bb_lower_dev = np.where(prices != 0, (bb_lower - prices) / prices, 0.0)
    vwap_dev = np.where(prices != 0, (vwap - prices) / prices, 0.0)

    # Momentum
    momentum = np.zeros(len(prices))
    for i in range(len(prices)):
        lookback_idx = max(0, i - 10)
        if prices[lookback_idx] != 0:
            momentum[i] = (prices[i] - prices[lookback_idx]) / prices[lookback_idx]

    # Order imbalance
    price_changes = np.concatenate([[0], np.diff(prices)])
    order_imbalance = np.sign(price_changes)

    # Drawdown
    rolling_high = pd.Series(prices).rolling(window=30, min_periods=1).max().values
    drawdown = np.where(rolling_high != 0, (prices - rolling_high) / rolling_high, 0.0)

    features = np.column_stack([
        log_close, return_1s, return_3s, return_5s,
        range_ratio, volume_log, rsi_norm, macd_norm,
        bb_upper_dev, bb_lower_dev, vwap_dev, momentum,
        order_imbalance, drawdown,
    ])

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    features = np.clip(features, -10.0, 10.0)

    return features.astype(np.float32)
