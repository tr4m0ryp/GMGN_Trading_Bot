"""
Feature extraction functions for Multi-Model Trading Architecture.

This module implements feature engineering for each model:
    - Model 1 (Screener): Static + early dynamic features
    - Model 2 (Entry): Time-series features with rolling windows
    - Model 3 (Exit): Position-aware features

Author: Trading Team
Date: 2025-12-29
"""

from typing import List, Optional
from dataclasses import dataclass

import numpy as np

from .loader import TokenData, Candle
from .technical_indicators import (
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_vwap,
    calculate_returns,
    calculate_momentum,
    calculate_drawdown,
)


def extract_screener_features(token: TokenData, decision_time: int = 30) -> Optional[np.ndarray]:
    """
    Extract static + early dynamic features for Model 1 (Screener).

    Returns feature vector of shape (19,), or None if insufficient data.
    """
    candles = token.candles[:decision_time]

    if len(candles) < 5:
        return None

    prices = np.array([c.close for c in candles])
    volumes = np.array([c.volume for c in candles])
    entry_price = prices[0]

    # Static Features
    mc_sub_5k = 1.0 if entry_price < 5000 else 0.0
    mc_5k_10k = 1.0 if 5000 <= entry_price < 10000 else 0.0
    mc_10k_15k = 1.0 if 10000 <= entry_price < 15000 else 0.0
    mc_15k_20k = 1.0 if 15000 <= entry_price < 20000 else 0.0
    mc_above_20k = 1.0 if entry_price >= 20000 else 0.0

    token_age = float(token.discovered_age_sec) / 600.0
    kol_count = 1.0  # Normalized placeholder

    # Dynamic Features (Early Window)
    current_price = prices[-1]

    def safe_return(idx):
        if idx < len(prices) and idx > 0:
            return (current_price - prices[-idx]) / prices[-idx] if prices[-idx] > 0 else 0.0
        return 0.0

    return_5s = safe_return(5)
    return_10s = safe_return(10)
    return_15s = safe_return(15)
    return_30s = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0

    # Volume features
    total_volume = np.sum(volumes)
    volume_10s = np.sum(volumes[-10:]) if len(volumes) >= 10 else np.sum(volumes)
    volume_20s = np.sum(volumes[-20:]) if len(volumes) >= 20 else np.sum(volumes)
    volume_30s = total_volume

    volume_10s_log = np.log1p(volume_10s)
    volume_20s_log = np.log1p(volume_20s)
    volume_30s_log = np.log1p(volume_30s)

    # Transaction count proxy
    volume_changes = np.diff(volumes) if len(volumes) > 1 else np.array([0])
    tx_count_30s = float(np.sum(volume_changes > 0)) / 30.0

    # Buy/sell ratio
    price_changes = np.diff(prices) if len(prices) > 1 else np.array([0])
    buys = np.sum(price_changes > 0)
    sells = np.sum(price_changes < 0)
    buy_sell_ratio = buys / (buys + sells + 1e-6)

    # Largest transaction
    largest_tx = np.max(volumes) if len(volumes) > 0 else 0.0
    largest_tx_log = np.log1p(largest_tx)

    # Volume acceleration
    if len(volumes) >= 10:
        vol_first_half = np.mean(volumes[:len(volumes)//2])
        vol_second_half = np.mean(volumes[len(volumes)//2:])
        volume_accel = (vol_second_half - vol_first_half) / (vol_first_half + 1e-6)
    else:
        volume_accel = 0.0

    # Price acceleration
    if len(prices) >= 10:
        price_first = (prices[len(prices)//2] - prices[0]) / (prices[0] + 1e-6)
        price_second = (prices[-1] - prices[len(prices)//2]) / (prices[len(prices)//2] + 1e-6)
        price_accel = price_second - price_first
    else:
        price_accel = 0.0

    features = np.array([
        # Static (7)
        mc_sub_5k, mc_5k_10k, mc_10k_15k, mc_15k_20k, mc_above_20k,
        kol_count, token_age,
        # Dynamic (12)
        return_5s, return_10s, return_15s, return_30s,
        volume_10s_log, volume_20s_log, volume_30s_log,
        tx_count_30s, buy_sell_ratio, largest_tx_log,
        volume_accel, price_accel,
    ], dtype=np.float32)

    features = np.clip(features, -10.0, 10.0)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


def extract_timeseries_features(candles: List[Candle]) -> np.ndarray:
    """
    Extract 14 time-series features per timestep for Models 2 and 3.

    Features: log_close, return_{1,3,5}s, range_ratio, volume_log,
    rsi_norm, macd_norm, bb_{upper,lower}_dev, vwap_dev, momentum,
    order_imbalance, drawdown_from_high.

    Returns:
        Feature matrix of shape (num_candles, 14).
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
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(prices)
    vwap = calculate_vwap(candles)
    momentum = calculate_momentum(prices, period=10)
    drawdown = calculate_drawdown(prices, window=30)

    # Log close relative to first candle
    log_close = np.log(np.clip(prices, 1e-10, None)) - np.log(np.clip(prices[0], 1e-10, None))

    # Returns
    return_1s = calculate_returns(prices, 1)
    return_3s = calculate_returns(prices, 3)
    return_5s = calculate_returns(prices, 5)

    # Range ratio
    range_ratio = np.where(prices != 0, (highs - lows) / prices, 0.0)

    # Volume log
    volume_log = np.log1p(np.clip(volumes, 0, None))

    # Normalized indicators
    rsi_norm = rsi / 100.0
    macd_norm = np.where(prices != 0, macd / prices, 0.0)
    bb_upper_dev = np.where(prices != 0, (bb_upper - prices) / prices, 0.0)
    bb_lower_dev = np.where(prices != 0, (bb_lower - prices) / prices, 0.0)
    vwap_dev = np.where(prices != 0, (vwap - prices) / prices, 0.0)

    # Order imbalance (using price direction as proxy)
    price_changes = np.concatenate([[0], np.diff(prices)])
    order_imbalance = np.sign(price_changes)

    # Assemble feature matrix
    features = np.column_stack([
        log_close, return_1s, return_3s, return_5s,
        range_ratio, volume_log, rsi_norm, macd_norm,
        bb_upper_dev, bb_lower_dev, vwap_dev, momentum,
        order_imbalance, drawdown,
    ])

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    features = np.clip(features, -10.0, 10.0)

    return features.astype(np.float32)


def extract_exit_features(
    candles: List[Candle],
    entry_idx: int,
    entry_price: float,
    current_idx: int,
) -> Optional[np.ndarray]:
    """
    Extract position-aware features for Model 3 (Exit).

    Combines base time-series features with P&L, time in position,
    drawdown from position high, momentum, volume trend, and RSI.
    Returns None if insufficient data.
    """
    current_candles = candles[:current_idx + 1]
    ts_features = extract_timeseries_features(current_candles)

    if len(ts_features) == 0:
        return None

    base_features = ts_features[-1]

    current_price = candles[current_idx].close
    position_candles = candles[entry_idx:current_idx + 1]

    unrealized_pnl = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0
    time_since_entry = (current_idx - entry_idx) / 300.0

    position_prices = [c.close for c in position_candles]
    position_high = max(position_prices) if position_prices else entry_price
    drawdown_from_position_high = (
        (current_price - position_high) / position_high if position_high > 0 else 0.0
    )

    if len(position_prices) >= 2:
        momentum_since_entry = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0
    else:
        momentum_since_entry = 0.0

    position_volumes = [c.volume for c in position_candles]
    if len(position_volumes) >= 2:
        vol_first = np.mean(position_volumes[:len(position_volumes)//2 + 1])
        vol_second = np.mean(position_volumes[len(position_volumes)//2:])
        volume_trend = (vol_second - vol_first) / (vol_first + 1e-6)
    else:
        volume_trend = 0.0

    current_rsi = calculate_rsi(np.array([c.close for c in current_candles]))[-1] / 100.0

    exit_features = np.concatenate([
        base_features,
        np.array([
            unrealized_pnl,
            time_since_entry,
            drawdown_from_position_high,
            momentum_since_entry,
            volume_trend,
            current_rsi,
        ], dtype=np.float32),
    ])

    exit_features = np.nan_to_num(exit_features, nan=0.0, posinf=0.0, neginf=0.0)
    exit_features = np.clip(exit_features, -10.0, 10.0)

    return exit_features.astype(np.float32)


@dataclass
class FeatureExtractor:
    """
    Unified feature extractor for all models.

    Provides consistent interface for extracting features for
    screening, entry timing, and exit optimization.
    """

    screener_decision_time: int = 30
    max_seq_len: int = 120

    def extract_for_screener(self, token: TokenData) -> Optional[np.ndarray]:
        """Extract features for Model 1 (Screener)."""
        return extract_screener_features(token, self.screener_decision_time)

    def extract_for_entry(self, candles: List[Candle]) -> np.ndarray:
        """Extract time-series features for Model 2 (Entry)."""
        features = extract_timeseries_features(candles)

        if len(features) > self.max_seq_len:
            features = features[-self.max_seq_len:]
            features[:, 0] = features[:, 0] - features[0, 0]

        return features

    def extract_for_exit(
        self,
        candles: List[Candle],
        entry_idx: int,
        entry_price: float,
        current_idx: int,
    ) -> Optional[np.ndarray]:
        """Extract features for Model 3 (Exit)."""
        return extract_exit_features(candles, entry_idx, entry_price, current_idx)
