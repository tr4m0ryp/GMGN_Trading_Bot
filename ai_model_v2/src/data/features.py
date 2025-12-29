"""
Feature extraction for Multi-Model Trading Architecture.

This module implements feature engineering for each model:
    - Model 1 (Screener): Static + early dynamic features
    - Model 2 (Entry): Time-series features with rolling windows
    - Model 3 (Exit): Position-aware features

Author: Trading Team
Date: 2025-12-29
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .loader import TokenData, Candle


# =============================================================================
# Technical Indicator Functions
# =============================================================================

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
    """Calculate MACD (difference between fast and slow EMA)."""
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


def calculate_vwap(candles: List[Candle]) -> np.ndarray:
    """Calculate Volume-Weighted Average Price (VWAP)."""
    typical_prices = np.array([(c.high + c.low + c.close) / 3 for c in candles])
    volumes = np.array([c.volume for c in candles])

    cumulative_tp_vol = np.cumsum(typical_prices * volumes)
    cumulative_vol = np.cumsum(volumes)

    vwap = np.divide(
        cumulative_tp_vol,
        cumulative_vol,
        where=cumulative_vol != 0,
        out=typical_prices.copy(),
    )

    return vwap


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


# =============================================================================
# Model 1: Screener Features
# =============================================================================

def extract_screener_features(token: TokenData, decision_time: int = 30) -> np.ndarray:
    """
    Extract features for Model 1 (Screener) at a specific decision time.

    Features are a combination of:
    - Static features: Market cap bin, token age, etc.
    - Early dynamic features: Returns, volume, etc. from first N seconds

    Args:
        token: TokenData object with candles.
        decision_time: Time (in seconds) at which to extract features.

    Returns:
        Feature vector of shape (num_features,).
    """
    candles = token.candles[:decision_time]

    if len(candles) < 5:
        return None  # Not enough data

    prices = np.array([c.close for c in candles])
    volumes = np.array([c.volume for c in candles])
    entry_price = prices[0]

    # Static Features
    # ---------------

    # Market cap bins (one-hot encoded based on entry price as proxy)
    # In production, use actual MC from token metadata
    mc_sub_5k = 1.0 if entry_price < 5000 else 0.0
    mc_5k_10k = 1.0 if 5000 <= entry_price < 10000 else 0.0
    mc_10k_15k = 1.0 if 10000 <= entry_price < 15000 else 0.0
    mc_15k_20k = 1.0 if 15000 <= entry_price < 20000 else 0.0
    mc_above_20k = 1.0 if entry_price >= 20000 else 0.0

    # Token age at detection
    token_age = float(token.discovered_age_sec) / 600.0  # Normalize to 10 min

    # KOL count (placeholder - would come from token metadata)
    kol_count = 1.0  # Normalized placeholder

    # Dynamic Features (Early Window)
    # --------------------------------

    current_price = prices[-1]

    # Returns at various windows
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

    # Normalize volumes (log scale)
    volume_10s_log = np.log1p(volume_10s)
    volume_20s_log = np.log1p(volume_20s)
    volume_30s_log = np.log1p(volume_30s)

    # Transaction count proxy (use volume spikes as proxy)
    volume_changes = np.diff(volumes) if len(volumes) > 1 else np.array([0])
    tx_count_30s = float(np.sum(volume_changes > 0)) / 30.0  # Normalized

    # Buy/sell ratio (use price direction as proxy)
    price_changes = np.diff(prices) if len(prices) > 1 else np.array([0])
    buys = np.sum(price_changes > 0)
    sells = np.sum(price_changes < 0)
    buy_sell_ratio = buys / (buys + sells + 1e-6)

    # Largest transaction (max volume spike)
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

    # Assemble feature vector
    features = np.array([
        # Static (7)
        mc_sub_5k,
        mc_5k_10k,
        mc_10k_15k,
        mc_15k_20k,
        mc_above_20k,
        kol_count,
        token_age,
        # Dynamic (12)
        return_5s,
        return_10s,
        return_15s,
        return_30s,
        volume_10s_log,
        volume_20s_log,
        volume_30s_log,
        tx_count_30s,
        buy_sell_ratio,
        largest_tx_log,
        volume_accel,
        price_accel,
    ], dtype=np.float32)

    # Clip extreme values
    features = np.clip(features, -10.0, 10.0)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


# =============================================================================
# Model 2 & 3: Time-Series Features
# =============================================================================

def extract_timeseries_features(candles: List[Candle]) -> np.ndarray:
    """
    Extract time-series features for Models 2 and 3.

    Features for each timestep (14 features):
        0: log_close (relative to first candle)
        1: return_1s
        2: return_3s
        3: return_5s
        4: range_ratio = (high - low) / close
        5: volume_log = log1p(volume)
        6: rsi_norm (0-1)
        7: macd_norm (macd / close)
        8: bb_upper_dev = (bb_upper - close) / close
        9: bb_lower_dev = (bb_lower - close) / close
        10: vwap_dev = (vwap - close) / close
        11: momentum_10s
        12: order_imbalance (price direction proxy)
        13: drawdown_from_high

    Args:
        candles: List of Candle objects.

    Returns:
        Feature matrix of shape (num_candles, 14).
    """
    if not candles:
        return np.array([]).reshape(0, 14)

    prices = np.array([c.close for c in candles], dtype=np.float32)
    highs = np.array([c.high for c in candles], dtype=np.float32)
    lows = np.array([c.low for c in candles], dtype=np.float32)
    volumes = np.array([c.volume for c in candles], dtype=np.float32)

    n = len(prices)

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
        log_close,       # 0
        return_1s,       # 1
        return_3s,       # 2
        return_5s,       # 3
        range_ratio,     # 4
        volume_log,      # 5
        rsi_norm,        # 6
        macd_norm,       # 7
        bb_upper_dev,    # 8
        bb_lower_dev,    # 9
        vwap_dev,        # 10
        momentum,        # 11
        order_imbalance, # 12
        drawdown,        # 13
    ])

    # Clean up
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    features = np.clip(features, -10.0, 10.0)

    return features.astype(np.float32)


# =============================================================================
# Model 3: Exit-Specific Features
# =============================================================================

def extract_exit_features(
    candles: List[Candle],
    entry_idx: int,
    entry_price: float,
    current_idx: int,
) -> np.ndarray:
    """
    Extract features for Model 3 (Exit) at a specific point after entry.

    Includes position-aware features:
    - Current P&L
    - Time since entry
    - Distance from position high
    - Momentum since entry

    Args:
        candles: List of Candle objects.
        entry_idx: Index where position was entered.
        entry_price: Price at entry.
        current_idx: Current candle index.

    Returns:
        Feature vector including position context.
    """
    # Get time-series features up to current point
    current_candles = candles[:current_idx + 1]
    ts_features = extract_timeseries_features(current_candles)

    if len(ts_features) == 0:
        return None

    # Take last timestep features
    base_features = ts_features[-1]

    # Position-specific features
    current_price = candles[current_idx].close
    position_candles = candles[entry_idx:current_idx + 1]

    # Unrealized P&L
    unrealized_pnl = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0

    # Time since entry
    time_since_entry = (current_idx - entry_idx) / 300.0  # Normalize to 5 min

    # Position high and drawdown from it
    position_prices = [c.close for c in position_candles]
    position_high = max(position_prices) if position_prices else entry_price
    drawdown_from_position_high = (current_price - position_high) / position_high if position_high > 0 else 0.0

    # Momentum since entry
    if len(position_prices) >= 2:
        momentum_since_entry = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0
    else:
        momentum_since_entry = 0.0

    # Volume trend since entry
    position_volumes = [c.volume for c in position_candles]
    if len(position_volumes) >= 2:
        vol_first = np.mean(position_volumes[:len(position_volumes)//2 + 1])
        vol_second = np.mean(position_volumes[len(position_volumes)//2:])
        volume_trend = (vol_second - vol_first) / (vol_first + 1e-6)
    else:
        volume_trend = 0.0

    # RSI at current point (for exhaustion detection)
    current_rsi = calculate_rsi(np.array([c.close for c in current_candles]))[-1] / 100.0

    # Assemble exit features
    exit_features = np.concatenate([
        base_features,  # 14 features
        np.array([
            unrealized_pnl,
            time_since_entry,
            drawdown_from_position_high,
            momentum_since_entry,
            volume_trend,
            current_rsi,
        ], dtype=np.float32),
    ])

    # Clean up
    exit_features = np.nan_to_num(exit_features, nan=0.0, posinf=0.0, neginf=0.0)
    exit_features = np.clip(exit_features, -10.0, 10.0)

    return exit_features.astype(np.float32)


# =============================================================================
# Feature Extractor Class
# =============================================================================

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

        # Truncate to max length if needed
        if len(features) > self.max_seq_len:
            features = features[-self.max_seq_len:]
            # Re-normalize log_close
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
