"""
Data preprocessing script for Multi-Model Trading Architecture.

This script processes raw token data and generates training datasets
for all three models. Self-contained to avoid import issues.

Usage:
    python preprocess.py --csv-path ../../ai_data/data/combined_tokens.csv

Author: Trading Team
Date: 2025-12-29
"""

import argparse
import pickle
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import IntEnum
import sys

import numpy as np
import pandas as pd


# =============================================================================
# Configuration Constants
# =============================================================================

SCREENER_DECISION_TIME = 30
SCREENER_WORTHY_THRESHOLD = 1.5  # Lowered from 2.0 to increase positive samples
SCREENER_LOOKAHEAD_SEC = 300
ENTRY_PROFIT_THRESHOLD = 0.10  # Relaxed for more entry signals
ENTRY_MAX_DRAWDOWN = 0.15  # Relaxed to allow reasonable volatility
ENTRY_LOOKAHEAD_SEC = 90  # Balanced lookahead window
ENTRY_SAMPLES_PER_TOKEN = 20  # Balanced sampling
EXIT_TRAILING_STOP_PCT = 0.15
EXIT_STOP_LOSS_PCT = 0.25
EXIT_PROFIT_TARGET_PCT = 2.00
DELAY_SECONDS = 1
TOTAL_FEE_PER_TX = 0.024
FIXED_POSITION_SIZE_SOL = 0.1


# =============================================================================
# Label Enums
# =============================================================================

class ScreenerLabel(IntEnum):
    AVOID = 0
    WORTHY = 1


class EntryLabel(IntEnum):
    WAIT = 0
    ENTER_NOW = 1
    ABORT = 2


class ExitLabel(IntEnum):
    HOLD = 0
    EXIT_NOW = 1
    PARTIAL_EXIT = 2


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Candle:
    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    @classmethod
    def from_dict(cls, d: Dict) -> "Candle":
        return cls(
            time=int(d.get("time", d.get("t", 0))),
            open=float(d.get("open", d.get("o", 0))),
            high=float(d.get("high", d.get("h", 0))),
            low=float(d.get("low", d.get("l", 0))),
            close=float(d.get("close", d.get("c", 0))),
            volume=float(d.get("volume", d.get("v", 0))),
        )

    def is_valid(self) -> bool:
        return self.close > 0 and self.open > 0


@dataclass
class TokenData:
    token_address: str
    symbol: str
    discovered_at_unix: int
    discovered_age_sec: int
    death_reason: str
    candles: List[Candle]
    lifespan_sec: int = field(init=False)
    peak_price: float = field(init=False)
    peak_time_sec: int = field(init=False)
    peak_ratio: float = field(init=False)
    entry_mc: float = field(init=False)

    def __post_init__(self):
        if self.candles:
            self.lifespan_sec = len(self.candles)
            prices = [c.close for c in self.candles]
            self.peak_price = max(prices)
            self.peak_time_sec = prices.index(self.peak_price)
            self.entry_mc = self.candles[0].close if self.candles else 0
            self.peak_ratio = self.peak_price / self.entry_mc if self.entry_mc > 0 else 1.0
        else:
            self.lifespan_sec = 0
            self.peak_price = 0
            self.peak_time_sec = 0
            self.entry_mc = 0
            self.peak_ratio = 1.0


# =============================================================================
# Data Loading
# =============================================================================

def parse_candles(candles_json: str) -> List[Candle]:
    """Parse JSON candles string."""
    data = json.loads(candles_json)

    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], dict) and "list" in data["data"]:
            raw_candles = data["data"]["list"]
        elif "list" in data:
            raw_candles = data["list"]
        else:
            return []
    elif isinstance(data, list):
        raw_candles = data
    else:
        return []

    candles = []
    for c in raw_candles:
        try:
            candle = Candle.from_dict(c)
            if candle.is_valid():
                candles.append(candle)
        except (ValueError, TypeError):
            continue

    candles.sort(key=lambda x: x.time)
    return candles


def load_raw_data(csv_path: str) -> List[TokenData]:
    """Load tokens from CSV."""
    df = pd.read_csv(csv_path, quotechar='"', escapechar="\\", on_bad_lines="warn", engine="python")

    candle_col = None
    for name in ["chart_data_json", "candles"]:
        if name in df.columns:
            candle_col = name
            break

    if candle_col is None:
        raise ValueError("Missing candle data column")

    tokens = []
    for _, row in df.iterrows():
        try:
            candles = parse_candles(row[candle_col])
            if len(candles) < 10:
                continue

            token = TokenData(
                token_address=row["token_address"],
                symbol=row.get("symbol", "UNKNOWN"),
                discovered_at_unix=int(row.get("discovered_at_unix", 0)),
                discovered_age_sec=int(row.get("discovered_age_sec", 0)),
                death_reason=str(row.get("death_reason", "unknown")),
                candles=candles,
            )
            tokens.append(token)
        except Exception:
            continue

    return tokens


def split_tokens_chronological(
    tokens: List[TokenData],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[List[TokenData], List[TokenData], List[TokenData]]:
    """Split tokens chronologically."""
    sorted_tokens = sorted(tokens, key=lambda t: t.discovered_at_unix)
    n = len(sorted_tokens)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return sorted_tokens[:train_end], sorted_tokens[train_end:val_end], sorted_tokens[val_end:]


# =============================================================================
# Feature Extraction
# =============================================================================

def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI."""
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
    """Calculate MACD."""
    if len(prices) < 2:
        return np.zeros(len(prices))

    ema_fast = pd.Series(prices).ewm(span=fast, adjust=False).mean().values
    ema_slow = pd.Series(prices).ewm(span=slow, adjust=False).mean().values

    return ema_fast - ema_slow


def calculate_bollinger(prices: np.ndarray, period: int = 20, num_std: float = 2.0):
    """Calculate Bollinger Bands."""
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
    """Calculate VWAP."""
    typical_prices = np.array([(c.high + c.low + c.close) / 3 for c in candles])
    volumes = np.array([c.volume for c in candles])

    cumulative_tp_vol = np.cumsum(typical_prices * volumes)
    cumulative_vol = np.cumsum(volumes)

    return np.divide(cumulative_tp_vol, cumulative_vol, where=cumulative_vol != 0, out=typical_prices.copy())


def extract_screener_features(token: TokenData, decision_time: int = 30) -> Optional[np.ndarray]:
    """
    Extract features for Model 1 (Screener).

    Features designed to capture early momentum and volume patterns
    that predict future price appreciation.
    """
    candles = token.candles[:decision_time]
    if len(candles) < 10:
        return None

    prices = np.array([c.close for c in candles])
    highs = np.array([c.high for c in candles])
    lows = np.array([c.low for c in candles])
    volumes = np.array([c.volume for c in candles])

    first_price = prices[0]
    current_price = prices[-1]

    if first_price <= 0 or current_price <= 0:
        return None

    # === PRICE MOMENTUM FEATURES ===
    # Returns at different timeframes
    def safe_pct_change(arr, periods):
        if len(arr) <= periods:
            return 0.0
        old_val = arr[-periods-1]
        new_val = arr[-1]
        return (new_val - old_val) / (old_val + 1e-10)

    return_5s = safe_pct_change(prices, 5)
    return_10s = safe_pct_change(prices, 10)
    return_15s = safe_pct_change(prices, 15)
    return_total = (current_price - first_price) / first_price

    # Price acceleration (2nd derivative)
    if len(prices) >= 15:
        mid = len(prices) // 2
        first_half_return = (prices[mid] - prices[0]) / (prices[0] + 1e-10)
        second_half_return = (prices[-1] - prices[mid]) / (prices[mid] + 1e-10)
        price_acceleration = second_half_return - first_half_return
    else:
        price_acceleration = 0.0

    # === VOLATILITY FEATURES ===
    # High-low range relative to price
    range_pcts = (highs - lows) / (prices + 1e-10)
    avg_range = np.mean(range_pcts)
    max_range = np.max(range_pcts)

    # Price volatility (std of returns)
    if len(prices) > 1:
        log_returns = np.diff(np.log(np.clip(prices, 1e-10, None)))
        volatility = np.std(log_returns) if len(log_returns) > 0 else 0.0
    else:
        volatility = 0.0

    # === VOLUME FEATURES ===
    total_volume = np.sum(volumes)
    avg_volume = np.mean(volumes)

    # Volume trend (comparing halves)
    if len(volumes) >= 10:
        first_half_vol = np.mean(volumes[:len(volumes)//2])
        second_half_vol = np.mean(volumes[len(volumes)//2:])
        volume_trend = (second_half_vol - first_half_vol) / (first_half_vol + 1e-10)
        volume_surge = second_half_vol / (first_half_vol + 1e-10)
    else:
        volume_trend = 0.0
        volume_surge = 1.0

    # Recent volume spike
    if len(volumes) >= 5:
        recent_vol = np.mean(volumes[-5:])
        early_vol = np.mean(volumes[:-5]) if len(volumes) > 5 else recent_vol
        volume_spike = recent_vol / (early_vol + 1e-10)
    else:
        volume_spike = 1.0

    # Volume consistency (low std = consistent buying)
    volume_consistency = np.std(volumes) / (np.mean(volumes) + 1e-10)

    # === ORDER FLOW FEATURES ===
    # Green vs red candles
    green_candles = np.sum(prices[1:] > prices[:-1]) if len(prices) > 1 else 0
    red_candles = np.sum(prices[1:] < prices[:-1]) if len(prices) > 1 else 0
    green_ratio = green_candles / (green_candles + red_candles + 1e-10)

    # Buy pressure (close near high vs low)
    buy_pressure = np.mean((prices - lows) / (highs - lows + 1e-10))

    # Consecutive green candles at end
    consecutive_green = 0
    for i in range(len(prices) - 1, 0, -1):
        if prices[i] > prices[i-1]:
            consecutive_green += 1
        else:
            break
    consecutive_green_norm = consecutive_green / len(prices)

    # === TREND STRENGTH ===
    # Linear regression slope
    if len(prices) >= 5:
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        trend_strength = slope / (first_price + 1e-10)
    else:
        trend_strength = 0.0

    # Distance from high
    period_high = np.max(highs)
    dist_from_high = (current_price - period_high) / (period_high + 1e-10)

    # === RELATIVE FEATURES ===
    # Current price position in range
    period_low = np.min(lows)
    price_position = (current_price - period_low) / (period_high - period_low + 1e-10)

    # === ASSEMBLE FEATURES ===
    features = np.array([
        # Price momentum (5)
        return_5s,
        return_10s,
        return_15s,
        return_total,
        price_acceleration,
        # Volatility (3)
        avg_range,
        max_range,
        volatility,
        # Volume (5)
        np.log1p(total_volume),
        volume_trend,
        volume_surge,
        volume_spike,
        volume_consistency,
        # Order flow (4)
        green_ratio,
        buy_pressure,
        consecutive_green_norm,
        trend_strength,
        # Price position (2)
        dist_from_high,
        price_position,
    ], dtype=np.float32)

    # Clip extreme values and handle NaN/inf
    features = np.clip(features, -5.0, 5.0)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


# Feature names for screener (must match extract_screener_features output)
SCREENER_FEATURE_NAMES = [
    "return_5s", "return_10s", "return_15s", "return_total", "price_acceleration",
    "avg_range", "max_range", "volatility",
    "volume_log", "volume_trend", "volume_surge", "volume_spike", "volume_consistency",
    "green_ratio", "buy_pressure", "consecutive_green", "trend_strength",
    "dist_from_high", "price_position",
]


def extract_timeseries_features(candles: List[Candle]) -> np.ndarray:
    """Extract time-series features for Models 2 & 3."""
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


# =============================================================================
# Label Generation
# =============================================================================

def get_execution_price(candles: List[Candle], start_idx: int, is_buy: bool = True) -> float:
    """Get execution price with delay."""
    end_idx = min(start_idx + DELAY_SECONDS + 1, len(candles))
    delay_window = candles[start_idx:end_idx]

    if not delay_window:
        return candles[start_idx].close

    if is_buy:
        return max(c.high for c in delay_window)
    else:
        return min(c.low for c in delay_window)


def generate_screener_label(token: TokenData, decision_time: int = 30) -> Optional[int]:
    """Generate screener label."""
    if len(token.candles) <= decision_time:
        return None

    entry_price = get_execution_price(token.candles, decision_time, is_buy=True)

    future_start = decision_time + DELAY_SECONDS
    future_end = min(future_start + SCREENER_LOOKAHEAD_SEC, len(token.candles))
    future_candles = token.candles[future_start:future_end]

    if not future_candles:
        return None

    max_future_price = max(c.high for c in future_candles)
    peak_ratio = max_future_price / entry_price if entry_price > 0 else 1.0

    return ScreenerLabel.WORTHY if peak_ratio >= SCREENER_WORTHY_THRESHOLD else ScreenerLabel.AVOID


def generate_entry_label(candles: List[Candle], current_idx: int) -> Optional[int]:
    """
    Generate entry label using forward-looking risk-reward analysis.

    All labels are based on future outcomes for consistency:
    - ENTER_NOW: Good risk-reward ratio (profit > loss, gain achievable)
    - ABORT: Poor future outcome (significant loss ahead)
    - WAIT: Unclear signal (neither good nor bad)
    """
    if current_idx >= len(candles) - ENTRY_LOOKAHEAD_SEC - DELAY_SECONDS:
        return None

    entry_price = get_execution_price(candles, current_idx, is_buy=True)
    if entry_price <= 0:
        return None

    future_start = current_idx + DELAY_SECONDS
    future_end = min(future_start + ENTRY_LOOKAHEAD_SEC, len(candles))
    future_candles = candles[future_start:future_end]

    if len(future_candles) < 10:
        return None

    # Calculate future price extremes
    future_highs = [c.high for c in future_candles]
    future_lows = [c.low for c in future_candles]
    future_closes = [c.close for c in future_candles]

    max_future_high = max(future_highs)
    min_future_low = min(future_lows)
    final_close = future_closes[-1]

    # Calculate gains and losses
    max_gain = (max_future_high - entry_price) / entry_price
    max_drawdown = (min_future_low - entry_price) / entry_price
    final_return = (final_close - entry_price) / entry_price

    # Find when max gain and max drawdown occur
    time_to_high = future_highs.index(max_future_high)
    time_to_low = future_lows.index(min_future_low)

    # Risk-reward ratio (only if there's potential gain)
    if max_gain > 0.01:
        risk_reward = max_gain / (abs(max_drawdown) + 0.01)
    else:
        risk_reward = 0.0

    # === LABELING LOGIC ===

    # ENTER_NOW conditions (relaxed for more signals):
    # 1. Potential gain exceeds threshold
    # 2. Max drawdown is acceptable
    # 3. Reasonable risk-reward ratio (> 1.0)
    is_profitable = max_gain >= ENTRY_PROFIT_THRESHOLD
    is_safe = max_drawdown > -ENTRY_MAX_DRAWDOWN
    decent_risk_reward = risk_reward >= 1.0

    if is_profitable and is_safe and decent_risk_reward:
        return EntryLabel.ENTER_NOW

    # ABORT conditions (clear danger signals):
    # 1. Severe drawdown ahead (> 25%)
    # 2. Price ends significantly lower than entry
    # 3. Immediate dump (drawdown early and significant)
    severe_drawdown = max_drawdown < -0.25
    ends_badly = final_return < -0.20
    early_dump = time_to_low < 15 and max_drawdown < -0.15

    if severe_drawdown or ends_badly or early_dump:
        return EntryLabel.ABORT

    # WAIT: Signal is unclear
    return EntryLabel.WAIT


def generate_exit_label(
    candles: List[Candle],
    entry_idx: int,
    entry_price: float,
    current_idx: int,
) -> int:
    """Generate exit label."""
    if current_idx >= len(candles):
        return ExitLabel.EXIT_NOW

    current_price = candles[current_idx].close
    unrealized_pnl = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0

    position_candles = candles[entry_idx:current_idx + 1]
    position_high = max(c.high for c in position_candles) if position_candles else entry_price
    drawdown_from_high = (current_price - position_high) / position_high if position_high > 0 else 0.0

    time_in_position = current_idx - entry_idx

    # Hard exit conditions
    if unrealized_pnl <= -EXIT_STOP_LOSS_PCT:
        return ExitLabel.EXIT_NOW
    if drawdown_from_high <= -EXIT_TRAILING_STOP_PCT:
        return ExitLabel.EXIT_NOW
    if unrealized_pnl >= EXIT_PROFIT_TARGET_PCT:
        return ExitLabel.EXIT_NOW
    if time_in_position >= 300 and unrealized_pnl < 1.0:
        return ExitLabel.EXIT_NOW

    # Partial exit
    if unrealized_pnl >= 0.30:
        if len(position_candles) >= 5:
            recent_momentum = (position_candles[-1].close - position_candles[-5].close) / position_candles[-5].close
            if recent_momentum < 0.05:
                return ExitLabel.PARTIAL_EXIT

    return ExitLabel.HOLD


# =============================================================================
# Dataset Generation
# =============================================================================

def process_screener_data(tokens: List[TokenData], decision_time: int = 30):
    """Process tokens for Model 1."""
    features_list = []
    labels_list = []
    metadata_list = []

    for token in tokens:
        features = extract_screener_features(token, decision_time)
        if features is None:
            continue

        label = generate_screener_label(token, decision_time)
        if label is None:
            continue

        features_list.append(features)
        labels_list.append(label)
        metadata_list.append({
            "token_address": token.token_address,
            "symbol": token.symbol,
            "peak_ratio": token.peak_ratio,
        })

    return np.array(features_list), np.array(labels_list), metadata_list


def process_entry_data(tokens: List[TokenData], start_time: int = 30, sample_interval: int = 5):
    """Process tokens for Model 2 with stratified sampling."""
    all_samples = []

    for token in tokens:
        candles = token.candles
        token_samples = []

        for t in range(start_time, len(candles) - ENTRY_LOOKAHEAD_SEC, sample_interval):
            current_candles = candles[:t + 1]
            features = extract_timeseries_features(current_candles)

            if len(features) < 10:
                continue

            label = generate_entry_label(candles, t)
            if label is None:
                continue

            token_samples.append({
                "features": features,
                "label": label,
                "timestamp": t,
                "token_address": token.token_address,
            })

            # Limit samples per token to reduce correlation
            if len(token_samples) >= ENTRY_SAMPLES_PER_TOKEN:
                break

        # Add all samples from this token
        all_samples.extend(token_samples)

    return all_samples


def process_exit_data(tokens: List[TokenData], sample_interval: int = 5):
    """Process tokens for Model 3."""
    all_samples = []

    for token in tokens:
        candles = token.candles
        if len(candles) < 40:
            continue

        entry_time = 30
        entry_price = candles[entry_time].close

        token_samples = []
        for t in range(entry_time + 1, len(candles), sample_interval):
            # Extract exit features (base + position-aware)
            current_candles = candles[:t + 1]
            base_features = extract_timeseries_features(current_candles)

            if len(base_features) == 0:
                continue

            current_price = candles[t].close
            position_candles = candles[entry_time:t + 1]
            position_high = max(c.high for c in position_candles) if position_candles else entry_price

            unrealized_pnl = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0
            time_since_entry = (t - entry_time) / 300.0
            drawdown_from_position_high = (current_price - position_high) / position_high if position_high > 0 else 0.0

            exit_features = np.concatenate([
                base_features[-1],
                np.array([unrealized_pnl, time_since_entry, drawdown_from_position_high], dtype=np.float32),
            ])

            label = generate_exit_label(candles, entry_time, entry_price, t)

            token_samples.append({
                "features": exit_features,
                "label": label,
                "timestamp": t,
                "token_address": token.token_address,
            })

            if label == ExitLabel.EXIT_NOW:
                break

            if len(token_samples) >= 30:
                break

        all_samples.extend(token_samples)

    return all_samples


# =============================================================================
# Main Preprocessing
# =============================================================================

def preprocess_all(
    csv_path: str,
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
):
    """Complete preprocessing pipeline."""
    start_time = time.time()
    np.random.seed(random_seed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DATA PREPROCESSING FOR MULTI-MODEL ARCHITECTURE")
    print("=" * 70)

    # Load data
    print("\n[1/6] Loading raw data...")
    tokens = load_raw_data(csv_path)
    print(f"  Loaded {len(tokens)} tokens")

    # Split
    print("\n[2/6] Splitting data chronologically...")
    train_tokens, val_tokens, test_tokens = split_tokens_chronological(
        tokens, train_ratio, val_ratio, test_ratio
    )
    print(f"  Train: {len(train_tokens)}, Val: {len(val_tokens)}, Test: {len(test_tokens)}")

    # Screener data
    print("\n[3/6] Processing screener data (Model 1)...")
    train_s_feat, train_s_lab, train_s_meta = process_screener_data(train_tokens)
    val_s_feat, val_s_lab, val_s_meta = process_screener_data(val_tokens)
    test_s_feat, test_s_lab, test_s_meta = process_screener_data(test_tokens)

    # Use the correct feature names matching extract_screener_features output
    with open(output_path / "screener_train.pkl", "wb") as f:
        pickle.dump({"X": train_s_feat, "y": train_s_lab, "feature_names": SCREENER_FEATURE_NAMES, "metadata": train_s_meta}, f)
    with open(output_path / "screener_val.pkl", "wb") as f:
        pickle.dump({"X": val_s_feat, "y": val_s_lab, "feature_names": SCREENER_FEATURE_NAMES, "metadata": val_s_meta}, f)
    with open(output_path / "screener_test.pkl", "wb") as f:
        pickle.dump({"X": test_s_feat, "y": test_s_lab, "feature_names": SCREENER_FEATURE_NAMES, "metadata": test_s_meta}, f)

    print(f"  Train: {len(train_s_lab)}, Val: {len(val_s_lab)}, Test: {len(test_s_lab)}")
    worthy_pct = np.sum(train_s_lab == 1) / len(train_s_lab) * 100
    print(f"  Class dist (train): AVOID={100-worthy_pct:.1f}%, WORTHY={worthy_pct:.1f}%")

    # Entry data
    print("\n[4/6] Processing entry data (Model 2)...")
    train_entry = process_entry_data(train_tokens)
    val_entry = process_entry_data(val_tokens)
    test_entry = process_entry_data(test_tokens)

    # Convert entry data to arrays (avoid pickling issues with custom classes)
    def convert_entry_to_arrays(samples):
        if not samples:
            return {"sequences": [], "labels": np.array([], dtype=np.int64), "timestamps": [], "token_addresses": []}
        sequences = [s["features"] for s in samples]
        labels = np.array([int(s["label"]) for s in samples], dtype=np.int64)
        timestamps = [s["timestamp"] for s in samples]
        token_addresses = [s["token_address"] for s in samples]
        return {"sequences": sequences, "labels": labels, "timestamps": timestamps, "token_addresses": token_addresses}

    with open(output_path / "entry_train.pkl", "wb") as f:
        pickle.dump(convert_entry_to_arrays(train_entry), f)
    with open(output_path / "entry_val.pkl", "wb") as f:
        pickle.dump(convert_entry_to_arrays(val_entry), f)
    with open(output_path / "entry_test.pkl", "wb") as f:
        pickle.dump(convert_entry_to_arrays(test_entry), f)

    print(f"  Train: {len(train_entry)}, Val: {len(val_entry)}, Test: {len(test_entry)}")

    # Exit data
    print("\n[5/6] Processing exit data (Model 3)...")
    train_exit = process_exit_data(train_tokens)
    val_exit = process_exit_data(val_tokens)
    test_exit = process_exit_data(test_tokens)

    # Convert exit data to arrays (avoid pickling issues with custom classes)
    def convert_exit_to_arrays(samples):
        if not samples:
            return {"X": np.array([]), "y": np.array([], dtype=np.int64), "timestamps": [], "token_addresses": []}
        X = np.array([s["features"] for s in samples], dtype=np.float32)
        y = np.array([int(s["label"]) for s in samples], dtype=np.int64)
        timestamps = [s["timestamp"] for s in samples]
        token_addresses = [s["token_address"] for s in samples]
        return {"X": X, "y": y, "timestamps": timestamps, "token_addresses": token_addresses}

    with open(output_path / "exit_train.pkl", "wb") as f:
        pickle.dump(convert_exit_to_arrays(train_exit), f)
    with open(output_path / "exit_val.pkl", "wb") as f:
        pickle.dump(convert_exit_to_arrays(val_exit), f)
    with open(output_path / "exit_test.pkl", "wb") as f:
        pickle.dump(convert_exit_to_arrays(test_exit), f)

    print(f"  Train: {len(train_exit)}, Val: {len(val_exit)}, Test: {len(test_exit)}")

    # Metadata
    print("\n[6/6] Saving metadata...")
    metadata = {
        "csv_path": str(csv_path),
        "random_seed": random_seed,
        "token_counts": {"total": len(tokens), "train": len(train_tokens), "val": len(val_tokens), "test": len(test_tokens)},
        "screener": {"train": len(train_s_lab), "val": len(val_s_lab), "test": len(test_s_lab)},
        "entry": {"train": len(train_entry), "val": len(val_entry), "test": len(test_entry)},
        "exit": {"train": len(train_exit), "val": len(val_exit), "test": len(test_exit)},
        "processing_time_seconds": time.time() - start_time,
    }

    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"Output: {output_path}")
    print(f"Time: {metadata['processing_time_seconds']:.1f}s")
    print("=" * 70)

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=str, default="../../ai_data/data/combined_tokens.csv")
    parser.add_argument("--output-dir", type=str, default="../data/processed")
    args = parser.parse_args()

    preprocess_all(args.csv_path, args.output_dir)
