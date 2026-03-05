"""
Data preprocessing and feature extraction for v1 trading model.

Implements data loading, feature extraction, and training sample generation
with full historical context for simulating real market observation.

Dependencies: numpy, pandas
Author: Trading Team
Date: 2025-12-21
"""

import json
from typing import List, Dict, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    FIXED_POSITION_SIZE,
    DELAY_SECONDS,
    TOTAL_FEE_PER_TX,
    MIN_HISTORY_LENGTH,
    LOOKAHEAD_SECONDS,
    TAKE_PROFIT_PCT,
    STOP_LOSS_PCT,
    TRAIL_BACKOFF_PCT,
)

# Shared technical indicators (same numpy-array interface as v2)
from .technical_indicators import calculate_rsi, calculate_macd


def load_raw_data(csv_path: str) -> pd.DataFrame:
    """Load raw token data from CSV into a DataFrame.

    Raises FileNotFoundError or ValueError on invalid input.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path, quotechar='"', escapechar='\\',
                     on_bad_lines='warn', engine='python')

    if df.empty:
        raise ValueError("CSV file is empty")

    if 'token_address' not in df.columns:
        raise ValueError("Missing required column: token_address")

    if 'chart_data_json' in df.columns and 'candles' not in df.columns:
        df = df.rename(columns={'chart_data_json': 'candles'})

    if 'candles' not in df.columns:
        raise ValueError("Missing required column: candles or chart_data_json")

    return df


def parse_candles(candles_json: str) -> List[Dict[str, float]]:
    """Parse candles JSON string into list of dicts with keys t,o,h,l,c,v."""
    data = json.loads(candles_json)

    if isinstance(data, dict):
        if 'data' in data and isinstance(data['data'], dict) and 'list' in data['data']:
            raw_candles = data['data']['list']
        elif 'list' in data:
            raw_candles = data['list']
        else:
            raise ValueError(f"Cannot extract candles from dict with keys: {data.keys()}")
    elif isinstance(data, list):
        raw_candles = data
    else:
        raise ValueError(f"Unknown candle format: {type(data)}")

    if not raw_candles:
        return []

    candles = []
    for c in raw_candles:
        try:
            candle = {
                't': int(c.get('time', c.get('t', 0))),
                'o': float(c.get('open', c.get('o', 0))),
                'h': float(c.get('high', c.get('h', 0))),
                'l': float(c.get('low', c.get('l', 0))),
                'c': float(c.get('close', c.get('c', 0))),
                'v': float(c.get('volume', c.get('v', 0))),
            }
            if candle['c'] > 0 and candle['o'] > 0:
                candles.append(candle)
        except (ValueError, TypeError):
            continue

    return candles


def calculate_bollinger_bands(prices: np.ndarray, period: int = 20,
                              num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Bollinger Bands. Returns (upper, lower) for v1 compatibility."""
    if len(prices) < 2:
        return np.full(len(prices), prices[0]), np.full(len(prices), prices[0])

    sma = pd.Series(prices).rolling(window=period, min_periods=1).mean().values
    std = pd.Series(prices).rolling(window=period, min_periods=1).std().values
    return sma + (num_std * std), sma - (num_std * std)


def calculate_vwap(candles: List[Dict[str, float]]) -> np.ndarray:
    """Calculate VWAP from dict-based candles (v1 interface)."""
    typical_prices = np.array([(c['h'] + c['l'] + c['c']) / 3 for c in candles])
    volumes = np.array([c['v'] for c in candles])
    cumulative_tp_vol = np.cumsum(typical_prices * volumes)
    cumulative_vol = np.cumsum(volumes)
    return np.divide(cumulative_tp_vol, cumulative_vol,
                     where=cumulative_vol != 0, out=typical_prices.copy())


def calculate_momentum(prices: np.ndarray, period: int = 10) -> np.ndarray:
    """Calculate price momentum (rate of change over lookback period)."""
    momentum = np.zeros(len(prices))
    for i in range(len(prices)):
        lookback_idx = max(0, i - period)
        if prices[lookback_idx] != 0:
            momentum[i] = (prices[i] - prices[lookback_idx]) / prices[lookback_idx]
    return momentum


def _log_return(series: np.ndarray, lag: int) -> np.ndarray:
    """Compute log returns with a given lag, padding the front with zeros."""
    safe = np.clip(series, 1e-8, None)
    logs = np.log(safe)
    if len(series) > lag:
        shifted = np.concatenate([np.full(lag, logs[0]), logs[:-lag]])
    else:
        shifted = np.full(len(series), logs[0])
    return logs - shifted


def extract_features(candles: List[Dict[str, float]]) -> np.ndarray:
    """Extract 14 position-agnostic features for each timestep.

    Features: log_close, ret_1s, ret_3s, ret_5s, range_ratio, volume_log,
    rsi_norm, macd_norm, bb_upper_dev, bb_lower_dev, vwap_dev, momentum_10,
    indicator_ready_short, indicator_ready_long.
    """
    closes = np.array([c['c'] for c in candles], dtype=np.float32)
    highs = np.array([c['h'] for c in candles], dtype=np.float32)
    lows = np.array([c['l'] for c in candles], dtype=np.float32)
    volumes = np.array([c['v'] for c in candles], dtype=np.float32)

    rsi = calculate_rsi(closes)
    macd = calculate_macd(closes)
    bb_upper, bb_lower = calculate_bollinger_bands(closes)
    vwap = calculate_vwap(candles)
    momentum = calculate_momentum(closes)

    log_close = np.log(np.clip(closes, 1e-8, None)) - np.log(np.clip(closes[0], 1e-8, None))
    ret_1 = _log_return(closes, 1)
    ret_3 = _log_return(closes, 3)
    ret_5 = _log_return(closes, 5)
    range_ratio = np.where(closes != 0, (highs - lows) / closes, 0.0)
    volume_log = np.log1p(np.clip(volumes, 0, None))
    macd_norm = np.where(closes != 0, macd / closes, 0.0)
    bb_upper_dev = np.where(closes != 0, (bb_upper - closes) / closes, 0.0)
    bb_lower_dev = np.where(closes != 0, (bb_lower - closes) / closes, 0.0)
    vwap_dev = np.where(closes != 0, (vwap - closes) / closes, 0.0)

    idx = np.arange(len(closes))
    features = np.column_stack([
        log_close, ret_1, ret_3, ret_5, range_ratio, volume_log,
        rsi / 100.0, macd_norm, bb_upper_dev, bb_lower_dev,
        vwap_dev, momentum,
        (idx >= 5).astype(np.float32), (idx >= 20).astype(np.float32),
    ])

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(features, -10.0, 10.0).astype(np.float32)


def get_execution_price(candles: List[Dict[str, float]],
                       start_idx: int, is_buy: bool = True) -> float:
    """Simulate execution price with realistic delay and worst-case slippage."""
    end_idx = min(start_idx + DELAY_SECONDS + 1, len(candles))
    delay_window = candles[start_idx:end_idx]
    if not delay_window:
        return candles[start_idx]['c']
    return max(c['h'] for c in delay_window) if is_buy else min(c['l'] for c in delay_window)


def calculate_net_profit(buy_price: float, sell_price: float) -> float:
    """Calculate net profit in SOL after transaction fees."""
    tokens = FIXED_POSITION_SIZE / buy_price
    sell_value = tokens * sell_price
    return sell_value - (2 * TOTAL_FEE_PER_TX) - FIXED_POSITION_SIZE


def prepare_realistic_training_data(token_candles: List[Dict[str, float]],
                                    min_history: int = MIN_HISTORY_LENGTH
                                    ) -> List[Dict[str, Any]]:
    """Prepare position-agnostic training samples with BUY/SELL/HOLD labels.

    Labels based on future price action:
      BUY (1): Strong bullish - good profit with acceptable risk
      SELL (2): Strong bearish - danger (drawdown or rollover)
      HOLD (0): Unclear - wait for better signal
    """
    if not token_candles:
        raise ValueError("token_candles cannot be empty")
    if len(token_candles) < min_history:
        return []

    samples: List[Dict[str, Any]] = []

    for current_time in range(min_history, len(token_candles)):
        features = extract_features(token_candles[0:current_time])
        buy_price = get_execution_price(token_candles, current_time, is_buy=True)

        future_start = current_time + DELAY_SECONDS
        future_end = min(future_start + LOOKAHEAD_SECONDS, len(token_candles))
        future_candles = token_candles[future_start:future_end]
        if not future_candles:
            continue

        max_high = max(c['h'] for c in future_candles)
        min_low = min(c['l'] for c in future_candles)
        end_close = future_candles[-1]['c']

        profit_pct = calculate_net_profit(buy_price, max_high) / FIXED_POSITION_SIZE
        drawdown_pct = (min_low - buy_price) / buy_price
        peak_gain_pct = (max_high - buy_price) / buy_price

        rolled_over = (peak_gain_pct >= TAKE_PROFIT_PCT
                       and end_close <= max_high * (1 - TRAIL_BACKOFF_PCT))

        if profit_pct >= TAKE_PROFIT_PCT and drawdown_pct >= STOP_LOSS_PCT:
            label = 1  # BUY
        elif drawdown_pct <= STOP_LOSS_PCT or rolled_over:
            label = 2  # SELL
        else:
            label = 0  # HOLD

        samples.append({
            'features': features, 'label': label,
            'seq_length': current_time, 'timestamp': current_time,
            'buy_price': buy_price, 'potential_profit_pct': profit_pct,
            'drawdown_pct': drawdown_pct,
        })

    return samples
