"""
Label generation functions for Multi-Model Trading Architecture.

Implements label generation for training each model:
    - Model 1 (Screener): WORTHY / AVOID binary classification
    - Model 2 (Entry): ENTER_NOW / WAIT / ABORT classification
    - Model 3 (Exit): EXIT_NOW / HOLD / PARTIAL_EXIT classification

Author: Trading Team
Date: 2025-12-29
"""

from typing import List, Dict, Optional, Tuple

import numpy as np

from .loader import TokenData, Candle
from .label_utils import (
    ScreenerLabel,
    EntryLabel,
    ExitLabel,
    get_execution_price,
)
from ..config import (
    SCREENER_WORTHY_THRESHOLD,
    SCREENER_LOOKAHEAD_SEC,
    ENTRY_PROFIT_THRESHOLD,
    ENTRY_MAX_DRAWDOWN,
    ENTRY_LOOKAHEAD_SEC,
    EXIT_TRAILING_STOP_PCT,
    EXIT_STOP_LOSS_PCT,
    EXIT_PROFIT_TARGET_PCT,
    DELAY_SECONDS,
)


def generate_screener_labels(
    token: TokenData,
    decision_time: int = 30,
    worthy_threshold: float = SCREENER_WORTHY_THRESHOLD,
    lookahead_sec: int = SCREENER_LOOKAHEAD_SEC,
) -> Optional[int]:
    """
    Generate WORTHY/AVOID label for Model 1 (Screener).

    WORTHY if peak price >= worthy_threshold within lookahead window.
    Returns None if insufficient data.
    """
    candles = token.candles

    if len(candles) <= decision_time:
        return None

    entry_price = get_execution_price(candles, decision_time, is_buy=True)

    future_start = decision_time + DELAY_SECONDS
    future_end = min(future_start + lookahead_sec, len(candles))
    future_candles = candles[future_start:future_end]

    if not future_candles:
        return None

    max_future_price = max(c.high for c in future_candles)
    peak_ratio = max_future_price / entry_price if entry_price > 0 else 1.0

    if peak_ratio >= worthy_threshold:
        return ScreenerLabel.WORTHY
    else:
        return ScreenerLabel.AVOID


def generate_screener_dataset(
    tokens: List[TokenData],
    decision_time: int = 30,
) -> Tuple[List[np.ndarray], List[int], List[Dict]]:
    """Generate complete screener dataset. Returns (features, labels, metadata)."""
    from .feature_extractors import extract_screener_features

    features_list = []
    labels_list = []
    metadata_list = []

    for token in tokens:
        features = extract_screener_features(token, decision_time)
        if features is None:
            continue

        label = generate_screener_labels(token, decision_time)
        if label is None:
            continue

        features_list.append(features)
        labels_list.append(label)
        metadata_list.append({
            "token_address": token.token_address,
            "symbol": token.symbol,
            "peak_ratio": token.peak_ratio,
            "death_reason": token.death_reason,
        })

    return features_list, labels_list, metadata_list


def generate_entry_labels(
    candles: List[Candle],
    current_idx: int,
    profit_threshold: float = ENTRY_PROFIT_THRESHOLD,
    max_drawdown: float = ENTRY_MAX_DRAWDOWN,
    lookahead_sec: int = ENTRY_LOOKAHEAD_SEC,
) -> Optional[int]:
    """
    Generate ENTER_NOW/WAIT/ABORT label for Model 2 (Entry).

    Returns None if insufficient lookahead data.
    """
    if current_idx >= len(candles) - lookahead_sec - DELAY_SECONDS:
        return None

    entry_price = get_execution_price(candles, current_idx, is_buy=True)

    future_start = current_idx + DELAY_SECONDS
    future_end = min(future_start + lookahead_sec, len(candles))
    future_candles = candles[future_start:future_end]

    if not future_candles:
        return None

    max_future_high = max(c.high for c in future_candles)
    min_future_low = min(c.low for c in future_candles)

    max_gain = (max_future_high - entry_price) / entry_price if entry_price > 0 else 0.0
    max_loss = (min_future_low - entry_price) / entry_price if entry_price > 0 else 0.0

    lookback = min(30, current_idx)
    recent_candles = candles[current_idx - lookback:current_idx + 1]
    recent_high = max(c.high for c in recent_candles) if recent_candles else entry_price
    current_price = candles[current_idx].close

    dist_from_peak = (current_price - recent_high) / recent_high if recent_high > 0 else 0.0

    if len(recent_candles) >= 10:
        vol_first = np.mean([c.volume for c in recent_candles[:len(recent_candles)//2]])
        vol_second = np.mean([c.volume for c in recent_candles[len(recent_candles)//2:]])
        volume_declining = vol_second < vol_first * 0.5
    else:
        volume_declining = False

    significant_drop = dist_from_peak < -0.20

    if max_gain >= profit_threshold and max_loss > -max_drawdown:
        return EntryLabel.ENTER_NOW
    elif significant_drop or volume_declining:
        return EntryLabel.ABORT
    else:
        return EntryLabel.WAIT


def generate_entry_samples(
    token: TokenData,
    start_time: int = 30,
    sample_interval: int = 5,
) -> List[Dict]:
    """Generate entry timing samples for a single token."""
    from .feature_extractors import extract_timeseries_features

    samples = []
    candles = token.candles

    for t in range(start_time, len(candles) - ENTRY_LOOKAHEAD_SEC, sample_interval):
        current_candles = candles[:t + 1]
        features = extract_timeseries_features(current_candles)

        if len(features) < 10:
            continue

        label = generate_entry_labels(candles, t)
        if label is None:
            continue

        samples.append({
            "features": features,
            "label": label,
            "timestamp": t,
            "token_address": token.token_address,
        })

    return samples


def generate_exit_labels(
    candles: List[Candle],
    entry_idx: int,
    entry_price: float,
    current_idx: int,
    trailing_stop: float = EXIT_TRAILING_STOP_PCT,
    stop_loss: float = EXIT_STOP_LOSS_PCT,
    profit_target: float = EXIT_PROFIT_TARGET_PCT,
) -> int:
    """Generate EXIT_NOW/HOLD/PARTIAL_EXIT label for Model 3 (Exit)."""
    if current_idx >= len(candles):
        return ExitLabel.EXIT_NOW

    current_price = candles[current_idx].close
    unrealized_pnl = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0

    position_candles = candles[entry_idx:current_idx + 1]
    position_high = max(c.high for c in position_candles) if position_candles else entry_price
    drawdown_from_high = (current_price - position_high) / position_high if position_high > 0 else 0.0

    time_in_position = current_idx - entry_idx

    # Hard exit conditions
    if unrealized_pnl <= -stop_loss:
        return ExitLabel.EXIT_NOW
    if drawdown_from_high <= -trailing_stop:
        return ExitLabel.EXIT_NOW
    if unrealized_pnl >= profit_target:
        return ExitLabel.EXIT_NOW
    if time_in_position >= 300 and unrealized_pnl < 1.0:
        return ExitLabel.EXIT_NOW

    # Partial exit conditions
    if unrealized_pnl >= 0.30:
        if len(position_candles) >= 5:
            recent_momentum = (
                (position_candles[-1].close - position_candles[-5].close)
                / position_candles[-5].close
            )
            if recent_momentum < 0.05:
                return ExitLabel.PARTIAL_EXIT

    # Volume exhaustion check
    if len(position_candles) >= 10:
        vol_first = np.mean([c.volume for c in position_candles[:5]])
        vol_last = np.mean([c.volume for c in position_candles[-5:]])

        if vol_last < vol_first * 0.3 and unrealized_pnl > 0.20:
            return ExitLabel.EXIT_NOW

    return ExitLabel.HOLD


def generate_exit_samples(
    token: TokenData,
    entry_time: int,
    entry_price: float,
    sample_interval: int = 5,
) -> List[Dict]:
    """Generate exit timing samples for a simulated trade."""
    from .feature_extractors import extract_exit_features

    samples = []
    candles = token.candles

    for t in range(entry_time + 1, len(candles), sample_interval):
        features = extract_exit_features(candles, entry_time, entry_price, t)

        if features is None:
            continue

        label = generate_exit_labels(candles, entry_time, entry_price, t)

        samples.append({
            "features": features,
            "label": label,
            "timestamp": t,
            "entry_time": entry_time,
            "entry_price": entry_price,
            "current_price": candles[t].close,
            "unrealized_pnl": (candles[t].close - entry_price) / entry_price,
            "token_address": token.token_address,
        })

        if label == ExitLabel.EXIT_NOW:
            break

    return samples
