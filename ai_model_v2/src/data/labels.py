"""
Label generation for Multi-Model Trading Architecture.

This module implements label generation for training each model:
    - Model 1 (Screener): WORTHY / AVOID binary classification
    - Model 2 (Entry): ENTER_NOW / WAIT / ABORT classification
    - Model 3 (Exit): EXIT_NOW / HOLD / PARTIAL_EXIT classification

Author: Trading Team
Date: 2025-12-29
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Dict, Optional, Tuple

import numpy as np

from .loader import TokenData, Candle
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
    TOTAL_FEE_PER_TX,
    FIXED_POSITION_SIZE_SOL,
)


# =============================================================================
# Label Enums
# =============================================================================

class ScreenerLabel(IntEnum):
    """Labels for Model 1 (Screener)."""
    AVOID = 0
    WORTHY = 1


class EntryLabel(IntEnum):
    """Labels for Model 2 (Entry)."""
    WAIT = 0
    ENTER_NOW = 1
    ABORT = 2


class ExitLabel(IntEnum):
    """Labels for Model 3 (Exit)."""
    HOLD = 0
    EXIT_NOW = 1
    PARTIAL_EXIT = 2


# =============================================================================
# Utility Functions
# =============================================================================

def get_execution_price(
    candles: List[Candle], start_idx: int, is_buy: bool = True
) -> float:
    """
    Simulate execution price with realistic delay and slippage.

    Args:
        candles: List of candles.
        start_idx: Index where order is placed.
        is_buy: True for buy (use high), False for sell (use low).

    Returns:
        Worst-case execution price.
    """
    end_idx = min(start_idx + DELAY_SECONDS + 1, len(candles))
    delay_window = candles[start_idx:end_idx]

    if not delay_window:
        return candles[start_idx].close

    if is_buy:
        return max(c.high for c in delay_window)
    else:
        return min(c.low for c in delay_window)


def calculate_net_profit(buy_price: float, sell_price: float) -> float:
    """
    Calculate net profit after transaction fees.

    Args:
        buy_price: Execution price for buy.
        sell_price: Execution price for sell.

    Returns:
        Net profit in SOL.
    """
    tokens = FIXED_POSITION_SIZE_SOL / buy_price
    sell_value = tokens * sell_price
    net_value = sell_value - (2 * TOTAL_FEE_PER_TX)
    return net_value - FIXED_POSITION_SIZE_SOL


# =============================================================================
# Model 1: Screener Labels
# =============================================================================

def generate_screener_labels(
    token: TokenData,
    decision_time: int = 30,
    worthy_threshold: float = SCREENER_WORTHY_THRESHOLD,
    lookahead_sec: int = SCREENER_LOOKAHEAD_SEC,
) -> Optional[int]:
    """
    Generate label for Model 1 (Screener).

    A token is WORTHY if it achieves >= worthy_threshold (2x by default)
    within the lookahead window after the decision time.

    Args:
        token: TokenData object.
        decision_time: Time at which decision is made (seconds from start).
        worthy_threshold: Multiple of price required to be worthy.
        lookahead_sec: How far ahead to look for peak.

    Returns:
        ScreenerLabel (0=AVOID, 1=WORTHY) or None if insufficient data.
    """
    candles = token.candles

    if len(candles) <= decision_time:
        return None

    # Entry price at decision time (with delay simulation)
    entry_price = get_execution_price(candles, decision_time, is_buy=True)

    # Look for peak in future window
    future_start = decision_time + DELAY_SECONDS
    future_end = min(future_start + lookahead_sec, len(candles))
    future_candles = candles[future_start:future_end]

    if not future_candles:
        return None

    # Find maximum price in future window
    max_future_price = max(c.high for c in future_candles)

    # Calculate peak ratio
    peak_ratio = max_future_price / entry_price if entry_price > 0 else 1.0

    # Generate label
    if peak_ratio >= worthy_threshold:
        return ScreenerLabel.WORTHY
    else:
        return ScreenerLabel.AVOID


def generate_screener_dataset(
    tokens: List[TokenData],
    decision_time: int = 30,
) -> Tuple[List[np.ndarray], List[int], List[Dict]]:
    """
    Generate complete screener dataset from list of tokens.

    Args:
        tokens: List of TokenData objects.
        decision_time: Time at which to make screening decision.

    Returns:
        Tuple of (features_list, labels_list, metadata_list).
    """
    from .features import extract_screener_features

    features_list = []
    labels_list = []
    metadata_list = []

    for token in tokens:
        # Extract features
        features = extract_screener_features(token, decision_time)
        if features is None:
            continue

        # Generate label
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


# =============================================================================
# Model 2: Entry Labels
# =============================================================================

def generate_entry_labels(
    candles: List[Candle],
    current_idx: int,
    profit_threshold: float = ENTRY_PROFIT_THRESHOLD,
    max_drawdown: float = ENTRY_MAX_DRAWDOWN,
    lookahead_sec: int = ENTRY_LOOKAHEAD_SEC,
) -> Optional[int]:
    """
    Generate label for Model 2 (Entry) at a specific timestamp.

    Labels:
    - ENTER_NOW: Price increases >= profit_threshold in next lookahead_sec
                 AND max drawdown < max_drawdown
    - WAIT: Price close to local peak OR volume declining
    - ABORT: Price dropped significantly OR volume collapsed

    Args:
        candles: List of Candle objects.
        current_idx: Current candle index.
        profit_threshold: Min profit for ENTER_NOW (default 20%).
        max_drawdown: Max acceptable drawdown (default 15%).
        lookahead_sec: Lookahead window for evaluation.

    Returns:
        EntryLabel or None if insufficient data.
    """
    if current_idx >= len(candles) - lookahead_sec - DELAY_SECONDS:
        return None

    # Current price (entry execution with delay)
    entry_price = get_execution_price(candles, current_idx, is_buy=True)

    # Future window
    future_start = current_idx + DELAY_SECONDS
    future_end = min(future_start + lookahead_sec, len(candles))
    future_candles = candles[future_start:future_end]

    if not future_candles:
        return None

    # Calculate metrics
    max_future_high = max(c.high for c in future_candles)
    min_future_low = min(c.low for c in future_candles)

    max_gain = (max_future_high - entry_price) / entry_price if entry_price > 0 else 0.0
    max_loss = (min_future_low - entry_price) / entry_price if entry_price > 0 else 0.0

    # Look at recent history for context
    lookback = min(30, current_idx)
    recent_candles = candles[current_idx - lookback:current_idx + 1]
    recent_high = max(c.high for c in recent_candles) if recent_candles else entry_price
    current_price = candles[current_idx].close

    # Current position relative to recent peak
    dist_from_peak = (current_price - recent_high) / recent_high if recent_high > 0 else 0.0

    # Volume trend (declining volume is warning sign)
    if len(recent_candles) >= 10:
        vol_first = np.mean([c.volume for c in recent_candles[:len(recent_candles)//2]])
        vol_second = np.mean([c.volume for c in recent_candles[len(recent_candles)//2:]])
        volume_declining = vol_second < vol_first * 0.5
    else:
        volume_declining = False

    # Price already dropped significantly from peak
    significant_drop = dist_from_peak < -0.20  # 20% from peak

    # Generate label based on conditions
    if max_gain >= profit_threshold and max_loss > -max_drawdown:
        # Strong bullish: good profit with acceptable drawdown
        return EntryLabel.ENTER_NOW
    elif significant_drop or volume_declining:
        # Danger signs: abort
        return EntryLabel.ABORT
    else:
        # Unclear: wait for better signal
        return EntryLabel.WAIT


def generate_entry_samples(
    token: TokenData,
    start_time: int = 30,
    sample_interval: int = 5,
) -> List[Dict]:
    """
    Generate entry timing samples for a single token.

    Generates samples every sample_interval seconds after start_time.

    Args:
        token: TokenData object.
        start_time: When to start generating samples (after screener passes).
        sample_interval: Interval between samples.

    Returns:
        List of sample dictionaries with features and labels.
    """
    from .features import extract_timeseries_features

    samples = []
    candles = token.candles

    for t in range(start_time, len(candles) - ENTRY_LOOKAHEAD_SEC, sample_interval):
        # Extract time-series features up to current point
        current_candles = candles[:t + 1]
        features = extract_timeseries_features(current_candles)

        if len(features) < 10:  # Need minimum history
            continue

        # Generate label
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


# =============================================================================
# Model 3: Exit Labels
# =============================================================================

def generate_exit_labels(
    candles: List[Candle],
    entry_idx: int,
    entry_price: float,
    current_idx: int,
    trailing_stop: float = EXIT_TRAILING_STOP_PCT,
    stop_loss: float = EXIT_STOP_LOSS_PCT,
    profit_target: float = EXIT_PROFIT_TARGET_PCT,
) -> int:
    """
    Generate label for Model 3 (Exit) at a specific timestamp.

    Labels:
    - EXIT_NOW: Hit stop loss, trailing stop, or profit target
    - PARTIAL_EXIT: Moderate gain with positive momentum
    - HOLD: Still making higher highs, momentum positive

    Args:
        candles: List of Candle objects.
        entry_idx: Index where position was entered.
        entry_price: Price at entry.
        current_idx: Current candle index.
        trailing_stop: Trailing stop percentage from high.
        stop_loss: Stop loss percentage from entry.
        profit_target: Profit target percentage.

    Returns:
        ExitLabel.
    """
    if current_idx >= len(candles):
        return ExitLabel.EXIT_NOW  # End of data

    current_price = candles[current_idx].close

    # Calculate current position metrics
    unrealized_pnl = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0

    # Find position high (from entry to current)
    position_candles = candles[entry_idx:current_idx + 1]
    position_high = max(c.high for c in position_candles) if position_candles else entry_price

    # Drawdown from position high
    drawdown_from_high = (current_price - position_high) / position_high if position_high > 0 else 0.0

    # Time since entry
    time_in_position = current_idx - entry_idx

    # Check hard exit conditions
    # 1. Stop loss hit
    if unrealized_pnl <= -stop_loss:
        return ExitLabel.EXIT_NOW

    # 2. Trailing stop hit
    if drawdown_from_high <= -trailing_stop:
        return ExitLabel.EXIT_NOW

    # 3. Profit target hit
    if unrealized_pnl >= profit_target:
        return ExitLabel.EXIT_NOW

    # 4. Time limit (unless significant gains)
    if time_in_position >= 300 and unrealized_pnl < 1.0:  # 5 min unless 100%+
        return ExitLabel.EXIT_NOW

    # Check for partial exit conditions
    if unrealized_pnl >= 0.30:  # 30% gain
        # Check if momentum is slowing
        if len(position_candles) >= 5:
            recent_momentum = (position_candles[-1].close - position_candles[-5].close) / position_candles[-5].close
            if recent_momentum < 0.05:  # Momentum slowing
                return ExitLabel.PARTIAL_EXIT

    # Check for exhaustion signals
    if len(position_candles) >= 10:
        # Volume exhaustion (declining volume on price increase)
        vol_first = np.mean([c.volume for c in position_candles[:5]])
        vol_last = np.mean([c.volume for c in position_candles[-5:]])

        if vol_last < vol_first * 0.3 and unrealized_pnl > 0.20:
            return ExitLabel.EXIT_NOW  # Volume drying up

    # Default: hold
    return ExitLabel.HOLD


def generate_exit_samples(
    token: TokenData,
    entry_time: int,
    entry_price: float,
    sample_interval: int = 5,
) -> List[Dict]:
    """
    Generate exit timing samples for a simulated trade.

    Args:
        token: TokenData object.
        entry_time: Index where position was entered.
        entry_price: Price at entry.
        sample_interval: Interval between samples.

    Returns:
        List of sample dictionaries with features and labels.
    """
    from .features import extract_exit_features

    samples = []
    candles = token.candles

    for t in range(entry_time + 1, len(candles), sample_interval):
        # Extract exit features
        features = extract_exit_features(candles, entry_time, entry_price, t)

        if features is None:
            continue

        # Generate label
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

        # Stop if exit signal is given
        if label == ExitLabel.EXIT_NOW:
            break

    return samples


# =============================================================================
# Label Generator Class
# =============================================================================

@dataclass
class LabelGenerator:
    """
    Unified label generator for all models.

    Provides consistent interface for generating labels for
    screening, entry timing, and exit optimization.
    """

    screener_threshold: float = SCREENER_WORTHY_THRESHOLD
    screener_lookahead: int = SCREENER_LOOKAHEAD_SEC
    entry_profit_threshold: float = ENTRY_PROFIT_THRESHOLD
    entry_max_drawdown: float = ENTRY_MAX_DRAWDOWN
    exit_trailing_stop: float = EXIT_TRAILING_STOP_PCT
    exit_stop_loss: float = EXIT_STOP_LOSS_PCT
    exit_profit_target: float = EXIT_PROFIT_TARGET_PCT

    def screener_label(self, token: TokenData, decision_time: int = 30) -> Optional[int]:
        """Generate screener label for a token."""
        return generate_screener_labels(
            token, decision_time, self.screener_threshold, self.screener_lookahead
        )

    def entry_label(self, candles: List[Candle], current_idx: int) -> Optional[int]:
        """Generate entry label at a specific timestamp."""
        return generate_entry_labels(
            candles, current_idx, self.entry_profit_threshold, self.entry_max_drawdown
        )

    def exit_label(
        self, candles: List[Candle], entry_idx: int, entry_price: float, current_idx: int
    ) -> int:
        """Generate exit label for a position."""
        return generate_exit_labels(
            candles, entry_idx, entry_price, current_idx,
            self.exit_trailing_stop, self.exit_stop_loss, self.exit_profit_target
        )
