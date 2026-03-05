"""
Label generation for Multi-Model Trading Architecture.

Provides label enums and label generation functions for:
- Screener (Model 1): AVOID / WORTHY
- Entry (Model 2): WAIT / ENTER_NOW / ABORT
- Exit (Model 3): HOLD / EXIT_NOW / PARTIAL_EXIT

Dependencies: numpy
Date: 2025-12-29
"""

from typing import List, Optional
from enum import IntEnum

import numpy as np

from .data_loading import Candle, TokenData


# =============================================================================
# Label Constants
# =============================================================================

SCREENER_DECISION_TIME = 30
SCREENER_WORTHY_THRESHOLD = 2.0
SCREENER_LOOKAHEAD_SEC = 300
ENTRY_PROFIT_THRESHOLD = 0.20
ENTRY_MAX_DRAWDOWN = 0.15
ENTRY_LOOKAHEAD_SEC = 60
ENTRY_SAMPLES_PER_TOKEN = 50
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
# Execution Price Helper
# =============================================================================

def get_execution_price(candles: List[Candle], start_idx: int, is_buy: bool = True) -> float:
    """Get execution price accounting for delay slippage."""
    end_idx = min(start_idx + DELAY_SECONDS + 1, len(candles))
    delay_window = candles[start_idx:end_idx]

    if not delay_window:
        return candles[start_idx].close

    if is_buy:
        return max(c.high for c in delay_window)
    else:
        return min(c.low for c in delay_window)


# =============================================================================
# Screener Label (Model 1)
# =============================================================================

def generate_screener_label(token: TokenData, decision_time: int = 30) -> Optional[int]:
    """Generate screener label based on future price peak ratio."""
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


# =============================================================================
# Entry Label (Model 2)
# =============================================================================

def generate_entry_label(candles: List[Candle], current_idx: int) -> Optional[int]:
    """
    Generate entry label - ORIGINAL working version.

    Labels:
    - ENTER_NOW: Good opportunity (gain possible, acceptable drawdown)
    - ABORT: Bad situation (significant drop or declining volume)
    - WAIT: Unclear signal
    """
    if current_idx >= len(candles) - ENTRY_LOOKAHEAD_SEC - DELAY_SECONDS:
        return None

    entry_price = get_execution_price(candles, current_idx, is_buy=True)

    future_start = current_idx + DELAY_SECONDS
    future_end = min(future_start + ENTRY_LOOKAHEAD_SEC, len(candles))
    future_candles = candles[future_start:future_end]

    if not future_candles:
        return None

    max_future_high = max(c.high for c in future_candles)
    min_future_low = min(c.low for c in future_candles)

    max_gain = (max_future_high - entry_price) / entry_price if entry_price > 0 else 0.0
    max_loss = (min_future_low - entry_price) / entry_price if entry_price > 0 else 0.0

    # Check for abort conditions based on current state
    lookback = min(30, current_idx)
    recent_candles = candles[current_idx - lookback:current_idx + 1]
    recent_high = max(c.high for c in recent_candles) if recent_candles else entry_price
    current_price = candles[current_idx].close

    dist_from_peak = (current_price - recent_high) / recent_high if recent_high > 0 else 0.0
    significant_drop = dist_from_peak < -0.20

    if len(recent_candles) >= 10:
        vol_first = np.mean([c.volume for c in recent_candles[:len(recent_candles)//2]])
        vol_second = np.mean([c.volume for c in recent_candles[len(recent_candles)//2:]])
        volume_declining = vol_second < vol_first * 0.5
    else:
        volume_declining = False

    # Label assignment
    if max_gain >= ENTRY_PROFIT_THRESHOLD and max_loss > -ENTRY_MAX_DRAWDOWN:
        return EntryLabel.ENTER_NOW
    elif significant_drop or volume_declining:
        return EntryLabel.ABORT
    else:
        return EntryLabel.WAIT


# =============================================================================
# Exit Label (Model 3)
# =============================================================================

def generate_exit_label(
    candles: List[Candle],
    entry_idx: int,
    entry_price: float,
    current_idx: int,
) -> int:
    """Generate exit label based on position P&L and trailing stop logic."""
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
            recent_momentum = (
                (position_candles[-1].close - position_candles[-5].close)
                / position_candles[-5].close
            )
            if recent_momentum < 0.05:
                return ExitLabel.PARTIAL_EXIT

    return ExitLabel.HOLD
