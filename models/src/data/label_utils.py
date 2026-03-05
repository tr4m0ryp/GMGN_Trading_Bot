"""
Label enums and utility functions for Multi-Model Trading Architecture.

Defines label types for each model and shared helper functions
used during label generation.

Author: Trading Team
Date: 2025-12-29
"""

from typing import List
from enum import IntEnum

from .loader import Candle
from ..config import (
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
