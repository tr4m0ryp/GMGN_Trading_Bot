"""
Label generation for Multi-Model Trading Architecture.

This module re-exports from label_utils and label_generators
for backward compatibility, and defines the LabelGenerator class.

Author: Trading Team
Date: 2025-12-29
"""

from dataclasses import dataclass
from typing import List, Optional

from .loader import TokenData, Candle
from ..config import (
    SCREENER_WORTHY_THRESHOLD,
    SCREENER_LOOKAHEAD_SEC,
    ENTRY_PROFIT_THRESHOLD,
    ENTRY_MAX_DRAWDOWN,
    EXIT_TRAILING_STOP_PCT,
    EXIT_STOP_LOSS_PCT,
    EXIT_PROFIT_TARGET_PCT,
)

# Re-export enums and utilities
from .label_utils import (
    ScreenerLabel,
    EntryLabel,
    ExitLabel,
    get_execution_price,
    calculate_net_profit,
)

# Re-export generators
from .label_generators import (
    generate_screener_labels,
    generate_screener_dataset,
    generate_entry_labels,
    generate_entry_samples,
    generate_exit_labels,
    generate_exit_samples,
)


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


__all__ = [
    # Enums
    "ScreenerLabel",
    "EntryLabel",
    "ExitLabel",
    # Utilities
    "get_execution_price",
    "calculate_net_profit",
    # Generators
    "generate_screener_labels",
    "generate_screener_dataset",
    "generate_entry_labels",
    "generate_entry_samples",
    "generate_exit_labels",
    "generate_exit_samples",
    # Class
    "LabelGenerator",
]
