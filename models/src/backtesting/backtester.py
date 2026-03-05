"""
Backtester: Data structures and reporting for backtesting.

Contains Trade, BacktestResult data classes and the run_backtest
convenience function for the complete trading pipeline.

Author: Trading Team
Date: 2025-12-29
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

import numpy as np

from ..config import (
    BacktestConfig,
    DEFAULT_BACKTEST_CONFIG,
    TOTAL_FEE_PER_TX,
    FIXED_POSITION_SIZE_SOL,
)
from ..data.loader import TokenData
from ..models.exit import ExitReason


class TradeStatus(Enum):
    """Status of a trade."""
    SCREENED_OUT = "screened_out"
    NO_ENTRY = "no_entry"
    ABORTED = "aborted"
    COMPLETED = "completed"


@dataclass
class Trade:
    """Record of a single trade."""

    token_address: str
    symbol: str

    # Screening
    screener_passed: bool = False
    screener_confidence: float = 0.0

    # Entry
    entry_time: Optional[int] = None
    entry_price: Optional[float] = None
    entry_confidence: float = 0.0

    # Exit
    exit_time: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[ExitReason] = None

    # P&L
    gross_pnl: float = 0.0
    fees: float = 0.0
    net_pnl: float = 0.0
    net_pnl_pct: float = 0.0

    # Status
    status: TradeStatus = TradeStatus.SCREENED_OUT

    def calculate_pnl(self, position_size: float = FIXED_POSITION_SIZE_SOL) -> None:
        """Calculate P&L metrics."""
        if self.entry_price and self.exit_price:
            tokens = position_size / self.entry_price
            self.gross_pnl = tokens * (self.exit_price - self.entry_price)
            self.fees = 2 * TOTAL_FEE_PER_TX
            self.net_pnl = self.gross_pnl - self.fees
            self.net_pnl_pct = self.net_pnl / position_size


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    trades: List[Trade] = field(default_factory=list)

    total_tokens: int = 0
    tokens_screened_in: int = 0
    tokens_entered: int = 0
    tokens_completed: int = 0

    winning_trades: int = 0
    losing_trades: int = 0
    total_profit: float = 0.0
    total_loss: float = 0.0
    net_pnl: float = 0.0
    max_drawdown: float = 0.0

    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0

    def calculate_metrics(self) -> None:
        """Calculate summary metrics from trades."""
        completed = [t for t in self.trades if t.status == TradeStatus.COMPLETED]

        self.tokens_screened_in = sum(1 for t in self.trades if t.screener_passed)
        self.tokens_entered = sum(1 for t in self.trades if t.entry_time is not None)
        self.tokens_completed = len(completed)

        if not completed:
            return

        wins = [t for t in completed if t.net_pnl > 0]
        losses = [t for t in completed if t.net_pnl <= 0]

        self.winning_trades = len(wins)
        self.losing_trades = len(losses)
        self.total_profit = sum(t.net_pnl for t in wins)
        self.total_loss = abs(sum(t.net_pnl for t in losses))
        self.net_pnl = self.total_profit - self.total_loss

        self.win_rate = self.winning_trades / len(completed) if completed else 0
        self.avg_win = self.total_profit / self.winning_trades if self.winning_trades else 0
        self.avg_loss = self.total_loss / self.losing_trades if self.losing_trades else 0
        self.profit_factor = self.total_profit / self.total_loss if self.total_loss > 0 else float('inf')

        returns = [t.net_pnl_pct for t in completed]
        equity = np.cumsum(returns)

        if len(equity) > 0:
            peak = np.maximum.accumulate(equity)
            drawdown = equity - peak
            self.max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0

            if len(returns) > 1 and np.std(returns) > 0:
                self.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(len(returns))


def run_backtest(
    tokens: List[TokenData],
    pipeline=None,
    config: Optional[BacktestConfig] = None,
    verbose: int = 1,
) -> BacktestResult:
    """
    Run backtest with optional pipeline.

    Args:
        tokens: List of tokens to backtest.
        pipeline: Optional TradingPipeline.
        config: Backtest configuration.
        verbose: Verbosity level.

    Returns:
        BacktestResult.
    """
    from .backtest_engine import Backtester

    print("=" * 70)
    print("RUNNING BACKTEST")
    print("=" * 70)

    backtester = Backtester(pipeline, config)
    result = backtester.run(tokens, verbose)

    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    print(f"Total tokens: {result.total_tokens}")
    print(f"Screened in: {result.tokens_screened_in} ({100*result.tokens_screened_in/result.total_tokens:.1f}%)")
    print(f"Entered: {result.tokens_entered}")
    print(f"Completed: {result.tokens_completed}")
    print()
    print(f"Winning trades: {result.winning_trades}")
    print(f"Losing trades: {result.losing_trades}")
    print(f"Win rate: {result.win_rate:.1%}")
    print()
    print(f"Total profit: {result.total_profit:.4f} SOL")
    print(f"Total loss: {result.total_loss:.4f} SOL")
    print(f"Net P&L: {result.net_pnl:.4f} SOL")
    print()
    print(f"Avg win: {result.avg_win:.4f} SOL")
    print(f"Avg loss: {result.avg_loss:.4f} SOL")
    print(f"Profit factor: {result.profit_factor:.2f}")
    print(f"Max drawdown: {result.max_drawdown:.2%}")
    print(f"Sharpe ratio: {result.sharpe_ratio:.2f}")
    print("=" * 70)

    return result
