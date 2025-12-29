"""
Backtesting engine for Multi-Model Trading Architecture.

Simulates the complete trading pipeline:
1. Token screening (Model 1)
2. Entry timing (Model 2)
3. Exit optimization (Model 3)

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
    DELAY_SECONDS,
)
from ..data.loader import TokenData
from ..data.features import FeatureExtractor
from ..models.exit import Position, ExitReason, ExitLabel


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

    # Trade records
    trades: List[Trade] = field(default_factory=list)

    # Summary metrics
    total_tokens: int = 0
    tokens_screened_in: int = 0
    tokens_entered: int = 0
    tokens_completed: int = 0

    # Performance
    winning_trades: int = 0
    losing_trades: int = 0
    total_profit: float = 0.0
    total_loss: float = 0.0
    net_pnl: float = 0.0
    max_drawdown: float = 0.0

    # Ratios
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

        # Win/loss
        wins = [t for t in completed if t.net_pnl > 0]
        losses = [t for t in completed if t.net_pnl <= 0]

        self.winning_trades = len(wins)
        self.losing_trades = len(losses)
        self.total_profit = sum(t.net_pnl for t in wins)
        self.total_loss = abs(sum(t.net_pnl for t in losses))
        self.net_pnl = self.total_profit - self.total_loss

        # Ratios
        self.win_rate = self.winning_trades / len(completed) if completed else 0
        self.avg_win = self.total_profit / self.winning_trades if self.winning_trades else 0
        self.avg_loss = self.total_loss / self.losing_trades if self.losing_trades else 0
        self.profit_factor = self.total_profit / self.total_loss if self.total_loss > 0 else float('inf')

        # Equity curve for drawdown and Sharpe
        returns = [t.net_pnl_pct for t in completed]
        equity = np.cumsum(returns)

        if len(equity) > 0:
            peak = np.maximum.accumulate(equity)
            drawdown = equity - peak
            self.max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0

            if len(returns) > 1 and np.std(returns) > 0:
                self.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(len(returns))


class Backtester:
    """
    Backtesting engine for the complete trading pipeline.

    Simulates trading with all three models on historical data.
    """

    def __init__(
        self,
        pipeline=None,
        config: Optional[BacktestConfig] = None,
    ):
        """
        Initialize backtester.

        Args:
            pipeline: TradingPipeline with trained models.
            config: Backtesting configuration.
        """
        self.pipeline = pipeline
        self.config = config or DEFAULT_BACKTEST_CONFIG
        self.feature_extractor = FeatureExtractor()

    def run(
        self,
        tokens: List[TokenData],
        verbose: int = 1,
    ) -> BacktestResult:
        """
        Run backtest on list of tokens.

        Args:
            tokens: List of TokenData objects.
            verbose: Verbosity level.

        Returns:
            BacktestResult with all trades and metrics.
        """
        result = BacktestResult(total_tokens=len(tokens))

        for i, token in enumerate(tokens):
            if verbose > 0 and (i + 1) % 50 == 0:
                print(f"  Processing token {i+1}/{len(tokens)}")

            trade = self._simulate_token(token)
            result.trades.append(trade)

        result.calculate_metrics()
        return result

    def _simulate_token(self, token: TokenData) -> Trade:
        """
        Simulate trading a single token.

        Args:
            token: TokenData to simulate.

        Returns:
            Trade record.
        """
        trade = Trade(
            token_address=token.token_address,
            symbol=token.symbol,
        )

        candles = token.candles

        # Step 1: Screener (at t=30)
        if len(candles) < 35:
            return trade  # Not enough data

        screener_features = self.feature_extractor.extract_for_screener(token)
        if screener_features is None:
            return trade

        if self.pipeline and self.pipeline.screener:
            is_worthy, confidence = self.pipeline.screen_token(screener_features)
            trade.screener_confidence = confidence
            trade.screener_passed = is_worthy
        else:
            trade.screener_passed = True
            trade.screener_confidence = 1.0

        if not trade.screener_passed:
            trade.status = TradeStatus.SCREENED_OUT
            return trade

        # Step 2: Entry Timing (check every 5s for 3 minutes max)
        max_wait_time = 180  # 3 minutes
        entry_start = 30

        for t in range(entry_start, min(entry_start + max_wait_time, len(candles) - 60), 5):
            entry_features = self.feature_extractor.extract_for_entry(candles[:t+1])

            if len(entry_features) < 10:
                continue

            if self.pipeline and self.pipeline.entry:
                label, confidence = self.pipeline.check_entry(entry_features, len(entry_features))
                trade.entry_confidence = confidence

                if label == 1:  # ENTER_NOW
                    trade.entry_time = t
                    # Simulate execution with delay
                    exec_idx = min(t + DELAY_SECONDS, len(candles) - 1)
                    trade.entry_price = candles[exec_idx].high  # Worst-case slippage
                    break
                elif label == 2:  # ABORT
                    trade.status = TradeStatus.ABORTED
                    return trade
            else:
                # Default: enter immediately if no model
                trade.entry_time = t
                exec_idx = min(t + DELAY_SECONDS, len(candles) - 1)
                trade.entry_price = candles[exec_idx].close
                break

        if trade.entry_time is None:
            trade.status = TradeStatus.NO_ENTRY
            return trade

        # Step 3: Exit (check every 5s)
        position = Position(
            entry_price=trade.entry_price,
            entry_time=trade.entry_time,
            position_high=trade.entry_price,
        )

        for t in range(trade.entry_time + 1, len(candles), 5):
            current_price = candles[t].close
            position.update(current_price)

            if self.pipeline and self.pipeline.exit:
                exit_features = self.feature_extractor.extract_for_exit(
                    candles, trade.entry_time, trade.entry_price, t
                )
                if exit_features is not None:
                    label, reason, fraction = self.pipeline.check_exit(
                        exit_features, position, current_price, t
                    )

                    if label in [ExitLabel.EXIT_NOW, ExitLabel.PARTIAL_EXIT]:
                        trade.exit_time = t
                        # Simulate execution with delay
                        exec_idx = min(t + DELAY_SECONDS, len(candles) - 1)
                        trade.exit_price = candles[exec_idx].low  # Worst-case slippage
                        trade.exit_reason = reason
                        break
            else:
                # Default: use risk manager only
                from ..models.exit import RiskManager
                rm = RiskManager()
                should_exit, reason, _ = rm.check_exit_rules(position, current_price, t)

                if should_exit:
                    trade.exit_time = t
                    exec_idx = min(t + DELAY_SECONDS, len(candles) - 1)
                    trade.exit_price = candles[exec_idx].low
                    trade.exit_reason = reason
                    break

        # If no exit signal, exit at end
        if trade.exit_time is None and trade.entry_price:
            trade.exit_time = len(candles) - 1
            trade.exit_price = candles[-1].close
            trade.exit_reason = ExitReason.TIME_STOP

        trade.status = TradeStatus.COMPLETED
        trade.calculate_pnl()

        return trade


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
