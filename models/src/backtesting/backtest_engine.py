"""
Backtest Engine: Core backtesting simulation logic.

Contains the Backtester class that simulates the complete
trading pipeline on historical data.

Author: Trading Team
Date: 2025-12-29
"""

from typing import Dict, Any, List, Optional, Tuple

from ..config import (
    BacktestConfig,
    DEFAULT_BACKTEST_CONFIG,
    DELAY_SECONDS,
)
from ..data.loader import TokenData
from ..data.features import FeatureExtractor
from ..models.exit import Position, ExitReason, ExitLabel

from .backtester import Trade, TradeStatus, BacktestResult


class Backtester:
    """
    Backtesting engine for the complete trading pipeline.

    Simulates trading with all three models on historical data.
    """

    def __init__(self, pipeline=None, config: Optional[BacktestConfig] = None):
        """
        Initialize backtester.

        Args:
            pipeline: TradingPipeline with trained models.
            config: Backtesting configuration.
        """
        self.pipeline = pipeline
        self.config = config or DEFAULT_BACKTEST_CONFIG
        self.feature_extractor = FeatureExtractor()

    def run(self, tokens: List[TokenData], verbose: int = 1) -> BacktestResult:
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
        """Simulate trading a single token."""
        trade = Trade(token_address=token.token_address, symbol=token.symbol)
        candles = token.candles

        # Step 1: Screener (at t=30)
        if len(candles) < 35:
            return trade

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

        # Step 2: Entry Timing
        trade = self._simulate_entry(trade, candles)
        if trade.entry_time is None:
            return trade

        # Step 3: Exit
        trade = self._simulate_exit(trade, candles)
        trade.status = TradeStatus.COMPLETED
        trade.calculate_pnl()
        return trade

    def _simulate_entry(self, trade: Trade, candles) -> Trade:
        """Simulate entry timing phase."""
        max_wait_time = 180
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
                    exec_idx = min(t + DELAY_SECONDS, len(candles) - 1)
                    trade.entry_price = candles[exec_idx].high
                    break
                elif label == 2:  # ABORT
                    trade.status = TradeStatus.ABORTED
                    return trade
            else:
                trade.entry_time = t
                exec_idx = min(t + DELAY_SECONDS, len(candles) - 1)
                trade.entry_price = candles[exec_idx].close
                break

        if trade.entry_time is None:
            trade.status = TradeStatus.NO_ENTRY

        return trade

    def _simulate_exit(self, trade: Trade, candles) -> Trade:
        """Simulate exit phase."""
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
                        exec_idx = min(t + DELAY_SECONDS, len(candles) - 1)
                        trade.exit_price = candles[exec_idx].low
                        trade.exit_reason = reason
                        break
            else:
                from ..models.exit import RiskManager
                rm = RiskManager()
                should_exit, reason, _ = rm.check_exit_rules(position, current_price, t)
                if should_exit:
                    trade.exit_time = t
                    exec_idx = min(t + DELAY_SECONDS, len(candles) - 1)
                    trade.exit_price = candles[exec_idx].low
                    trade.exit_reason = reason
                    break

        if trade.exit_time is None and trade.entry_price:
            trade.exit_time = len(candles) - 1
            trade.exit_price = candles[-1].close
            trade.exit_reason = ExitReason.TIME_STOP

        return trade
