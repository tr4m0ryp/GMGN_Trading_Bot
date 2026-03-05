"""
Kelly Backtesting: Backtesting utility for Kelly-based strategies.

Simulates portfolio growth using Kelly position sizing
with predicted returns and variances.

Dependencies:
    numpy

Date: 2025-12-25
"""

from typing import Dict, Union

import numpy as np


class KellyBacktester:
    """
    Backtesting utility for Kelly-based strategies.

    Simulates portfolio growth using Kelly position sizing
    with predicted returns and variances.

    Args:
        sizer: KellyPositionSizer instance
        initial_capital: Starting portfolio value. Default 1.0

    Example:
        >>> backtester = KellyBacktester(sizer)
        >>> results = backtester.run(mus, log_vars, actual_returns)
    """

    def __init__(self, sizer, initial_capital: float = 1.0):
        self.sizer = sizer
        self.initial_capital = initial_capital

    def run(
        self,
        mus: np.ndarray,
        log_vars: np.ndarray,
        actual_returns: np.ndarray,
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Run backtest on historical predictions.

        Args:
            mus: Predicted means [N]
            log_vars: Predicted log-variances [N]
            actual_returns: Actual realized returns [N]

        Returns:
            Dictionary with equity curve and performance metrics
        """
        n_samples = len(mus)

        equity = np.zeros(n_samples + 1)
        equity[0] = self.initial_capital

        positions_taken = np.zeros(n_samples)
        pnl_per_trade = np.zeros(n_samples)
        trades = 0
        wins = 0

        for i in range(n_samples):
            rec = self.sizer.get_position(
                mus[i], log_vars[i], equity[i]
            )

            if rec.action == 'BUY' and rec.position_fraction > 0:
                position = rec.position_fraction * equity[i]
                pnl = position * actual_returns[i]
                pnl -= position * self.sizer.transaction_cost

                equity[i + 1] = equity[i] + pnl
                positions_taken[i] = rec.position_fraction
                pnl_per_trade[i] = pnl
                trades += 1

                if pnl > 0:
                    wins += 1
            else:
                equity[i + 1] = equity[i]

        # Compute metrics
        final_value = equity[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital

        active_trades = positions_taken > 0
        trade_returns = pnl_per_trade[active_trades] / (
            positions_taken[active_trades] * equity[:-1][active_trades] + 1e-10
        )

        win_rate = wins / trades if trades > 0 else 0
        avg_trade_return = trade_returns.mean() if len(trade_returns) > 0 else 0

        # Sharpe ratio (annualized, assuming 1-second candles)
        returns = np.diff(equity) / equity[:-1]
        returns = returns[returns != 0]
        if len(returns) > 1:
            sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(31536000)
        else:
            sharpe = 0

        # Maximum drawdown
        running_max = np.maximum.accumulate(equity)
        drawdowns = (running_max - equity) / running_max
        max_drawdown = drawdowns.max()

        return {
            'equity_curve': equity,
            'final_value': final_value,
            'total_return': total_return,
            'total_trades': trades,
            'win_rate': win_rate,
            'avg_trade_return': avg_trade_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'positions': positions_taken,
            'pnl': pnl_per_trade,
        }

    def run_comparison(
        self,
        mus: np.ndarray,
        log_vars: np.ndarray,
        actual_returns: np.ndarray,
        kelly_fractions: list = None,
    ) -> Dict[str, Dict]:
        """
        Compare different Kelly fractions.

        Args:
            mus: Predicted means [N]
            log_vars: Predicted log-variances [N]
            actual_returns: Actual returns [N]
            kelly_fractions: List of fractions to compare

        Returns:
            Dictionary mapping fraction to results
        """
        if kelly_fractions is None:
            kelly_fractions = [0.1, 0.25, 0.5, 1.0]

        results = {}
        original_fraction = self.sizer.kelly_fraction

        for frac in kelly_fractions:
            self.sizer.kelly_fraction = frac
            results[f'kelly_{frac}'] = self.run(mus, log_vars, actual_returns)

        self.sizer.kelly_fraction = original_fraction

        return results
