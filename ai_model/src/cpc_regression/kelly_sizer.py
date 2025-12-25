"""
KellyPositionSizer: Optimal position sizing using Kelly Criterion.

The Kelly Criterion provides the mathematically optimal bet size
for maximizing long-term growth rate.

For continuous returns modeled as Gaussian(mu, sigma^2):
    f* = mu / sigma^2

Where f* is the fraction of capital to bet.

Dependencies:
    numpy

Date: 2025-12-25
"""

import math
from typing import Dict, Optional, Union
from dataclasses import dataclass

import numpy as np


@dataclass
class PositionRecommendation:
    """Position sizing recommendation from Kelly sizer."""
    action: str  # 'BUY', 'SELL', 'HOLD'
    position_fraction: float  # Fraction of portfolio (0 to max_position)
    position_size: float  # Absolute position size
    reason: str  # Explanation for the decision
    kelly_raw: float  # Raw Kelly fraction before adjustments
    mu: float  # Predicted mean return
    sigma: float  # Predicted standard deviation
    sharpe: float  # Approximate Sharpe ratio (mu / sigma)
    confidence: str  # 'low', 'medium', 'high' based on variance


class KellyPositionSizer:
    """
    Position sizing using Kelly Criterion with safety adjustments.

    The Kelly Criterion maximizes expected log-wealth growth.
    We use fractional Kelly (default 1/4) for reduced volatility.

    Key formulas:
        Full Kelly: f* = mu / sigma^2
        Fractional Kelly: f_adj = kelly_fraction * f*
        Final position: clamp(f_adj, 0, max_position)

    Args:
        kelly_fraction: Fraction of Kelly to use. Default 0.25 (1/4 Kelly).
                       Lower = more conservative, higher = more aggressive.
        max_position: Maximum position as fraction of portfolio. Default 0.05.
        min_edge: Minimum predicted return to consider trading. Default 0.02.
        max_variance: Maximum variance to accept. Default 0.01.
        transaction_cost: Total transaction costs (fees). Default 0.007 (0.7%).
        min_sharpe: Minimum Sharpe ratio to trade. Default 0.5.

    Example:
        >>> sizer = KellyPositionSizer(kelly_fraction=0.25)
        >>> rec = sizer.get_position(mu=0.05, log_var=-4.0)
        >>> print(f"Action: {rec.action}, Size: {rec.position_fraction:.2%}")
    """

    def __init__(
        self,
        kelly_fraction: float = 0.25,
        max_position: float = 0.05,
        min_edge: float = 0.02,
        max_variance: float = 0.01,
        transaction_cost: float = 0.007,
        min_sharpe: float = 0.5,
    ):
        self.kelly_fraction = kelly_fraction
        self.max_position = max_position
        self.min_edge = min_edge
        self.max_variance = max_variance
        self.transaction_cost = transaction_cost
        self.min_sharpe = min_sharpe

    def compute_kelly_fraction(
        self,
        mu: float,
        sigma_sq: float,
        adjust_for_fees: bool = True,
    ) -> float:
        """
        Compute Kelly-optimal position fraction.

        Args:
            mu: Predicted mean return
            sigma_sq: Predicted variance
            adjust_for_fees: Whether to subtract transaction costs

        Returns:
            Optimal fraction of capital to allocate
        """
        # Adjust expected return for transaction costs
        if adjust_for_fees:
            mu_adj = mu - self.transaction_cost
        else:
            mu_adj = mu

        # Kelly fraction: f* = mu / sigma^2
        if sigma_sq < 1e-8:
            sigma_sq = 1e-8  # Prevent division by zero

        f_kelly = mu_adj / sigma_sq

        # Apply fractional Kelly (conservative)
        f_adjusted = self.kelly_fraction * f_kelly

        return f_adjusted

    def get_position(
        self,
        mu: float,
        log_var: float,
        portfolio_value: float = 1.0,
    ) -> PositionRecommendation:
        """
        Get position sizing recommendation.

        Args:
            mu: Predicted mean return (from model)
            log_var: Predicted log-variance (from model)
            portfolio_value: Current portfolio value

        Returns:
            PositionRecommendation with action and sizing
        """
        # Convert log-variance to variance and std
        sigma_sq = math.exp(log_var)
        sigma = math.sqrt(sigma_sq)

        # Compute raw Kelly fraction
        kelly_raw = self.compute_kelly_fraction(mu, sigma_sq)

        # Approximate Sharpe ratio
        sharpe = mu / sigma if sigma > 1e-8 else 0

        # Confidence level based on variance
        if sigma_sq < 0.001:
            confidence = 'high'
        elif sigma_sq < 0.005:
            confidence = 'medium'
        else:
            confidence = 'low'

        # Decision logic

        # Check 1: Variance too high (too uncertain)
        if sigma_sq > self.max_variance:
            return PositionRecommendation(
                action='HOLD',
                position_fraction=0.0,
                position_size=0.0,
                reason='variance_too_high',
                kelly_raw=kelly_raw,
                mu=mu,
                sigma=sigma,
                sharpe=sharpe,
                confidence=confidence,
            )

        # Check 2: Sharpe ratio too low
        if sharpe < self.min_sharpe:
            return PositionRecommendation(
                action='HOLD',
                position_fraction=0.0,
                position_size=0.0,
                reason='sharpe_too_low',
                kelly_raw=kelly_raw,
                mu=mu,
                sigma=sigma,
                sharpe=sharpe,
                confidence=confidence,
            )

        # Check 3: Expected return below threshold
        if mu < self.min_edge:
            if mu < -self.min_edge:
                # Strong bearish signal
                return PositionRecommendation(
                    action='SELL',
                    position_fraction=0.0,
                    position_size=0.0,
                    reason='bearish_signal',
                    kelly_raw=kelly_raw,
                    mu=mu,
                    sigma=sigma,
                    sharpe=sharpe,
                    confidence=confidence,
                )
            else:
                # Weak signal - hold
                return PositionRecommendation(
                    action='HOLD',
                    position_fraction=0.0,
                    position_size=0.0,
                    reason='insufficient_edge',
                    kelly_raw=kelly_raw,
                    mu=mu,
                    sigma=sigma,
                    sharpe=sharpe,
                    confidence=confidence,
                )

        # Check 4: Kelly fraction suggests short (negative)
        if kelly_raw < 0:
            return PositionRecommendation(
                action='HOLD',
                position_fraction=0.0,
                position_size=0.0,
                reason='kelly_negative',
                kelly_raw=kelly_raw,
                mu=mu,
                sigma=sigma,
                sharpe=sharpe,
                confidence=confidence,
            )

        # Bullish signal - compute position size
        f_clamped = min(kelly_raw, self.max_position)
        f_clamped = max(f_clamped, 0.0)

        position_size = f_clamped * portfolio_value

        return PositionRecommendation(
            action='BUY',
            position_fraction=f_clamped,
            position_size=position_size,
            reason='positive_edge',
            kelly_raw=kelly_raw,
            mu=mu,
            sigma=sigma,
            sharpe=sharpe,
            confidence=confidence,
        )

    def get_position_batch(
        self,
        mus: np.ndarray,
        log_vars: np.ndarray,
        portfolio_value: float = 1.0,
    ) -> Dict[str, np.ndarray]:
        """
        Get position recommendations for a batch of predictions.

        Args:
            mus: Predicted means [batch]
            log_vars: Predicted log-variances [batch]
            portfolio_value: Portfolio value

        Returns:
            Dictionary with arrays of actions, positions, etc.
        """
        batch_size = len(mus)

        actions = []
        positions = np.zeros(batch_size)
        reasons = []
        sharpes = np.zeros(batch_size)

        for i in range(batch_size):
            rec = self.get_position(mus[i], log_vars[i], portfolio_value)
            actions.append(rec.action)
            positions[i] = rec.position_fraction
            reasons.append(rec.reason)
            sharpes[i] = rec.sharpe

        return {
            'actions': np.array(actions),
            'positions': positions,
            'reasons': np.array(reasons),
            'sharpes': sharpes,
        }


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

    def __init__(
        self,
        sizer: KellyPositionSizer,
        initial_capital: float = 1.0,
    ):
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

        # Track portfolio value over time
        equity = np.zeros(n_samples + 1)
        equity[0] = self.initial_capital

        # Track decisions
        positions_taken = np.zeros(n_samples)
        pnl_per_trade = np.zeros(n_samples)
        trades = 0
        wins = 0

        for i in range(n_samples):
            rec = self.sizer.get_position(
                mus[i], log_vars[i], equity[i]
            )

            if rec.action == 'BUY' and rec.position_fraction > 0:
                # Execute trade
                position = rec.position_fraction * equity[i]
                pnl = position * actual_returns[i]

                # Apply transaction costs
                pnl -= position * self.sizer.transaction_cost

                equity[i + 1] = equity[i] + pnl
                positions_taken[i] = rec.position_fraction
                pnl_per_trade[i] = pnl
                trades += 1

                if pnl > 0:
                    wins += 1
            else:
                # No trade
                equity[i + 1] = equity[i]

        # Compute metrics
        final_value = equity[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital

        # Only consider non-zero trades
        active_trades = positions_taken > 0
        trade_returns = pnl_per_trade[active_trades] / (
            positions_taken[active_trades] * equity[:-1][active_trades] + 1e-10
        )

        win_rate = wins / trades if trades > 0 else 0
        avg_trade_return = trade_returns.mean() if len(trade_returns) > 0 else 0

        # Sharpe ratio (annualized, assuming 1-second candles)
        # Approximately 31.5M seconds per year
        returns = np.diff(equity) / equity[:-1]
        returns = returns[returns != 0]  # Remove non-trading periods
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

        # Restore original
        self.sizer.kelly_fraction = original_fraction

        return results
