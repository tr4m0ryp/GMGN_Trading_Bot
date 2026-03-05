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
from typing import Dict
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
        self, mu: float, sigma_sq: float, adjust_for_fees: bool = True,
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
        if adjust_for_fees:
            mu_adj = mu - self.transaction_cost
        else:
            mu_adj = mu

        if sigma_sq < 1e-8:
            sigma_sq = 1e-8

        f_kelly = mu_adj / sigma_sq
        f_adjusted = self.kelly_fraction * f_kelly
        return f_adjusted

    def get_position(
        self, mu: float, log_var: float, portfolio_value: float = 1.0,
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
        sigma_sq = math.exp(log_var)
        sigma = math.sqrt(sigma_sq)
        kelly_raw = self.compute_kelly_fraction(mu, sigma_sq)
        sharpe = mu / sigma if sigma > 1e-8 else 0

        if sigma_sq < 0.001:
            confidence = 'high'
        elif sigma_sq < 0.005:
            confidence = 'medium'
        else:
            confidence = 'low'

        # Check filters
        hold_reason = self._check_filters(mu, sigma_sq, sharpe, kelly_raw)
        if hold_reason is not None:
            action = 'SELL' if hold_reason == 'bearish_signal' else 'HOLD'
            return PositionRecommendation(
                action=action, position_fraction=0.0, position_size=0.0,
                reason=hold_reason, kelly_raw=kelly_raw, mu=mu,
                sigma=sigma, sharpe=sharpe, confidence=confidence,
            )

        # Bullish signal - compute position size
        f_clamped = min(kelly_raw, self.max_position)
        f_clamped = max(f_clamped, 0.0)
        position_size = f_clamped * portfolio_value

        return PositionRecommendation(
            action='BUY', position_fraction=f_clamped, position_size=position_size,
            reason='positive_edge', kelly_raw=kelly_raw, mu=mu,
            sigma=sigma, sharpe=sharpe, confidence=confidence,
        )

    def _check_filters(self, mu, sigma_sq, sharpe, kelly_raw):
        """Check trading filters. Returns reason string if should hold, else None."""
        if sigma_sq > self.max_variance:
            return 'variance_too_high'
        if sharpe < self.min_sharpe:
            return 'sharpe_too_low'
        if mu < self.min_edge:
            if mu < -self.min_edge:
                return 'bearish_signal'
            return 'insufficient_edge'
        if kelly_raw < 0:
            return 'kelly_negative'
        return None

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
