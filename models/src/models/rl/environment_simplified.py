"""
Simplified trading environment for A/B comparison against V5.1.

Contains TradingEnvironmentSimplified - a cleaner reward system with
fewer hyperparameters for comparison testing.

Author: Trading Team
Date: 2025-12-24
"""

from typing import Dict, Tuple, List, Any, Optional
import numpy as np

from config import (
    FIXED_POSITION_SIZE,
    DELAY_SECONDS,
    MIN_HISTORY_LENGTH,
)
from .environment import TradingEnvironmentV2


class TradingEnvironmentSimplified(TradingEnvironmentV2):
    """
    Trading environment with SIMPLIFIED reward system for A/B comparison.

    SIMPLIFIED REWARD SYSTEM:
    =========================
    Per-Trade:
        Win:  +1.0 + (return x 10)     Example: +5% = +1.5 reward
        Loss: -0.5 + (return x 5)      Example: -5% = -0.75 reward (capped at -1.0)

    End-of-Episode:
        0 trades: -5.0 penalty
        Win rate >= 70%: +1.0 bonus
        Win rate >= 85%: +2.0 bonus
        Profit > 0: +1.0 bonus

    Key Differences from V5.1:
    - No base_trade_reward (simpler)
    - No sweet_spot_bonus, synergy_bonus, quick_trade_bonus
    - No momentum rewards
    - Losses are NOT guaranteed positive (can be negative)
    - Much simpler end-of-episode bonuses
    """

    def __init__(
        self,
        candles: List[Dict[str, float]],
        max_steps: Optional[int] = None,
        fee_multiplier: float = 0.5,
        position_size: float = FIXED_POSITION_SIZE,
        **kwargs  # Accept but ignore V5.1 params for compatibility
    ):
        super().__init__(
            candles=candles,
            max_steps=max_steps,
            fee_multiplier=fee_multiplier,
            position_size=position_size,
        )

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step with SIMPLIFIED reward system."""
        reward = 0.0
        terminated = False
        truncated = False
        trade_executed = False
        trade_pnl = 0.0

        # Execute action
        if action == 1:  # BUY
            if not self.in_position:
                self.entry_price = self._get_execution_price(is_buy=True)
                self.entry_step = self.current_step
                self.in_position = True
                trade_executed = True
            else:
                reward -= 0.01

        elif action == 2:  # SELL
            if self.in_position:
                sell_price = self._get_execution_price(is_buy=False)
                trade_pnl = self._calculate_trade_pnl(self.entry_price, sell_price)
                trade_return = trade_pnl / self.position_size

                self.total_pnl += trade_pnl
                self.n_trades += 1

                if trade_return > 0:
                    self.n_wins += 1
                    reward = 1.0 + (trade_return * 10)
                else:
                    reward = max(-1.0, -0.5 + (trade_return * 5))

                self.in_position = False
                self.entry_price = 0.0
                trade_executed = True
            else:
                reward -= 0.01

        # HOLD: No reward/penalty (simplest approach)

        # Advance time
        self.current_step += 1

        # Check termination
        if self.current_step >= self.n_candles - DELAY_SECONDS - 1:
            terminated = True

            # Force close position
            if self.in_position:
                sell_price = self._get_current_price()
                trade_pnl = self._calculate_trade_pnl(self.entry_price, sell_price)
                trade_return = trade_pnl / self.position_size
                self.total_pnl += trade_pnl
                self.n_trades += 1

                if trade_return > 0:
                    self.n_wins += 1
                    reward += 1.0 + (trade_return * 10)
                else:
                    reward += max(-1.0, -0.5 + (trade_return * 5))

            # SIMPLIFIED end-of-episode bonuses
            if self.n_trades == 0:
                reward -= 5.0
            else:
                win_rate = self.n_wins / self.n_trades
                if win_rate >= 0.85:
                    reward += 2.0
                elif win_rate >= 0.70:
                    reward += 1.0

                if self.total_pnl > 0:
                    reward += 1.0

        if self.current_step - MIN_HISTORY_LENGTH >= self.max_steps:
            truncated = True

        self.episode_rewards.append(reward)
        obs = self._get_obs()

        win_rate = self.n_wins / max(1, self.n_trades)
        info = {
            "total_pnl": self.total_pnl,
            "n_trades": self.n_trades,
            "win_rate": win_rate,
            "trade_executed": trade_executed,
            "trade_pnl": trade_pnl,
            "in_position": self.in_position,
            "current_step": self.current_step,
            "missed_opportunities": 0,
            "fee_multiplier": self.fee_multiplier,
            "reward_version": "simplified",
        }

        return obs, reward, terminated, truncated, info
