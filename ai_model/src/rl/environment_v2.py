"""
Improved trading environment with advanced reward shaping.

This version addresses policy collapse by:
1. Hindsight rewards - penalize for missing profitable opportunities
2. Potential-based reward shaping - smooth learning signal
3. Curriculum learning - gradually introduce fees
4. Exploration bonuses - encourage trying actions

Author: Trading Team
Date: 2025-12-23
"""

from typing import Dict, Tuple, List, Any, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import (
    FIXED_POSITION_SIZE,
    DELAY_SECONDS,
    TOTAL_FEE_PER_TX,
    MIN_HISTORY_LENGTH,
)
from data.preparation import extract_features, get_execution_price


class TradingEnvironmentV2(gym.Env):
    """
    Improved trading environment with hindsight rewards.

    Key improvements over V1:
    1. **Hindsight rewards**: Penalizes missing profitable moves
    2. **Potential-based shaping**: Rewards based on price momentum
    3. **Curriculum learning**: Fee multiplier starts at 0, increases over time
    4. **Minimum trade requirement**: Episode reward bonus for active trading
    5. **Asymmetric rewards**: Bigger bonus for wins than penalty for losses

    The goal is to prevent the agent from learning to "do nothing".

    Args:
        candles: List of candle dictionaries.
        max_steps: Maximum steps per episode.
        fee_multiplier: Fee scaling (0=no fees, 1=full fees). Use for curriculum.
        opportunity_penalty: Penalty for missing profitable moves. Default 0.01.
        min_trades_bonus: Bonus if agent makes at least N trades. Default 0.1.
        min_trades_required: Minimum trades for bonus. Default 3.
        win_bonus_multiplier: Extra multiplier for winning trades. Default 1.5.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        candles: List[Dict[str, float]],
        max_steps: Optional[int] = None,
        fee_multiplier: float = 0.5,  # Start with half fees
        position_size: float = FIXED_POSITION_SIZE,
        opportunity_penalty: float = 0.005,  # Penalty for missing opportunities
        min_trades_bonus: float = 0.05,  # Bonus for meeting trade minimum
        min_trades_required: int = 2,  # Minimum trades to get bonus
        win_bonus_multiplier: float = 1.5,  # Extra reward for wins
        momentum_reward_scale: float = 0.01,  # Reward for riding momentum
    ):
        super().__init__()

        self.candles = candles
        self.n_candles = len(candles)
        self.max_steps = max_steps or (self.n_candles - DELAY_SECONDS - 1)
        self.fee_multiplier = fee_multiplier
        self.fee_per_trade = TOTAL_FEE_PER_TX * fee_multiplier
        self.position_size = position_size
        self.opportunity_penalty = opportunity_penalty
        self.min_trades_bonus = min_trades_bonus
        self.min_trades_required = min_trades_required
        self.win_bonus_multiplier = win_bonus_multiplier
        self.momentum_reward_scale = momentum_reward_scale

        # Pre-compute price changes for hindsight rewards
        self.all_features = extract_features(candles)
        self.n_features = self.all_features.shape[1]

        # Pre-compute forward returns for opportunity detection
        self.forward_returns = self._compute_forward_returns(lookahead=10)

        # Action space: 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(3)

        # Observation includes forward-looking hint (normalized momentum)
        obs_dim = self.n_features + 5  # +1 for momentum hint
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Episode state
        self._reset_state()

    def _compute_forward_returns(self, lookahead: int = 10) -> np.ndarray:
        """Pre-compute forward returns for opportunity detection."""
        closes = np.array([c['c'] for c in self.candles])
        forward_returns = np.zeros(len(closes))

        for i in range(len(closes) - lookahead):
            future_max = np.max(closes[i+1:i+lookahead+1])
            forward_returns[i] = (future_max - closes[i]) / closes[i]

        return forward_returns

    def _reset_state(self):
        """Reset episode state."""
        self.current_step = MIN_HISTORY_LENGTH + 1
        self.in_position = False
        self.entry_price = 0.0
        self.entry_step = 0
        self.total_pnl = 0.0
        self.n_trades = 0
        self.n_wins = 0
        self.missed_opportunities = 0
        self.episode_rewards = []

    def _get_current_price(self) -> float:
        """Get current close price."""
        return self.candles[self.current_step]['c']

    def _get_execution_price(self, is_buy: bool) -> float:
        """Get execution price with slippage."""
        return get_execution_price(self.candles, self.current_step, is_buy)

    def _get_forward_return(self) -> float:
        """Get forward return for current step (opportunity indicator)."""
        if self.current_step < len(self.forward_returns):
            return self.forward_returns[self.current_step]
        return 0.0

    def _get_obs(self) -> np.ndarray:
        """Construct observation vector with momentum hint."""
        features = self.all_features[self.current_step].copy()
        current_price = self._get_current_price()

        in_position = float(self.in_position)
        entry_price_norm = 0.0
        unrealized_pnl = 0.0
        time_in_position = 0.0

        if self.in_position and self.entry_price > 0:
            entry_price_norm = (self.entry_price - current_price) / current_price
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price
            time_in_position = (self.current_step - self.entry_step) / 100.0

        # Add momentum hint (smoothed recent return)
        if self.current_step >= 5:
            recent_prices = [self.candles[i]['c'] for i in range(self.current_step-5, self.current_step)]
            momentum_hint = (current_price - recent_prices[0]) / recent_prices[0]
        else:
            momentum_hint = 0.0

        obs = np.concatenate([
            features,
            [in_position, entry_price_norm, unrealized_pnl, time_in_position, momentum_hint]
        ])

        return obs.astype(np.float32)

    def _calculate_trade_pnl(self, buy_price: float, sell_price: float) -> float:
        """Calculate net PnL from a trade."""
        tokens = self.position_size / buy_price
        sell_value = tokens * sell_price
        net_value = sell_value - (2 * self.fee_per_trade)
        return net_value - self.position_size

    def _is_good_opportunity(self, threshold: float = 0.02) -> bool:
        """Check if current step is a good buying opportunity."""
        return self._get_forward_return() >= threshold

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        super().reset(seed=seed)
        self._reset_state()

        obs = self._get_obs()
        info = {"total_pnl": 0.0, "n_trades": 0, "win_rate": 0.0}

        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step with improved reward shaping."""
        reward = 0.0
        terminated = False
        truncated = False
        trade_executed = False
        trade_pnl = 0.0

        prev_price = self._get_current_price()
        is_opportunity = self._is_good_opportunity()

        # Execute action
        if action == 1:  # BUY
            if not self.in_position:
                self.entry_price = self._get_execution_price(is_buy=True)
                self.entry_step = self.current_step
                self.in_position = True
                trade_executed = True

                # Small positive reward for taking action (exploration)
                reward += 0.001
            else:
                # Penalty for invalid action
                reward -= 0.005

        elif action == 2:  # SELL
            if self.in_position:
                sell_price = self._get_execution_price(is_buy=False)
                trade_pnl = self._calculate_trade_pnl(self.entry_price, sell_price)

                self.total_pnl += trade_pnl
                self.n_trades += 1

                if trade_pnl > 0:
                    self.n_wins += 1
                    # Asymmetric: bigger bonus for wins
                    reward = trade_pnl * 100 * self.win_bonus_multiplier
                else:
                    # Standard penalty for losses
                    reward = trade_pnl * 100

                self.in_position = False
                self.entry_price = 0.0
                trade_executed = True
            else:
                reward -= 0.005

        else:  # HOLD
            # Hindsight penalty for missing opportunities
            if not self.in_position and is_opportunity:
                self.missed_opportunities += 1
                reward -= self.opportunity_penalty

            # Momentum reward when in position and price moving favorably
            if self.in_position:
                current_price = self._get_current_price()
                unrealized_return = (current_price - self.entry_price) / self.entry_price
                # Small reward for holding profitable positions
                if unrealized_return > 0:
                    reward += unrealized_return * self.momentum_reward_scale
                # Small penalty for holding losing positions too long
                else:
                    reward += unrealized_return * self.momentum_reward_scale * 0.5

        # Advance time
        self.current_step += 1

        # Check termination
        if self.current_step >= self.n_candles - DELAY_SECONDS - 1:
            terminated = True

            # Force close position
            if self.in_position:
                sell_price = self._get_current_price()
                trade_pnl = self._calculate_trade_pnl(self.entry_price, sell_price)
                self.total_pnl += trade_pnl
                self.n_trades += 1
                if trade_pnl > 0:
                    self.n_wins += 1
                    reward += trade_pnl * 100 * self.win_bonus_multiplier
                else:
                    reward += trade_pnl * 100

            # End-of-episode bonus for active trading
            if self.n_trades >= self.min_trades_required:
                reward += self.min_trades_bonus
            else:
                # Penalty for being too passive
                reward -= self.min_trades_bonus * 0.5

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
            "missed_opportunities": self.missed_opportunities,
            "fee_multiplier": self.fee_multiplier,
        }

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """Render current state."""
        current_price = self._get_current_price()
        position_str = "IN POSITION" if self.in_position else "NOT IN POSITION"
        print(f"Step {self.current_step}: Price={current_price:.6f}, "
              f"{position_str}, PnL={self.total_pnl:.6f}, Trades={self.n_trades}")

    def close(self) -> None:
        """Clean up."""
        pass


class CurriculumTradingEnvironment(gym.Env):
    """
    Trading environment with curriculum learning.

    Automatically increases difficulty (fee level) as agent improves.
    Starts with no fees, gradually increases to full fees.

    Args:
        all_candles: List of candle lists for all tokens.
        initial_fee_mult: Starting fee multiplier. Default 0.0.
        target_fee_mult: Target fee multiplier. Default 1.0.
        curriculum_episodes: Episodes to reach target. Default 500.
    """

    def __init__(
        self,
        all_candles: List[List[Dict[str, float]]],
        initial_fee_mult: float = 0.0,
        target_fee_mult: float = 1.0,
        curriculum_episodes: int = 500,
        **kwargs
    ):
        super().__init__()

        self.all_candles = all_candles
        self.n_tokens = len(all_candles)
        self.initial_fee_mult = initial_fee_mult
        self.target_fee_mult = target_fee_mult
        self.curriculum_episodes = curriculum_episodes
        self.env_kwargs = kwargs

        self.total_episodes = 0
        self.current_fee_mult = initial_fee_mult
        self.current_env = None

        # Create initial env
        self._create_env(0)
        self.action_space = self.current_env.action_space
        self.observation_space = self.current_env.observation_space

        # Tracking
        self.recent_win_rates = []
        self.recent_pnls = []

    def _get_current_fee_mult(self) -> float:
        """Calculate current fee multiplier based on progress."""
        progress = min(1.0, self.total_episodes / self.curriculum_episodes)
        return self.initial_fee_mult + progress * (self.target_fee_mult - self.initial_fee_mult)

    def _create_env(self, token_idx: int) -> None:
        """Create environment with current curriculum settings."""
        candles = self.all_candles[token_idx]
        self.current_fee_mult = self._get_current_fee_mult()

        self.current_env = TradingEnvironmentV2(
            candles,
            fee_multiplier=self.current_fee_mult,
            **self.env_kwargs
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset with curriculum progression."""
        super().reset(seed=seed)

        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        token_idx = self.np_random.integers(0, self.n_tokens)
        self._create_env(token_idx)
        self.total_episodes += 1

        obs, info = self.current_env.reset(seed=seed, options=options)
        info["curriculum_progress"] = min(1.0, self.total_episodes / self.curriculum_episodes)
        info["current_fee_mult"] = self.current_fee_mult

        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute step."""
        obs, reward, terminated, truncated, info = self.current_env.step(action)

        if terminated or truncated:
            self.recent_win_rates.append(info.get("win_rate", 0))
            self.recent_pnls.append(info.get("total_pnl", 0))

            # Keep only last 100 episodes
            self.recent_win_rates = self.recent_win_rates[-100:]
            self.recent_pnls = self.recent_pnls[-100:]

        info["curriculum_progress"] = min(1.0, self.total_episodes / self.curriculum_episodes)
        info["current_fee_mult"] = self.current_fee_mult
        info["avg_recent_win_rate"] = np.mean(self.recent_win_rates) if self.recent_win_rates else 0
        info["avg_recent_pnl"] = np.mean(self.recent_pnls) if self.recent_pnls else 0

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """Render."""
        if self.current_env:
            self.current_env.render()

    def close(self) -> None:
        """Clean up."""
        if self.current_env:
            self.current_env.close()
