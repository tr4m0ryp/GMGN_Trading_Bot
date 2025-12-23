"""
Gym-compatible trading environment for reinforcement learning.

This module implements a trading environment where an RL agent learns
to make optimal buy/sell decisions by interacting with historical
price data and receiving profit-based rewards.

Dependencies:
    gymnasium: OpenAI Gym interface
    numpy: Numerical computations
    torch: For feature extraction

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


class TradingEnvironment(gym.Env):
    """
    Trading environment for reinforcement learning.

    The agent observes price history features and decides to BUY, SELL, or HOLD.
    Rewards are based on realized profit/loss from trades, accounting for
    transaction fees and slippage.

    State:
        - Feature vector from price history (14 features per timestep)
        - Current position status (in_position: 0 or 1)
        - Entry price (if in position)
        - Unrealized PnL (if in position)

    Actions:
        0: HOLD - Do nothing
        1: BUY - Enter long position (if not in position)
        2: SELL - Exit position (if in position)

    Rewards:
        - Realized profit/loss on trade close (after fees)
        - Small negative reward for holding too long in position
        - Penalty for invalid actions (trying to buy when already in position)

    Episode ends when:
        - End of price data reached
        - Maximum steps exceeded

    Args:
        candles: List of candle dictionaries with o, h, l, c, v keys.
        max_steps: Maximum steps per episode. Default is None (use all data).
        fee_per_trade: Transaction fee per trade. Default uses config value.
        position_size: Fixed position size in SOL. Default uses config value.
        hold_penalty: Per-step penalty for holding position. Default is 0.0001.
        invalid_action_penalty: Penalty for invalid actions. Default is 0.01.

    Example:
        >>> env = TradingEnvironment(candles)
        >>> obs, info = env.reset()
        >>> action = agent.predict(obs)
        >>> obs, reward, done, truncated, info = env.step(action)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        candles: List[Dict[str, float]],
        max_steps: Optional[int] = None,
        fee_per_trade: float = TOTAL_FEE_PER_TX,
        position_size: float = FIXED_POSITION_SIZE,
        hold_penalty: float = 0.0001,
        invalid_action_penalty: float = 0.01,
    ):
        super().__init__()

        self.candles = candles
        self.n_candles = len(candles)
        self.max_steps = max_steps or (self.n_candles - DELAY_SECONDS - 1)
        self.fee_per_trade = fee_per_trade
        self.position_size = position_size
        self.hold_penalty = hold_penalty
        self.invalid_action_penalty = invalid_action_penalty

        # Pre-extract all features for efficiency
        self.all_features = extract_features(candles)
        self.n_features = self.all_features.shape[1]  # Should be 14

        # Action space: 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(3)

        # Observation space: features + position info
        # [features..., in_position, entry_price_normalized, unrealized_pnl, time_in_position]
        obs_dim = self.n_features + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Episode state
        self.current_step = 0
        self.in_position = False
        self.entry_price = 0.0
        self.entry_step = 0
        self.total_pnl = 0.0
        self.n_trades = 0
        self.n_wins = 0

    def _get_current_price(self) -> float:
        """Get current close price."""
        return self.candles[self.current_step]['c']

    def _get_execution_price(self, is_buy: bool) -> float:
        """Get execution price with slippage simulation."""
        return get_execution_price(self.candles, self.current_step, is_buy)

    def _get_obs(self) -> np.ndarray:
        """
        Construct observation vector.

        Returns:
            Observation array with features and position information.
        """
        # Get features for current step (last row of historical features)
        features = self.all_features[self.current_step].copy()

        # Current price for normalization
        current_price = self._get_current_price()

        # Position information
        in_position = float(self.in_position)

        # Entry price normalized to current price
        entry_price_norm = 0.0
        unrealized_pnl = 0.0
        time_in_position = 0.0

        if self.in_position and self.entry_price > 0:
            entry_price_norm = (self.entry_price - current_price) / current_price
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price
            time_in_position = (self.current_step - self.entry_step) / 100.0

        # Combine features with position info
        obs = np.concatenate([
            features,
            [in_position, entry_price_norm, unrealized_pnl, time_in_position]
        ])

        return obs.astype(np.float32)

    def _calculate_trade_pnl(self, buy_price: float, sell_price: float) -> float:
        """
        Calculate net PnL from a trade including fees.

        Args:
            buy_price: Entry price.
            sell_price: Exit price.

        Returns:
            Net profit/loss in SOL.
        """
        tokens = self.position_size / buy_price
        sell_value = tokens * sell_price
        net_value = sell_value - (2 * self.fee_per_trade)  # Fee on both buy and sell
        return net_value - self.position_size

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Additional options (unused).

        Returns:
            Tuple of (initial_observation, info_dict).
        """
        super().reset(seed=seed)

        # Start after minimum history is available
        self.current_step = MIN_HISTORY_LENGTH + 1
        self.in_position = False
        self.entry_price = 0.0
        self.entry_step = 0
        self.total_pnl = 0.0
        self.n_trades = 0
        self.n_wins = 0

        obs = self._get_obs()
        info = {"total_pnl": 0.0, "n_trades": 0, "win_rate": 0.0}

        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Action to take (0=HOLD, 1=BUY, 2=SELL).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        reward = 0.0
        terminated = False
        truncated = False
        trade_executed = False
        trade_pnl = 0.0

        # Execute action
        if action == 1:  # BUY
            if not self.in_position:
                # Enter position
                self.entry_price = self._get_execution_price(is_buy=True)
                self.entry_step = self.current_step
                self.in_position = True
                trade_executed = True
            else:
                # Invalid: already in position
                reward -= self.invalid_action_penalty

        elif action == 2:  # SELL
            if self.in_position:
                # Exit position
                sell_price = self._get_execution_price(is_buy=False)
                trade_pnl = self._calculate_trade_pnl(self.entry_price, sell_price)

                # Update statistics
                self.total_pnl += trade_pnl
                self.n_trades += 1
                if trade_pnl > 0:
                    self.n_wins += 1

                # Reward is the realized profit/loss
                reward = trade_pnl * 100  # Scale for learning

                # Reset position
                self.in_position = False
                self.entry_price = 0.0
                trade_executed = True
            else:
                # Invalid: not in position
                reward -= self.invalid_action_penalty

        else:  # HOLD
            if self.in_position:
                # Small penalty for holding (encourages active trading)
                reward -= self.hold_penalty

        # Advance time
        self.current_step += 1

        # Check termination conditions
        if self.current_step >= self.n_candles - DELAY_SECONDS - 1:
            terminated = True
            # Force close position at end
            if self.in_position:
                sell_price = self._get_current_price()
                trade_pnl = self._calculate_trade_pnl(self.entry_price, sell_price)
                self.total_pnl += trade_pnl
                self.n_trades += 1
                if trade_pnl > 0:
                    self.n_wins += 1
                reward += trade_pnl * 100

        if self.current_step - MIN_HISTORY_LENGTH >= self.max_steps:
            truncated = True

        # Get new observation
        obs = self._get_obs()

        # Info dictionary
        win_rate = self.n_wins / max(1, self.n_trades)
        info = {
            "total_pnl": self.total_pnl,
            "n_trades": self.n_trades,
            "win_rate": win_rate,
            "trade_executed": trade_executed,
            "trade_pnl": trade_pnl,
            "in_position": self.in_position,
            "current_step": self.current_step,
        }

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """Render current state (for debugging)."""
        current_price = self._get_current_price()
        position_str = "IN POSITION" if self.in_position else "NOT IN POSITION"
        print(f"Step {self.current_step}: Price={current_price:.6f}, "
              f"{position_str}, PnL={self.total_pnl:.6f}")

    def close(self) -> None:
        """Clean up resources."""
        pass


class MultiTokenTradingEnvironment(gym.Env):
    """
    Trading environment that cycles through multiple tokens.

    Provides more diverse training experience by exposing the agent
    to different token price patterns. Each episode uses a different
    token's price history.

    Args:
        all_candles: List of candle lists, one per token.
        max_steps_per_token: Maximum steps per token episode. Default is None.
        **kwargs: Additional arguments passed to TradingEnvironment.

    Example:
        >>> env = MultiTokenTradingEnvironment(all_token_candles)
        >>> obs, info = env.reset()
        >>> # Agent trains across multiple tokens
    """

    def __init__(
        self,
        all_candles: List[List[Dict[str, float]]],
        max_steps_per_token: Optional[int] = None,
        **kwargs
    ):
        super().__init__()

        self.all_candles = all_candles
        self.n_tokens = len(all_candles)
        self.max_steps_per_token = max_steps_per_token
        self.env_kwargs = kwargs

        self.current_token_idx = 0
        self.current_env: Optional[TradingEnvironment] = None

        # Create initial environment to get spaces
        self._create_env(0)
        self.action_space = self.current_env.action_space
        self.observation_space = self.current_env.observation_space

        # Track aggregate statistics
        self.total_episodes = 0
        self.aggregate_pnl = 0.0
        self.aggregate_trades = 0
        self.aggregate_wins = 0

    def _create_env(self, token_idx: int) -> None:
        """Create environment for a specific token."""
        candles = self.all_candles[token_idx]
        self.current_env = TradingEnvironment(
            candles,
            max_steps=self.max_steps_per_token,
            **self.env_kwargs
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset with a new token."""
        super().reset(seed=seed)

        # Randomly select next token
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.current_token_idx = self.np_random.integers(0, self.n_tokens)
        self._create_env(self.current_token_idx)

        self.total_episodes += 1

        return self.current_env.reset(seed=seed, options=options)

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute step in current token environment."""
        obs, reward, terminated, truncated, info = self.current_env.step(action)

        # Update aggregate stats on episode end
        if terminated or truncated:
            self.aggregate_pnl += info.get("total_pnl", 0.0)
            self.aggregate_trades += info.get("n_trades", 0)
            self.aggregate_wins += int(
                info.get("n_trades", 0) * info.get("win_rate", 0.0)
            )

        # Add aggregate stats to info
        info["token_idx"] = self.current_token_idx
        info["total_episodes"] = self.total_episodes
        info["aggregate_pnl"] = self.aggregate_pnl
        info["aggregate_trades"] = self.aggregate_trades

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """Render current state."""
        if self.current_env is not None:
            self.current_env.render()

    def close(self) -> None:
        """Clean up resources."""
        if self.current_env is not None:
            self.current_env.close()
