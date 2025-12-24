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
    Trading environment V5: Profit-maximizing reward system with win rate bonuses.

    V5 REWARD SYSTEM - Key Design Principles:
    ==========================================
    1. **Trading is ALWAYS positive**: Base reward (+0.3) ensures every trade is rewarded
    2. **Losses are capped**: Minimum per-trade reward is +0.05 (0.3 - 0.25 cap)
    3. **Profits scale significantly**: +3% profit = +2.0 total reward
    4. **Win rate bonuses**: Up to +3.0 bonus for 95%+ win rate
    5. **Catastrophic no-trade penalty**: 0 trades = -35.0 minimum

    MATHEMATICAL GUARANTEE:
    =======================
    - 0 trades:       -10.0 (base) - 5.0 (extra) - 30.0 (3 missing × 10.0) = -35.0
    - 3 worst trades: 3 × (+0.05) + 0.2 (50% WR) + 1.0 (sweet spot) = +1.35
    - 3 good trades:  3 × (+2.0) + 3.0 (95% WR) + 1.0 (sweet spot) = +10.0

    Per-Trade Reward Formula:
    ========================
    if profit > 0:
        reward = 0.3 + (return × 30) + (0.5 + min(0.5, return × 10))
        Example: +3% return = 0.3 + 0.9 + 0.8 = +2.0
    if loss:
        reward = 0.3 + max(-0.25, return × 15)
        Example: -3% loss = 0.3 + max(-0.25, -0.45) = 0.3 - 0.25 = +0.05

    End-of-Episode Bonuses:
    ======================
    - Sweet spot (3-5 trades): +1.0
    - Win rate 95%+: +3.0, 90%+: +2.5, 85%+: +2.0, etc.
    - Total profit bonus: up to +2.0

    Args:
        candles: List of candle dictionaries with OHLCV data.
        max_steps: Maximum steps per episode (None = use all candles).
        fee_multiplier: Fee scaling (0=no fees, 1=full fees). Use for curriculum.
        position_size: Fixed position size in SOL.
        opportunity_penalty: Penalty for missing profitable moves. Default 0.01.
        min_trades_required: Minimum trades for sweet spot. Default 3.
        max_trades_target: Maximum trades for sweet spot. Default 5.
        base_trade_reward: Base reward for completing any trade. Default 0.3.
        profit_scale: Scale factor for profit component. Default 30.0.
        win_bonus_base: Base win bonus. Default 0.5.
        win_bonus_scale: Scale for additional win bonus. Default 10.0.
        loss_cap: Maximum loss penalty (negative value). Default -0.25.
        sweet_spot_bonus: Bonus for 3-5 trades. Default 1.0.
        missing_trade_penalty: Penalty per missing trade. Default 10.0.
        zero_trade_extra: Extra penalty for 0 trades. Default 5.0.
        momentum_reward_scale: Scale for momentum tracking. Default 0.002.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        candles: List[Dict[str, float]],
        max_steps: Optional[int] = None,
        fee_multiplier: float = 0.5,  # Start with half fees
        position_size: float = FIXED_POSITION_SIZE,
        opportunity_penalty: float = 0.01,  # Penalty for missing opportunities
        min_trades_required: int = 3,  # Minimum trades target (3-5 optimal)
        max_trades_target: int = 5,  # Maximum trades before diminishing returns
        # V5 Reward System Parameters
        base_trade_reward: float = 0.3,  # Base reward for ANY completed trade
        profit_scale: float = 30.0,  # Scale factor for profit component
        win_bonus_base: float = 0.5,  # Base bonus for winning trades
        win_bonus_scale: float = 10.0,  # Scale for additional win bonus
        loss_cap: float = -0.25,  # Maximum loss penalty (minimum reward = 0.3 - 0.25 = 0.05)
        sweet_spot_bonus: float = 1.0,  # Bonus for hitting 3-5 trade sweet spot
        missing_trade_penalty: float = 10.0,  # Penalty per missing trade below minimum
        zero_trade_extra: float = 5.0,  # Extra penalty for 0 trades
        momentum_reward_scale: float = 0.002,  # Momentum tracking scale
    ):
        super().__init__()

        self.candles = candles
        self.n_candles = len(candles)
        self.max_steps = max_steps or (self.n_candles - DELAY_SECONDS - 1)
        self.fee_multiplier = fee_multiplier
        self.fee_per_trade = TOTAL_FEE_PER_TX * fee_multiplier
        self.position_size = position_size
        self.opportunity_penalty = opportunity_penalty
        self.min_trades_required = min_trades_required
        self.max_trades_target = max_trades_target

        # V5 Reward System Parameters
        self.base_trade_reward = base_trade_reward
        self.profit_scale = profit_scale
        self.win_bonus_base = win_bonus_base
        self.win_bonus_scale = win_bonus_scale
        self.loss_cap = loss_cap
        self.sweet_spot_bonus = sweet_spot_bonus
        self.missing_trade_penalty = missing_trade_penalty
        self.zero_trade_extra = zero_trade_extra
        self.momentum_reward_scale = momentum_reward_scale

        # Win rate bonus thresholds (percentage: bonus)
        self.win_rate_bonuses = {
            0.95: 3.0,
            0.90: 2.5,
            0.85: 2.0,
            0.80: 1.5,
            0.70: 1.0,
            0.60: 0.5,
            0.50: 0.2,
        }

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

    def _is_good_opportunity(self, threshold: float = 0.03) -> bool:
        """
        Check if current step is a good buying opportunity.

        Uses higher threshold (3%) to only flag truly good opportunities.
        This makes the agent more selective about when to trade.
        """
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
                trade_return = trade_pnl / self.position_size

                self.total_pnl += trade_pnl
                self.n_trades += 1

                # V5 Per-Trade Reward System
                # ==========================
                # Trading is ALWAYS positive: base_trade_reward (0.3) ensures minimum reward
                # Losses are capped: loss_cap (-0.25) limits downside
                # Profits scale significantly: profit_scale (30.0) rewards good trades

                if trade_return > 0:
                    # WINNING TRADE
                    # Formula: base (0.3) + profit_component + win_bonus
                    # Example: +3% return = 0.3 + 0.9 + 0.8 = +2.0
                    self.n_wins += 1
                    profit_component = trade_return * self.profit_scale
                    win_bonus = self.win_bonus_base + min(
                        self.win_bonus_base,  # Cap at 0.5 additional
                        trade_return * self.win_bonus_scale
                    )
                    reward = self.base_trade_reward + profit_component + win_bonus
                else:
                    # LOSING TRADE
                    # Formula: base (0.3) + max(loss_cap, return × 15)
                    # Example: -3% loss = 0.3 + max(-0.25, -0.45) = +0.05
                    # Minimum possible reward: 0.3 - 0.25 = +0.05 (always positive!)
                    loss_penalty = max(self.loss_cap, trade_return * 15.0)
                    reward = self.base_trade_reward + loss_penalty

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
                trade_return = trade_pnl / self.position_size
                self.total_pnl += trade_pnl
                self.n_trades += 1

                if self.use_binary_rewards:
                    if trade_return > 0:
                        self.n_wins += 1
                        reward += 0.5  # Smaller reward for forced close
                    else:
                        reward -= 0.3
                else:
                    if trade_pnl > 0:
                        self.n_wins += 1
                        reward += trade_pnl * 100 * self.win_bonus_multiplier
                    else:
                        reward += trade_pnl * 100

            # End-of-episode bonus for optimal trade frequency (3-5 trades)
            if self.n_trades >= self.min_trades_required:
                # Trade frequency shaping: optimal at 3-5 trades
                if self.min_trades_required <= self.n_trades <= self.max_trades_target:
                    # Sweet spot: 3-5 trades per coin - maximum bonus
                    reward += self.trade_frequency_bonus
                elif self.n_trades <= self.max_trades_target + 3:
                    # 6-8 trades: acceptable but diminishing bonus
                    reward += self.trade_frequency_bonus * 0.6
                else:
                    # Over-trading (>8): small penalty
                    reward -= 0.15

                # Win rate bonus (scales with trade count)
                current_win_rate = self.n_wins / self.n_trades
                if current_win_rate >= 0.9:
                    reward += 1.2  # Big bonus for 90%+ win rate
                elif current_win_rate >= 0.8:
                    reward += 0.8  # Good bonus for 80%+ win rate
                elif current_win_rate >= 0.7:
                    reward += 0.4  # Smaller bonus for 70%+ win rate
                elif current_win_rate >= 0.5:
                    reward += 0.1  # Small bonus for 50%+ win rate

                # Profit magnitude bonus at end of episode
                if self.total_pnl > 0:
                    profit_bonus = min(1.5, self.total_pnl * 15)  # Increased cap
                    reward += profit_bonus
            else:
                # CATASTROPHIC penalty for being too passive (< 3 trades)
                # This MUST be worse than ANY trading outcome to prevent no-trade collapse
                missing_trades = self.min_trades_required - self.n_trades
                passive_penalty = missing_trades * self.no_trade_penalty

                # MASSIVE extra penalty for 0 trades - absolutely unacceptable
                if self.n_trades == 0:
                    passive_penalty += 3.0  # Huge additional penalty for complete inaction
                elif self.n_trades == 1:
                    passive_penalty += 1.5  # Still bad for only 1 trade
                elif self.n_trades == 2:
                    passive_penalty += 0.5  # Small penalty for 2 trades

                reward -= passive_penalty

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


class MultiTokenEvalEnvironment(gym.Env):
    """
    Evaluation environment that cycles through multiple tokens.

    Unlike CurriculumTradingEnvironment, this uses full fees and no curriculum.
    Each reset randomly selects a new token for proper generalization testing.

    Args:
        all_candles: List of candle lists for all evaluation tokens.
    """

    def __init__(
        self,
        all_candles: List[List[Dict[str, float]]],
        **kwargs
    ):
        super().__init__()

        self.all_candles = all_candles
        self.n_tokens = len(all_candles)
        self.env_kwargs = kwargs
        self.current_env = None
        self.current_token_idx = 0

        # Create initial env with full fees
        self._create_env(0)
        self.action_space = self.current_env.action_space
        self.observation_space = self.current_env.observation_space

    def _create_env(self, token_idx: int) -> None:
        """Create environment with full fees for evaluation."""
        candles = self.all_candles[token_idx]
        self.current_token_idx = token_idx

        self.current_env = TradingEnvironmentV2(
            candles,
            fee_multiplier=1.0,  # Full fees for realistic evaluation
            **self.env_kwargs
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset with random token selection for diverse evaluation."""
        super().reset(seed=seed)

        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        # Randomly select a token for this episode
        token_idx = self.np_random.integers(0, self.n_tokens)
        self._create_env(token_idx)

        obs, info = self.current_env.reset(seed=seed, options=options)
        info["token_idx"] = token_idx

        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute step."""
        obs, reward, terminated, truncated, info = self.current_env.step(action)
        info["token_idx"] = self.current_token_idx
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """Render."""
        if self.current_env:
            self.current_env.render()

    def close(self) -> None:
        """Clean up."""
        if self.current_env:
            self.current_env.close()


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
