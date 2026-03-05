"""
Trading environment V5.1 - main TradingEnvironmentV2 class.
Reward calculation logic is in reward_helpers module.
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
from .reward_helpers import calculate_trade_reward, calculate_episode_bonus


class TradingEnvironmentV2(gym.Env):
    """
    Trading environment V5.1: Enhanced profit-maximizing reward system.

    Per-trade rewards guarantee positive minimum (+0.05) for any trade.
    Profits scale via configurable parameters. End-of-episode bonuses
    reward win rate, profit, and trade frequency. Catastrophic penalty
    (-35.0) prevents no-trade collapse. See reward_helpers.py for details.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        candles: List[Dict[str, float]],
        max_steps: Optional[int] = None,
        fee_multiplier: float = 0.5,
        position_size: float = FIXED_POSITION_SIZE,
        opportunity_penalty: float = 0.01,
        min_trades_required: int = 3,
        max_trades_target: int = 5,
        base_trade_reward: float = 0.3,
        profit_scale: float = 30.0,
        win_bonus_base: float = 0.5,
        win_bonus_scale: float = 10.0,
        win_bonus_cap: float = 1.5,
        loss_scale: float = 15.0,
        loss_cap: float = -0.25,
        sweet_spot_bonus: float = 1.0,
        missing_trade_penalty: float = 10.0,
        zero_trade_extra: float = 5.0,
        momentum_reward_scale: float = 0.002,
        synergy_scale: float = 5.0,
        quick_trade_threshold: int = 30,
        quick_trade_bonus: float = 0.3,
        overtrade_penalty_scale: float = 0.3,
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

        # V5.1 Reward System Parameters
        self.base_trade_reward = base_trade_reward
        self.profit_scale = profit_scale
        self.win_bonus_base = win_bonus_base
        self.win_bonus_scale = win_bonus_scale
        self.win_bonus_cap = win_bonus_cap
        self.loss_scale = loss_scale
        self.loss_cap = loss_cap
        self.sweet_spot_bonus = sweet_spot_bonus
        self.missing_trade_penalty = missing_trade_penalty
        self.zero_trade_extra = zero_trade_extra
        self.momentum_reward_scale = momentum_reward_scale
        self.synergy_scale = synergy_scale
        self.quick_trade_threshold = quick_trade_threshold
        self.quick_trade_bonus = quick_trade_bonus
        self.overtrade_penalty_scale = overtrade_penalty_scale

        self.win_rate_bonuses = {
            0.95: 3.0, 0.90: 2.5, 0.85: 2.0, 0.80: 1.5,
            0.70: 1.0, 0.60: 0.5, 0.50: 0.2,
        }

        # Pre-compute features and forward returns
        self.all_features = extract_features(candles)
        self.n_features = self.all_features.shape[1]
        self.forward_returns = self._compute_forward_returns(lookahead=10)

        # Action space: 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(3)
        obs_dim = self.n_features + 5
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
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
        """Get forward return for current step."""
        if self.current_step < len(self.forward_returns):
            return self.forward_returns[self.current_step]
        return 0.0

    def _get_obs(self) -> np.ndarray:
        """Construct observation vector with momentum hint."""
        features = self.all_features[self.current_step].copy()
        current_price = self._get_current_price()
        in_pos = float(self.in_position)
        entry_norm, unreal_pnl, time_in_pos = 0.0, 0.0, 0.0
        if self.in_position and self.entry_price > 0:
            entry_norm = (self.entry_price - current_price) / current_price
            unreal_pnl = (current_price - self.entry_price) / self.entry_price
            time_in_pos = (self.current_step - self.entry_step) / 100.0
        if self.current_step >= 5:
            p0 = self.candles[self.current_step - 5]['c']
            momentum_hint = (current_price - p0) / p0
        else:
            momentum_hint = 0.0
        obs = np.concatenate([
            features, [in_pos, entry_norm, unreal_pnl, time_in_pos, momentum_hint]
        ])
        return obs.astype(np.float32)

    def _calculate_trade_pnl(self, buy_price: float, sell_price: float) -> float:
        """Calculate net PnL from a trade."""
        tokens = self.position_size / buy_price
        sell_value = tokens * sell_price
        net_value = sell_value - (2 * self.fee_per_trade)
        return net_value - self.position_size

    def _is_good_opportunity(self, threshold: float = 0.03) -> bool:
        """Check if current step is a good buying opportunity."""
        return self._get_forward_return() >= threshold

    def reset(self, *, seed=None, options=None):
        """Reset environment."""
        super().reset(seed=seed)
        self._reset_state()
        obs = self._get_obs()
        info = {"total_pnl": 0.0, "n_trades": 0, "win_rate": 0.0}
        return obs, info

    def _get_trade_reward_params(self) -> dict:
        """Return reward parameters dict for calculate_trade_reward."""
        return dict(
            base_trade_reward=self.base_trade_reward,
            profit_scale=self.profit_scale,
            win_bonus_base=self.win_bonus_base,
            win_bonus_scale=self.win_bonus_scale,
            win_bonus_cap=self.win_bonus_cap,
            loss_scale=self.loss_scale,
            loss_cap=self.loss_cap,
            quick_trade_threshold=self.quick_trade_threshold,
            quick_trade_bonus=self.quick_trade_bonus,
        )

    def step(self, action: int):
        """Execute one step with V5.1 reward shaping."""
        reward = 0.0
        terminated = False
        truncated = False
        trade_executed = False
        trade_pnl = 0.0
        is_opportunity = self._is_good_opportunity()
        reward_params = self._get_trade_reward_params()

        if action == 1 and not self.in_position:  # BUY
            self.entry_price = self._get_execution_price(is_buy=True)
            self.entry_step = self.current_step
            self.in_position = True
            trade_executed = True
            reward += 0.001
        elif action == 1 and self.in_position:
            reward -= 0.005
        elif action == 2 and self.in_position:  # SELL
            sell_price = self._get_execution_price(is_buy=False)
            trade_pnl = self._calculate_trade_pnl(self.entry_price, sell_price)
            trade_return = trade_pnl / self.position_size
            self.total_pnl += trade_pnl
            self.n_trades += 1
            is_win = trade_return > 0
            if is_win:
                self.n_wins += 1
            hold_duration = self.current_step - self.entry_step
            reward = calculate_trade_reward(
                trade_return=trade_return, hold_duration=hold_duration,
                is_win=is_win, **reward_params
            )
            self.in_position = False
            self.entry_price = 0.0
            trade_executed = True
        elif action == 2 and not self.in_position:
            reward -= 0.005
        else:  # HOLD
            if not self.in_position and is_opportunity:
                self.missed_opportunities += 1
                reward -= self.opportunity_penalty
            if self.in_position:
                current_price = self._get_current_price()
                unrealized_return = (current_price - self.entry_price) / self.entry_price
                if unrealized_return > 0:
                    reward += unrealized_return * self.momentum_reward_scale
                else:
                    reward += unrealized_return * self.momentum_reward_scale * 0.5

        self.current_step += 1

        if self.current_step >= self.n_candles - DELAY_SECONDS - 1:
            terminated = True
            if self.in_position:
                sell_price = self._get_current_price()
                trade_pnl = self._calculate_trade_pnl(self.entry_price, sell_price)
                trade_return = trade_pnl / self.position_size
                self.total_pnl += trade_pnl
                self.n_trades += 1
                is_win = trade_return > 0
                if is_win:
                    self.n_wins += 1
                reward += calculate_trade_reward(
                    trade_return=trade_return, hold_duration=0,
                    is_win=is_win, **reward_params
                )
            reward += calculate_episode_bonus(
                n_trades=self.n_trades, n_wins=self.n_wins,
                total_pnl=self.total_pnl,
                min_trades_required=self.min_trades_required,
                max_trades_target=self.max_trades_target,
                sweet_spot_bonus=self.sweet_spot_bonus,
                overtrade_penalty_scale=self.overtrade_penalty_scale,
                win_rate_bonuses=self.win_rate_bonuses,
                zero_trade_extra=self.zero_trade_extra,
                missing_trade_penalty=self.missing_trade_penalty,
                synergy_scale=self.synergy_scale,
            )

        if self.current_step - MIN_HISTORY_LENGTH >= self.max_steps:
            truncated = True

        self.episode_rewards.append(reward)
        obs = self._get_obs()
        win_rate = self.n_wins / max(1, self.n_trades)
        info = {
            "total_pnl": self.total_pnl, "n_trades": self.n_trades,
            "win_rate": win_rate, "trade_executed": trade_executed,
            "trade_pnl": trade_pnl, "in_position": self.in_position,
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
