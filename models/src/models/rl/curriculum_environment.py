"""
Curriculum learning trading environment.

Contains CurriculumTradingEnvironment which automatically increases
difficulty (fee level) as the agent improves during training.

Author: Trading Team
Date: 2025-12-24
"""

from typing import Dict, Tuple, List, Any, Optional
import numpy as np
import gymnasium as gym

from .environment import TradingEnvironmentV2


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
        return self.initial_fee_mult + progress * (
            self.target_fee_mult - self.initial_fee_mult
        )

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
        info["curriculum_progress"] = min(
            1.0, self.total_episodes / self.curriculum_episodes
        )
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

        info["curriculum_progress"] = min(
            1.0, self.total_episodes / self.curriculum_episodes
        )
        info["current_fee_mult"] = self.current_fee_mult
        info["avg_recent_win_rate"] = (
            np.mean(self.recent_win_rates) if self.recent_win_rates else 0
        )
        info["avg_recent_pnl"] = (
            np.mean(self.recent_pnls) if self.recent_pnls else 0
        )

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """Render."""
        if self.current_env:
            self.current_env.render()

    def close(self) -> None:
        """Clean up."""
        if self.current_env:
            self.current_env.close()
