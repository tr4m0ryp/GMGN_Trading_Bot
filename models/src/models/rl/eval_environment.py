"""
Multi-token evaluation environment for generalization testing.

Contains MultiTokenEvalEnvironment which cycles through multiple tokens
with full fees for realistic evaluation.

Author: Trading Team
Date: 2025-12-24
"""

from typing import Dict, Tuple, List, Any, Optional
import numpy as np
import gymnasium as gym

from .environment import TradingEnvironmentV2


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
            fee_multiplier=1.0,
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
