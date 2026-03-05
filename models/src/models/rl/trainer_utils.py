"""
RL Training Utilities: Data loading, callbacks, and environment creation.

Provides helper functions for the RL training pipeline including
data loading, custom callbacks, and environment setup.

Author: Trading Team
Date: 2025-12-23
"""

from pathlib import Path
from typing import Dict, List, Any

import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from .environment import (
    TradingEnvironmentV2,
    TradingEnvironmentSimplified,
    CurriculumTradingEnvironment,
)


def load_token_candles(data_dir: str) -> List[List[Dict[str, float]]]:
    """Load candle data for all tokens."""
    import pandas as pd
    from data.preparation import parse_candles

    csv_path = Path(data_dir) / "raw" / "rawdata.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw data not found: {csv_path}")

    df = pd.read_csv(
        csv_path,
        quotechar='"',
        escapechar='\\',
        on_bad_lines='warn',
        engine='python'
    )

    # Detect correct column name (chart_data_json for new data, candles for old)
    candle_col = 'chart_data_json' if 'chart_data_json' in df.columns else 'candles'
    print(f"Using candle column: {candle_col}")

    all_candles = []
    for idx in range(len(df)):
        try:
            candles = parse_candles(df.iloc[idx][candle_col])
            if len(candles) >= 50:
                all_candles.append(candles)
        except Exception:
            continue

    print(f"Loaded {len(all_candles)} tokens with valid candle data")
    return all_candles


class ImprovedTradingCallback(BaseCallback):
    """
    Enhanced callback with curriculum monitoring - Compact output.
    """

    def __init__(self, check_freq: int = 5000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_pnls = []
        self.episode_trades = []
        self.episode_win_rates = []
        self.curriculum_progress = []
        self.last_print_step = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "total_pnl" in info:
                self.episode_pnls.append(info["total_pnl"])
                self.episode_trades.append(info.get("n_trades", 0))
                self.episode_win_rates.append(info.get("win_rate", 0.0))
                self.curriculum_progress.append(info.get("curriculum_progress", 1.0))

        if self.n_calls % self.check_freq == 0 and len(self.episode_pnls) > 0:
            mean_pnl = np.mean(self.episode_pnls[-100:])
            mean_trades = np.mean(self.episode_trades[-100:])
            mean_win_rate = np.mean(self.episode_win_rates[-100:])
            curr_progress = np.mean(self.curriculum_progress[-100:]) if self.curriculum_progress else 0

            if self.verbose >= 1:
                print(f"[{self.n_calls:7d}] WR:{mean_win_rate:5.1%} PnL:{mean_pnl:+.4f} Trades:{mean_trades:.1f} Curr:{curr_progress:.0%}")

            if self.logger is not None:
                self.logger.record("trading/mean_pnl", mean_pnl)
                self.logger.record("trading/mean_trades", mean_trades)
                self.logger.record("trading/mean_win_rate", mean_win_rate)
                self.logger.record("trading/curriculum_progress", curr_progress)

        return True


def create_curriculum_envs(
    all_candles: List[List[Dict[str, float]]],
    n_envs: int = 8,
    curriculum_episodes: int = 1000,
    use_subproc: bool = False,
    use_simplified_reward: bool = False,
) -> DummyVecEnv:
    """
    Create vectorized curriculum environments.

    Uses DummyVecEnv (single-process) by default to avoid memory duplication.
    SubprocVecEnv creates separate processes, each loading a copy of all data,
    causing massive RAM usage (32 envs = 32x data copies).

    Args:
        all_candles: List of candle data for all tokens.
        n_envs: Number of parallel environments. Default 8 (reduced from 16).
        curriculum_episodes: Episodes to reach full difficulty.
        use_subproc: Use SubprocVecEnv (True) or DummyVecEnv (False).
        use_simplified_reward: Use simplified reward system.

    Returns:
        Vectorized environment for training.
    """

    def make_env(seed: int):
        def _init():
            BaseEnvClass = TradingEnvironmentSimplified if use_simplified_reward else TradingEnvironmentV2

            class CurriculumEnv(CurriculumTradingEnvironment):
                def _create_env(self, token_idx: int) -> None:
                    candles = self.all_candles[token_idx]
                    self.current_token_idx = token_idx
                    self.current_fee_mult = self._get_current_fee_mult()
                    self.current_env = BaseEnvClass(
                        candles,
                        fee_multiplier=self.current_fee_mult,
                        **self.env_kwargs
                    )

            env = CurriculumEnv(
                all_candles,
                initial_fee_mult=0.0,
                target_fee_mult=1.0,
                curriculum_episodes=curriculum_episodes,
            )
            env = Monitor(env)
            env.reset(seed=seed)
            return env
        return _init

    if use_subproc and n_envs > 1:
        print(f"[WARN] SubprocVecEnv uses {n_envs}x RAM. Consider use_subproc=False.")
        return SubprocVecEnv([make_env(i) for i in range(n_envs)])
    else:
        return DummyVecEnv([make_env(i) for i in range(n_envs)])
