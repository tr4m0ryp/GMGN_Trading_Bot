"""
Improved RL Training with curriculum learning and better hyperparameters.

This version addresses policy collapse through:
1. Curriculum learning environment
2. Higher entropy coefficient for exploration
3. Better reward shaping
4. Recurrent policy option for temporal dependencies

Author: Trading Team
Date: 2025-12-23
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import numpy as np
import torch

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor

from .environment_v2 import TradingEnvironmentV2, CurriculumTradingEnvironment
from .agent import TradingFeaturesExtractor, AdvancedTradingFeaturesExtractor


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

    all_candles = []
    for idx in range(len(df)):
        try:
            candles = parse_candles(df.iloc[idx]['candles'])
            if len(candles) >= 50:
                all_candles.append(candles)
        except Exception:
            continue

    print(f"Loaded {len(all_candles)} tokens with valid candle data")
    return all_candles


class ImprovedTradingCallback(BaseCallback):
    """
    Enhanced callback with curriculum monitoring.
    """

    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_pnls = []
        self.episode_trades = []
        self.episode_win_rates = []
        self.curriculum_progress = []

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
                print(f"\n[Step {self.n_calls}] "
                      f"PnL: {mean_pnl:.4f} | "
                      f"Trades: {mean_trades:.1f} | "
                      f"WinRate: {mean_win_rate:.1%} | "
                      f"Curriculum: {curr_progress:.0%}")

            if self.logger is not None:
                self.logger.record("trading/mean_pnl", mean_pnl)
                self.logger.record("trading/mean_trades", mean_trades)
                self.logger.record("trading/mean_win_rate", mean_win_rate)
                self.logger.record("trading/curriculum_progress", curr_progress)

        return True


def create_curriculum_envs(
    all_candles: List[List[Dict[str, float]]],
    n_envs: int = 4,
    curriculum_episodes: int = 1000,
) -> DummyVecEnv:
    """Create vectorized curriculum environments."""

    def make_env(seed: int):
        def _init():
            env = CurriculumTradingEnvironment(
                all_candles,
                initial_fee_mult=0.0,
                target_fee_mult=1.0,
                curriculum_episodes=curriculum_episodes,
            )
            env = Monitor(env)
            env.reset(seed=seed)
            return env
        return _init

    return DummyVecEnv([make_env(i) for i in range(n_envs)])


def train_rl_agent_v2(
    data_dir: str,
    output_dir: str,
    total_timesteps: int = 1_000_000,
    learning_rate: float = 1e-4,  # Lower LR for stability
    n_envs: int = 4,
    eval_freq: int = 10000,
    save_freq: int = 50000,
    device: str = 'auto',
    verbose: int = 1,
    curriculum_episodes: int = 1000,
) -> Dict[str, Any]:
    """
    Train RL agent with improved settings.

    Key improvements:
    1. Curriculum learning (fees increase gradually)
    2. Higher entropy coefficient (0.05) for exploration
    3. Smaller clip range (0.1) for stability
    4. Lower learning rate for smoother training
    5. Larger n_steps for better advantage estimation

    Args:
        data_dir: Directory containing data.
        output_dir: Directory for saving models.
        total_timesteps: Total training steps. Default 1M.
        learning_rate: Learning rate. Default 1e-4.
        n_envs: Parallel environments. Default 4.
        curriculum_episodes: Episodes to full difficulty. Default 1000.

    Returns:
        Training results dictionary.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    log_dir = output_path / "logs"
    log_dir.mkdir(exist_ok=True)

    # Load data
    print("Loading token data...")
    all_candles = load_token_candles(data_dir)

    # Split
    np.random.seed(42)
    indices = np.random.permutation(len(all_candles))
    n_train = int(len(all_candles) * 0.9)

    train_candles = [all_candles[i] for i in indices[:n_train]]
    eval_candles = [all_candles[i] for i in indices[n_train:]]

    print(f"Training tokens: {len(train_candles)}")
    print(f"Evaluation tokens: {len(eval_candles)}")

    # Create environments
    print("Creating curriculum training environments...")
    train_env = create_curriculum_envs(
        train_candles,
        n_envs=n_envs,
        curriculum_episodes=curriculum_episodes,
    )

    print("Creating evaluation environment...")
    eval_env = TradingEnvironmentV2(
        eval_candles[0],
        fee_multiplier=1.0,  # Full fees for eval
    )
    eval_env = Monitor(eval_env)

    # PPO with improved hyperparameters
    print("Creating PPO agent with improved hyperparameters...")

    policy_kwargs = {
        "features_extractor_class": AdvancedTradingFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": dict(pi=[256, 128], vf=[256, 128]),
        "activation_fn": torch.nn.GELU,
    }

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        n_steps=4096,  # Larger for better advantage estimation
        batch_size=128,  # Smaller for more frequent updates
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,  # Smaller for stability
        clip_range_vf=0.1,
        ent_coef=0.05,  # HIGHER entropy for exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=verbose,
        tensorboard_log=str(log_dir),
        device=device,
    )

    print(f"\nPPO Hyperparameters:")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  N Steps: 4096")
    print(f"  Batch Size: 128")
    print(f"  Entropy Coef: 0.05 (high for exploration)")
    print(f"  Clip Range: 0.1")
    print(f"  Curriculum Episodes: {curriculum_episodes}")

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_path),
        log_path=str(log_dir),
        eval_freq=eval_freq // n_envs,
        n_eval_episodes=10,
        deterministic=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // n_envs,
        save_path=str(checkpoint_dir),
        name_prefix="rl_trader_v2",
    )

    trading_callback = ImprovedTradingCallback(check_freq=1000, verbose=verbose)

    callbacks = CallbackList([eval_callback, checkpoint_callback, trading_callback])

    # Train
    print(f"\nStarting training for {total_timesteps:,} timesteps...")
    print("=" * 60)

    start_time = datetime.now()
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    training_time = datetime.now() - start_time

    print("=" * 60)
    print(f"Training completed in {training_time}")

    # Final evaluation
    print("\nFinal evaluation...")
    episode_pnls = []
    episode_trades = []
    episode_win_rates = []

    for _ in range(20):
        obs, _ = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated

        episode_pnls.append(info["total_pnl"])
        episode_trades.append(info["n_trades"])
        episode_win_rates.append(info["win_rate"])

    final_metrics = {
        "mean_pnl": float(np.mean(episode_pnls)),
        "std_pnl": float(np.std(episode_pnls)),
        "mean_trades": float(np.mean(episode_trades)),
        "win_rate": float(np.mean(episode_win_rates)),
    }

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Mean PnL: {final_metrics['mean_pnl']:.4f} +/- {final_metrics['std_pnl']:.4f}")
    print(f"Mean Trades: {final_metrics['mean_trades']:.1f}")
    print(f"Win Rate: {final_metrics['win_rate']:.1%}")
    print("=" * 60)

    # Save
    model.save(str(output_path / "final_model_v2"))
    print(f"\nFinal model saved to: {output_path / 'final_model_v2'}")

    results = {
        "algorithm": "ppo_v2",
        "total_timesteps": total_timesteps,
        "learning_rate": learning_rate,
        "training_time": str(training_time),
        "curriculum_episodes": curriculum_episodes,
        "n_training_tokens": len(train_candles),
        "n_eval_tokens": len(eval_candles),
        "final_metrics": final_metrics,
    }

    with open(output_path / "training_results_v2.json", 'w') as f:
        json.dump(results, f, indent=2)

    train_env.close()
    eval_env.close()

    return results
