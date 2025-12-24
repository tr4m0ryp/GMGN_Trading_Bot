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

# Try to import RecurrentPPO from sb3-contrib for LSTM support
try:
    from sb3_contrib import RecurrentPPO
    RECURRENT_AVAILABLE = True
except ImportError:
    RECURRENT_AVAILABLE = False
    print("Warning: sb3-contrib not installed. RecurrentPPO unavailable.")
    print("Install with: pip install sb3-contrib")
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor

from .environment import TradingEnvironmentV2, CurriculumTradingEnvironment, MultiTokenEvalEnvironment
from .agent import (
    TradingFeaturesExtractor,
    AdvancedTradingFeaturesExtractor,
    HybridLSTMAttentionExtractor,
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
                # Compact single-line output
                print(f"[{self.n_calls:7d}] WR:{mean_win_rate:5.1%} PnL:{mean_pnl:+.4f} Trades:{mean_trades:.1f} Curr:{curr_progress:.0%}")

            if self.logger is not None:
                self.logger.record("trading/mean_pnl", mean_pnl)
                self.logger.record("trading/mean_trades", mean_trades)
                self.logger.record("trading/mean_win_rate", mean_win_rate)
                self.logger.record("trading/curriculum_progress", curr_progress)

        return True


def create_curriculum_envs(
    all_candles: List[List[Dict[str, float]]],
    n_envs: int = 64,
    curriculum_episodes: int = 2000,
    use_subproc: bool = True,  # CPU-optimized: use SubprocVecEnv for better parallelization
) -> DummyVecEnv:
    """
    Create vectorized curriculum environments.

    CPU Optimization (50GB RAM):
    - SubprocVecEnv (64 envs): Each process loads data once, 64x parallel throughput
    - With 50GB RAM, we can safely support 64 parallel environments
    - Data replication is negligible with 64 cores processing simultaneously

    GPU (14GB constraint):
    - Would use DummyVecEnv (8 envs) to avoid memory duplication
    - But CPU has no such constraint with 50GB available

    Args:
        all_candles: List of candle data for all tokens.
        n_envs: Number of parallel environments. Default 64 (CPU-optimized).
        curriculum_episodes: Episodes to reach full difficulty. Default 2000.
        use_subproc: Use SubprocVecEnv (True) or DummyVecEnv (False).
                    Default True for CPU with 50GB RAM.

    Returns:
        Vectorized environment for training.
    """

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

    # CPU-optimized: Use SubprocVecEnv with 64 parallel processes
    # Each process gets its own environment, 50GB RAM supports this easily
    # Data replication is worth the parallelization benefit
    if use_subproc and n_envs > 1:
        return SubprocVecEnv([make_env(i) for i in range(n_envs)])
    else:
        print(f"[WARN] DummyVecEnv with {n_envs} envs limits CPU parallelization. Consider use_subproc=True.")
        return DummyVecEnv([make_env(i) for i in range(n_envs)])


def train_rl_agent(
    data_dir: str,
    output_dir: str,
    total_timesteps: int = 10_000_000,  # CPU-optimized: 10M steps (can train much longer)
    learning_rate: float = 5e-4,  # CPU-optimized: higher LR for CPU batch normalization
    n_envs: int = 64,  # CPU-optimized: 64 parallel environments with SubprocVecEnv
    eval_freq: int = 10000,
    save_freq: int = 300000,  # Reduced checkpoint frequency to save disk space
    device: str = 'cpu',  # CPU-optimized: much faster for MLP policies than GPU
    verbose: int = 1,
    curriculum_episodes: int = 2000,  # More phases with more environments
    use_recurrent: bool = True,  # LSTM for temporal modeling (good on CPU)
    use_subproc: bool = True,  # CPU-optimized: SubprocVecEnv for 64 envs (safe with 50GB RAM)
    use_hybrid: bool = False,  # CPU doesn't need massive models
) -> Dict[str, Any]:
    """
    Train RL agent optimized for CPU with 50GB RAM.

    CPU Optimization Strategy (50GB RAM):
    - SubprocVecEnv (64 envs): 64x parallel throughput, each process independent
    - Large batch sizes (4096): Leverages CPU cache efficiency
    - Large n_steps (2048): 131K samples per rollout (64 * 2048)
    - Result: High CPU utilization, excellent sample efficiency

    Why CPU is Better than GPU for MLP:
    1. PPO with MLP policy is CPU-bound (not compute-bound)
    2. SubprocVecEnv: 64 environments running in parallel
    3. Each CPU core runs an environment independently
    4. GPU would bottleneck waiting for data (GPU not saturated with MLP)
    5. Torch CPU has excellent native vectorization support

    Key features:
    1. RecurrentPPO with LSTM for temporal pattern recognition
    2. CPU-optimized parallel data pipeline (SubprocVecEnv 64 envs)
    3. Higher learning rate for CPU batch processing
    4. Win-rate focused reward shaping
    5. Curriculum learning (fees increase gradually)

    Args:
        data_dir: Directory containing data.
        output_dir: Directory for saving models.
        total_timesteps: Total training steps. Default 10M.
        learning_rate: Learning rate. Default 5e-4 (higher for CPU).
        n_envs: Parallel environments. Default 64 (CPU-optimized with SubprocVecEnv).
        curriculum_episodes: Episodes to full difficulty. Default 2000.
        use_recurrent: Use RecurrentPPO with LSTM. Default True.
        use_subproc: Use SubprocVecEnv. Default True (CPU-safe with 50GB).
        use_hybrid: Use Hybrid LSTM + Attention. Default False (CPU doesn't need).

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
    all_candles = load_token_candles(data_dir)
    np.random.seed(42)
    indices = np.random.permutation(len(all_candles))
    n_train = int(len(all_candles) * 0.9)
    train_candles = [all_candles[i] for i in indices[:n_train]]
    eval_candles = [all_candles[i] for i in indices[n_train:]]

    print(f"\n{'='*60}")
    print(f"DATA: {len(train_candles)} train tokens, {len(eval_candles)} eval tokens")

    # Create environments with multiprocessing
    train_env = create_curriculum_envs(
        train_candles,
        n_envs=n_envs,
        curriculum_episodes=curriculum_episodes,
        use_subproc=use_subproc,
    )
    # Use MultiTokenEvalEnvironment to evaluate across ALL eval tokens (not just one)
    # This ensures proper generalization testing and meaningful variance in metrics
    eval_env = MultiTokenEvalEnvironment(eval_candles)
    eval_env = Monitor(eval_env)

    # Determine algorithm: Hybrid LSTM+Attention, RecurrentPPO (LSTM), or standard PPO
    use_lstm = use_recurrent and RECURRENT_AVAILABLE

    if use_hybrid:
        # MASSIVE MODEL: Fully utilize 14GB T4 GPU for maximum win rate
        policy_kwargs = {
            "features_extractor_class": HybridLSTMAttentionExtractor,
            "features_extractor_kwargs": {
                "features_dim": 2048,      # MASSIVE: 512 -> 2048 (4x larger)
                "lstm_hidden": 1024,       # MASSIVE: 256 -> 1024 (4x larger)
                "lstm_layers": 3,          # Deeper: 2 -> 3 layers
                "n_heads": 16,             # More attention: 8 -> 16 heads
                "dropout": 0.2,            # Higher dropout for regularization
            },
            "net_arch": dict(
                pi=[2048, 1024, 512, 256],  # MASSIVE policy: 4 layers
                vf=[2048, 1024, 512, 256]   # MASSIVE value: 4 layers
            ),
            "activation_fn": torch.nn.GELU,
        }

        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=learning_rate * 0.5,  # Lower LR for stability with large model
            n_steps=4096,              # DOUBLED: 4096 * 8 = 32768 samples/rollout
            batch_size=8192,           # DOUBLED: 8192 for maximum GPU saturation
            n_epochs=10,               # Reduced epochs (large batches need fewer passes)
            gamma=0.995,               # Higher gamma for long-term thinking
            gae_lambda=0.98,           # Higher GAE for better advantage estimation
            clip_range=0.15,           # Slightly larger clip range
            clip_range_vf=0.15,
            ent_coef=0.01,             # Lower entropy (more exploitation)
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            verbose=0,
            tensorboard_log=str(log_dir),
            device=device,
        )

        print(f"MODEL: MASSIVE Hybrid (14GB GPU) | Features:2048 LSTM:1024x3 Heads:16 | Batch:8192 | Target WR:85-95%")

    elif use_lstm:
        # CPU-optimized RecurrentPPO with LSTM
        policy_kwargs = {
            "lstm_hidden_size": 512,   # CPU-friendly: reduced from 1024
            "n_lstm_layers": 2,        # CPU-friendly: reduced from 3
            "shared_lstm": False,
            "enable_critic_lstm": True,
            "net_arch": dict(pi=[512, 256], vf=[512, 256]),  # CPU-friendly sizing
            "activation_fn": torch.nn.ReLU,  # ReLU faster on CPU than GELU
        }

        model = RecurrentPPO(
            "MlpLstmPolicy",
            train_env,
            learning_rate=learning_rate,  # 5e-4 for CPU
            n_steps=2048,              # CPU-optimized: 2048 * 64 = 131,072 samples/rollout
            batch_size=4096,           # Large batches for CPU cache efficiency
            n_epochs=10,               # Fewer epochs (more data per step)
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            clip_range_vf=0.1,
            ent_coef=0.05,             # Maintain exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            verbose=0,  # Suppress SB3 verbose output
            tensorboard_log=str(log_dir),
            device=device,  # 'cpu' from function parameter
        )

        print(f"MODEL: RecurrentPPO (LSTM) | CPU-Optimized | Batch:{4096} | Steps:{2048} | Envs:{n_envs} | LR:{learning_rate}")

    else:
        # CPU-optimized Standard PPO
        policy_kwargs = {
            "features_extractor_class": AdvancedTradingFeaturesExtractor,
            "features_extractor_kwargs": {"features_dim": 512},
            "net_arch": dict(pi=[512, 256], vf=[512, 256]),
            "activation_fn": torch.nn.ReLU,  # ReLU faster on CPU
        }

        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=learning_rate,  # 5e-4 for CPU
            n_steps=2048,              # CPU-optimized: 2048 * 64 = 131,072 samples/rollout
            batch_size=4096,           # Large batches for CPU cache efficiency
            n_epochs=10,               # Fewer epochs with more data per step
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            clip_range_vf=0.1,
            ent_coef=0.05,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            verbose=0,  # Suppress SB3 verbose output
            tensorboard_log=str(log_dir),
            device=device,  # 'cpu' from function parameter
        )

        print(f"MODEL: Standard PPO | CPU-Optimized | Batch:{4096} | Steps:{2048} | Envs:{n_envs} | LR:{learning_rate}")

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

    trading_callback = ImprovedTradingCallback(check_freq=5000, verbose=1)
    callbacks = CallbackList([eval_callback, checkpoint_callback, trading_callback])

    # Train
    print(f"{'='*60}")
    print(f"TRAINING: {total_timesteps:,} steps | Curriculum: {curriculum_episodes} episodes")
    print(f"{'='*60}")

    start_time = datetime.now()
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    training_time = datetime.now() - start_time

    print(f"{'='*60}")
    print(f"COMPLETE: {training_time}")

    # Final evaluation
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

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS (20 eval episodes)")
    print(f"{'='*60}")
    print(f"Win Rate: {final_metrics['win_rate']:5.1%} | PnL: {final_metrics['mean_pnl']:+.4f}±{final_metrics['std_pnl']:.4f} | Trades: {final_metrics['mean_trades']:.1f}")
    print(f"{'='*60}")

    # Save
    model.save(str(output_path / "final_model_v2"))

    if use_hybrid:
        algorithm_name = "hybrid_lstm_attention"
    elif use_lstm:
        algorithm_name = "recurrent_ppo"
    else:
        algorithm_name = "ppo_v2"

    results = {
        "algorithm": algorithm_name,
        "use_hybrid": use_hybrid,
        "use_lstm": use_lstm,
        "total_timesteps": total_timesteps,
        "learning_rate": learning_rate,
        "n_envs": n_envs,
        "training_time": str(training_time),
        "curriculum_episodes": curriculum_episodes,
        "n_training_tokens": len(train_candles),
        "n_eval_tokens": len(eval_candles),
        "final_metrics": final_metrics,
    }

    with open(output_path / "training_results_v2.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"SAVED: {output_path / 'final_model_v2'}\n")

    train_env.close()
    eval_env.close()

    return results
