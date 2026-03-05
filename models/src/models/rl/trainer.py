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

import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

import numpy as np
import torch

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor

# Try to import RecurrentPPO from sb3-contrib for LSTM support
try:
    from sb3_contrib import RecurrentPPO
    RECURRENT_AVAILABLE = True
except ImportError:
    RECURRENT_AVAILABLE = False
    print("Warning: sb3-contrib not installed. RecurrentPPO unavailable.")
    print("Install with: pip install sb3-contrib")

from .environment import MultiTokenEvalEnvironment
from .agent import (
    TradingFeaturesExtractor,
    AdvancedTradingFeaturesExtractor,
    HybridLSTMAttentionExtractor,
)
from .trainer_utils import (
    load_token_candles,
    ImprovedTradingCallback,
    create_curriculum_envs,
)


def train_rl_agent(
    data_dir: str,
    output_dir: str,
    total_timesteps: int = 2_000_000,
    learning_rate: float = 3e-4,
    n_envs: int = 8,
    eval_freq: int = 10000,
    save_freq: int = 300000,
    device: str = 'cuda',
    verbose: int = 1,
    curriculum_episodes: int = 1000,
    use_recurrent: bool = True,
    use_subproc: bool = False,
    use_hybrid: bool = False,
    use_simplified_reward: bool = False,
) -> Dict[str, Any]:
    """
    Train RL agent optimized for GPU utilization and maximum win rate.

    Args:
        data_dir: Directory containing data.
        output_dir: Directory for saving models.
        total_timesteps: Total training steps. Default 2M.
        learning_rate: Learning rate. Default 3e-4.
        n_envs: Parallel environments. Default 8.
        eval_freq: Evaluation frequency.
        save_freq: Checkpoint frequency.
        device: Training device.
        verbose: Verbosity level.
        curriculum_episodes: Episodes to full difficulty. Default 1000.
        use_recurrent: Use RecurrentPPO with LSTM. Default True.
        use_subproc: Use SubprocVecEnv. Default False.
        use_hybrid: Use Hybrid LSTM + Attention. Default False.
        use_simplified_reward: Use simplified reward system.

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

    # Create environments
    train_env = create_curriculum_envs(
        train_candles, n_envs=n_envs, curriculum_episodes=curriculum_episodes,
        use_subproc=use_subproc, use_simplified_reward=use_simplified_reward,
    )
    eval_env = MultiTokenEvalEnvironment(eval_candles)
    eval_env = Monitor(eval_env)

    # Build model
    use_lstm = use_recurrent and RECURRENT_AVAILABLE
    model = _build_model(
        train_env, log_dir, learning_rate, device,
        use_hybrid, use_lstm, n_envs,
    )

    # Callbacks
    eval_callback = EvalCallback(
        eval_env, best_model_save_path=str(output_path), log_path=str(log_dir),
        eval_freq=eval_freq // n_envs, n_eval_episodes=10, deterministic=True,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // n_envs, save_path=str(checkpoint_dir),
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
    final_metrics = _evaluate_final(model, eval_env)

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS (20 eval episodes)")
    print(f"{'='*60}")
    print(f"Win Rate: {final_metrics['win_rate']:5.1%} | PnL: {final_metrics['mean_pnl']:+.4f}+/-{final_metrics['std_pnl']:.4f} | Trades: {final_metrics['mean_trades']:.1f}")
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
        "algorithm": algorithm_name, "use_hybrid": use_hybrid, "use_lstm": use_lstm,
        "total_timesteps": total_timesteps, "learning_rate": learning_rate,
        "n_envs": n_envs, "training_time": str(training_time),
        "curriculum_episodes": curriculum_episodes,
        "n_training_tokens": len(train_candles), "n_eval_tokens": len(eval_candles),
        "final_metrics": final_metrics,
    }

    with open(output_path / "training_results_v2.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"SAVED: {output_path / 'final_model_v2'}\n")
    train_env.close()
    eval_env.close()
    return results


def _build_model(train_env, log_dir, learning_rate, device, use_hybrid, use_lstm, n_envs):
    """Build the PPO or RecurrentPPO model with appropriate configuration."""
    if use_hybrid:
        policy_kwargs = {
            "features_extractor_class": HybridLSTMAttentionExtractor,
            "features_extractor_kwargs": {
                "features_dim": 2048, "lstm_hidden": 1024,
                "lstm_layers": 3, "n_heads": 16, "dropout": 0.2,
            },
            "net_arch": dict(pi=[2048, 1024, 512, 256], vf=[2048, 1024, 512, 256]),
            "activation_fn": torch.nn.GELU,
        }
        model = PPO(
            "MlpPolicy", train_env, learning_rate=learning_rate * 0.5,
            n_steps=4096, batch_size=8192, n_epochs=10, gamma=0.995,
            gae_lambda=0.98, clip_range=0.15, clip_range_vf=0.15,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
            policy_kwargs=policy_kwargs, verbose=0,
            tensorboard_log=str(log_dir), device=device,
        )
        print(f"MODEL: MASSIVE Hybrid (14GB GPU) | Features:2048 LSTM:1024x3 Heads:16 | Batch:8192 | Target WR:85-95%")

    elif use_lstm:
        policy_kwargs = {
            "lstm_hidden_size": 1024, "n_lstm_layers": 3,
            "shared_lstm": False, "enable_critic_lstm": True,
            "net_arch": dict(pi=[1024, 512, 256], vf=[1024, 512, 256]),
            "activation_fn": torch.nn.GELU,
        }
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO(
            "MlpLstmPolicy", train_env, learning_rate=learning_rate,
            n_steps=2048, batch_size=4096, n_epochs=15, gamma=0.99,
            gae_lambda=0.95, clip_range=0.1, clip_range_vf=0.1,
            ent_coef=0.05, vf_coef=0.5, max_grad_norm=0.5,
            policy_kwargs=policy_kwargs, verbose=0,
            tensorboard_log=str(log_dir), device=device,
        )
        print(f"MODEL: RecurrentPPO (LSTM) | Batch:4096 | Steps:2048 | Envs:{n_envs} | Target WR:75-85%")

    else:
        policy_kwargs = {
            "features_extractor_class": AdvancedTradingFeaturesExtractor,
            "features_extractor_kwargs": {"features_dim": 512},
            "net_arch": dict(pi=[512, 256], vf=[512, 256]),
            "activation_fn": torch.nn.GELU,
        }
        model = PPO(
            "MlpPolicy", train_env, learning_rate=learning_rate,
            n_steps=2048, batch_size=4096, n_epochs=15, gamma=0.99,
            gae_lambda=0.95, clip_range=0.1, clip_range_vf=0.1,
            ent_coef=0.05, vf_coef=0.5, max_grad_norm=0.5,
            policy_kwargs=policy_kwargs, verbose=0,
            tensorboard_log=str(log_dir), device=device,
        )
        print(f"MODEL: Standard PPO | Batch:4096 | Steps:2048 | Envs:{n_envs} | Target WR:65-80%")

    return model


def _evaluate_final(model, eval_env, n_episodes: int = 20) -> Dict[str, float]:
    """Run final evaluation episodes and return metrics."""
    episode_pnls = []
    episode_trades = []
    episode_win_rates = []

    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
        episode_pnls.append(info["total_pnl"])
        episode_trades.append(info["n_trades"])
        episode_win_rates.append(info["win_rate"])

    return {
        "mean_pnl": float(np.mean(episode_pnls)),
        "std_pnl": float(np.std(episode_pnls)),
        "mean_trades": float(np.mean(episode_trades)),
        "win_rate": float(np.mean(episode_win_rates)),
    }
