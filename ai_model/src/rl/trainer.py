"""
RL Training utilities for the trading agent.

This module provides high-level training functions that handle
data loading, environment creation, training, and evaluation.

Dependencies:
    torch: Deep learning framework
    stable_baselines3: RL algorithms
    tqdm: Progress bars

Author: Trading Team
Date: 2025-12-23
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor

from .environment import TradingEnvironment, MultiTokenTradingEnvironment
from .agent import RLTradingAgent, TradingTrainCallback


def load_token_candles(data_dir: str) -> List[List[Dict[str, float]]]:
    """
    Load candle data for all tokens from raw CSV.

    Args:
        data_dir: Directory containing rawdata.csv.

    Returns:
        List of candle lists, one per token.
    """
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
            if len(candles) >= 50:  # Minimum length for meaningful episode
                all_candles.append(candles)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to parse token {idx}: {e}")
            continue

    print(f"Loaded {len(all_candles)} tokens with valid candle data")
    return all_candles


def create_training_envs(
    all_candles: List[List[Dict[str, float]]],
    n_envs: int = 4,
    use_multiprocessing: bool = False,
) -> DummyVecEnv:
    """
    Create vectorized training environments.

    Args:
        all_candles: List of candle lists for all tokens.
        n_envs: Number of parallel environments.
        use_multiprocessing: Whether to use SubprocVecEnv.

    Returns:
        Vectorized environment.
    """
    def make_env(seed: int):
        def _init():
            env = MultiTokenTradingEnvironment(all_candles)
            env = Monitor(env)
            env.reset(seed=seed)
            return env
        return _init

    if use_multiprocessing:
        envs = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    else:
        envs = DummyVecEnv([make_env(i) for i in range(n_envs)])

    return envs


def train_rl_agent(
    data_dir: str,
    output_dir: str,
    algorithm: str = 'ppo',
    total_timesteps: int = 500000,
    learning_rate: float = 3e-4,
    n_envs: int = 4,
    eval_freq: int = 10000,
    save_freq: int = 50000,
    device: str = 'auto',
    verbose: int = 1,
) -> Dict[str, Any]:
    """
    Train an RL trading agent.

    Complete training pipeline that:
    1. Loads token data
    2. Creates training and evaluation environments
    3. Trains agent with PPO/A2C/DQN
    4. Evaluates periodically
    5. Saves checkpoints and final model

    Args:
        data_dir: Directory containing ai_model/data/.
        output_dir: Directory for saving models and logs.
        algorithm: RL algorithm ('ppo', 'a2c', 'dqn'). Default is 'ppo'.
        total_timesteps: Total training steps. Default is 500000.
        learning_rate: Learning rate. Default is 3e-4.
        n_envs: Number of parallel environments. Default is 4.
        eval_freq: Evaluation frequency in steps. Default is 10000.
        save_freq: Checkpoint save frequency. Default is 50000.
        device: Device to use ('cuda', 'cpu', 'auto'). Default is 'auto'.
        verbose: Verbosity level. Default is 1.

    Returns:
        Dictionary with training results and metrics.

    Example:
        >>> results = train_rl_agent(
        ...     data_dir='../data',
        ...     output_dir='../models/rl',
        ...     total_timesteps=1000000,
        ... )
        >>> print(f"Final PnL: {results['final_metrics']['mean_pnl']:.4f}")
    """
    # Setup directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    log_dir = output_path / "logs"
    log_dir.mkdir(exist_ok=True)

    # Load data
    print("Loading token data...")
    all_candles = load_token_candles(data_dir)

    # Split into train/eval
    np.random.seed(42)
    indices = np.random.permutation(len(all_candles))
    n_train = int(len(all_candles) * 0.9)

    train_candles = [all_candles[i] for i in indices[:n_train]]
    eval_candles = [all_candles[i] for i in indices[n_train:]]

    print(f"Training tokens: {len(train_candles)}")
    print(f"Evaluation tokens: {len(eval_candles)}")

    # Create environments
    print("Creating training environments...")
    train_env = create_training_envs(train_candles, n_envs=n_envs)

    print("Creating evaluation environment...")
    eval_env = MultiTokenTradingEnvironment(eval_candles)
    eval_env = Monitor(eval_env)

    # Create agent
    print(f"Creating {algorithm.upper()} agent...")
    agent = RLTradingAgent(
        train_env,
        algorithm=algorithm,
        learning_rate=learning_rate,
        device=device,
        verbose=verbose,
        tensorboard_log=str(log_dir),
    )

    # Setup callbacks
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
        name_prefix="rl_trader",
    )

    trading_callback = TradingTrainCallback(check_freq=1000, verbose=verbose)

    callbacks = CallbackList([eval_callback, checkpoint_callback, trading_callback])

    # Train
    print(f"\nStarting training for {total_timesteps:,} timesteps...")
    print("=" * 60)

    start_time = datetime.now()
    agent.train(total_timesteps=total_timesteps, callback=callbacks)
    training_time = datetime.now() - start_time

    print("=" * 60)
    print(f"Training completed in {training_time}")

    # Final evaluation
    print("\nFinal evaluation...")
    final_metrics = agent.evaluate(n_episodes=20)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Mean PnL: {final_metrics['mean_pnl']:.4f} +/- {final_metrics['std_pnl']:.4f}")
    print(f"Mean Trades: {final_metrics['mean_trades']:.1f}")
    print(f"Win Rate: {final_metrics['win_rate']:.2%}")
    print("=" * 60)

    # Save final model
    final_model_path = output_path / "final_model"
    agent.save(str(final_model_path))
    print(f"\nFinal model saved to: {final_model_path}")

    # Save results
    results = {
        "algorithm": algorithm,
        "total_timesteps": total_timesteps,
        "learning_rate": learning_rate,
        "training_time": str(training_time),
        "n_training_tokens": len(train_candles),
        "n_eval_tokens": len(eval_candles),
        "final_metrics": final_metrics,
    }

    results_path = output_path / "training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Cleanup
    train_env.close()
    eval_env.close()

    return results


def backtest_rl_agent(
    model_path: str,
    candles: List[Dict[str, float]],
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Backtest a trained RL agent on historical data.

    Runs the agent through the price history and records all trades,
    providing detailed performance metrics.

    Args:
        model_path: Path to saved model.
        candles: List of candle dictionaries.
        verbose: Whether to print progress.

    Returns:
        Dictionary with backtest results including:
        - total_pnl: Total profit/loss
        - trades: List of trade details
        - win_rate: Winning trade percentage
        - sharpe_ratio: Risk-adjusted return
    """
    from stable_baselines3 import PPO

    # Create environment
    env = TradingEnvironment(candles)

    # Load model
    model = PPO.load(model_path, env=env)

    # Run backtest
    obs, _ = env.reset()
    trades = []
    actions_taken = []

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        actions_taken.append(int(action))

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if info.get("trade_executed", False):
            trades.append({
                "step": info["current_step"],
                "action": "BUY" if len(trades) % 2 == 0 else "SELL",
                "pnl": info.get("trade_pnl", 0.0),
            })

    # Calculate metrics
    pnls = [t["pnl"] for t in trades if t["action"] == "SELL"]
    total_pnl = sum(pnls)
    n_trades = len(pnls)
    n_wins = sum(1 for p in pnls if p > 0)
    win_rate = n_wins / max(1, n_trades)

    # Sharpe ratio (simplified)
    if len(pnls) > 1:
        sharpe = np.mean(pnls) / (np.std(pnls) + 1e-8) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Action distribution
    from collections import Counter
    action_counts = Counter(actions_taken)

    results = {
        "total_pnl": total_pnl,
        "n_trades": n_trades,
        "n_wins": n_wins,
        "win_rate": win_rate,
        "sharpe_ratio": sharpe,
        "trades": trades,
        "action_distribution": dict(action_counts),
        "mean_pnl_per_trade": np.mean(pnls) if pnls else 0.0,
    }

    if verbose:
        print("\nBacktest Results:")
        print(f"  Total PnL: {total_pnl:.4f}")
        print(f"  Number of Trades: {n_trades}")
        print(f"  Win Rate: {win_rate:.2%}")
        print(f"  Sharpe Ratio: {sharpe:.2f}")
        print(f"  Action Distribution: {dict(action_counts)}")

    return results
