"""
RL Trading Agent with custom policy network.

Wraps Stable Baselines3 algorithms (PPO, A2C, DQN) with custom policy
networks optimized for trading, plus a training callback for metrics.

Dependencies: torch, stable_baselines3, gymnasium
Author: Trading Team
Date: 2025-12-23
"""

from typing import Dict, List, Tuple, Type, Optional

import numpy as np
import torch

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym

from .feature_extractors import (
    TradingFeaturesExtractor,
    AdvancedTradingFeaturesExtractor,
    HybridLSTMAttentionExtractor,
)


class TradingTrainCallback(BaseCallback):
    """Callback for logging trading-specific metrics (PnL, win rate, trades)."""

    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_pnls = []
        self.episode_trades = []
        self.episode_win_rates = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "total_pnl" in info:
                self.episode_pnls.append(info["total_pnl"])
                self.episode_trades.append(info.get("n_trades", 0))
                self.episode_win_rates.append(info.get("win_rate", 0.0))

        if self.n_calls % self.check_freq == 0 and len(self.episode_pnls) > 0:
            mean_pnl = np.mean(self.episode_pnls[-100:])
            mean_trades = np.mean(self.episode_trades[-100:])
            mean_win_rate = np.mean(self.episode_win_rates[-100:])

            if self.verbose >= 1:
                print(f"\n[Step {self.n_calls}] "
                      f"Mean PnL: {mean_pnl:.4f} | "
                      f"Mean Trades: {mean_trades:.1f} | "
                      f"Win Rate: {mean_win_rate:.2%}")

            if self.logger is not None:
                self.logger.record("trading/mean_pnl", mean_pnl)
                self.logger.record("trading/mean_trades", mean_trades)
                self.logger.record("trading/mean_win_rate", mean_win_rate)

        return True


class RLTradingAgent:
    """High-level interface for RL trading agent.

    Supports PPO, A2C, and DQN algorithms with custom feature extractors,
    confidence-based selective trading, and evaluation utilities.
    """

    ALGORITHMS = {
        'ppo': PPO,
        'a2c': A2C,
        'dqn': DQN,
    }

    def __init__(
        self,
        env: gym.Env,
        algorithm: str = 'ppo',
        learning_rate: float = 3e-4,
        features_extractor: Type[BaseFeaturesExtractor] = TradingFeaturesExtractor,
        features_dim: int = 256,
        net_arch: Optional[List[int]] = None,
        device: str = 'auto',
        verbose: int = 1,
        **kwargs
    ):
        self.env = env
        self.algorithm_name = algorithm.lower()
        self.verbose = verbose

        if net_arch is None:
            net_arch = [256, 128]

        policy_kwargs = {
            "features_extractor_class": features_extractor,
            "features_extractor_kwargs": {"features_dim": features_dim},
            "net_arch": net_arch,
        }

        algorithm_class = self.ALGORITHMS.get(self.algorithm_name)
        if algorithm_class is None:
            raise ValueError(f"Unknown algorithm: {algorithm}. "
                           f"Choose from {list(self.ALGORITHMS.keys())}")

        if self.algorithm_name == 'dqn':
            policy_kwargs.pop("net_arch")
            self.model = algorithm_class(
                "MlpPolicy", env, learning_rate=learning_rate,
                policy_kwargs=policy_kwargs, device=device, verbose=verbose,
                buffer_size=100000, learning_starts=1000, batch_size=256,
                **kwargs)
        else:
            self.model = algorithm_class(
                "MlpPolicy", env, learning_rate=learning_rate,
                policy_kwargs=policy_kwargs, device=device, verbose=verbose,
                n_steps=2048, batch_size=256, n_epochs=10,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                **kwargs)

    def train(self, total_timesteps: int,
              callback: Optional[BaseCallback] = None,
              log_interval: int = 100) -> None:
        """Train the agent for total_timesteps steps."""
        if callback is None:
            callback = TradingTrainCallback(check_freq=1000, verbose=self.verbose)
        self.model.learn(total_timesteps=total_timesteps,
                        callback=callback, log_interval=log_interval)

    def predict(self, observation: np.ndarray,
                deterministic: bool = True) -> Tuple[int, Optional[np.ndarray]]:
        """Predict action for given observation. Returns (action, states)."""
        return self.model.predict(observation, deterministic=deterministic)

    def save(self, path: str) -> None:
        """Save model to disk."""
        self.model.save(path)

    def load(self, path: str) -> None:
        """Load model from disk."""
        algorithm_class = self.ALGORITHMS[self.algorithm_name]
        self.model = algorithm_class.load(path, env=self.env)

    def predict_with_confidence(self, observation: np.ndarray,
                                confidence_threshold: float = 0.6) -> Tuple[int, float]:
        """Predict action only if confidence exceeds threshold.

        Returns (0, confidence) for HOLD if confidence is below threshold.
        """
        obs_tensor = torch.tensor(observation).float().unsqueeze(0)
        if hasattr(self.model, 'device'):
            obs_tensor = obs_tensor.to(self.model.device)

        with torch.no_grad():
            features = self.model.policy.extract_features(obs_tensor)
            if hasattr(self.model.policy, 'mlp_extractor'):
                latent_pi, _ = self.model.policy.mlp_extractor(features)
            else:
                latent_pi = features

            action_logits = self.model.policy.action_net(latent_pi)
            action_probs = torch.softmax(action_logits, dim=-1).squeeze()

            best_action = action_probs.argmax().item()
            confidence = action_probs[best_action].item()

            if best_action != 0 and confidence < confidence_threshold:
                return 0, confidence
            return best_action, confidence

    def evaluate(self, n_episodes: int = 10,
                 deterministic: bool = True) -> Dict[str, float]:
        """Evaluate agent over n_episodes. Returns metrics dict."""
        total_pnls, total_trades, total_wins, episode_lengths = [], [], [], []

        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            ep_length = 0
            while not done:
                action, _ = self.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                ep_length += 1

            total_pnls.append(info.get("total_pnl", 0.0))
            total_trades.append(info.get("n_trades", 0))
            total_wins.append(info.get("n_trades", 0) * info.get("win_rate", 0.0))
            episode_lengths.append(ep_length)

        return {
            "mean_pnl": np.mean(total_pnls),
            "std_pnl": np.std(total_pnls),
            "mean_trades": np.mean(total_trades),
            "win_rate": np.sum(total_wins) / max(1, np.sum(total_trades)),
            "mean_episode_length": np.mean(episode_lengths),
        }

    def evaluate_with_confidence(self, n_episodes: int = 10,
                                 confidence_threshold: float = 0.7) -> Dict[str, float]:
        """Evaluate with confidence-based selective trading."""
        total_pnls, total_trades, total_wins = [], [], []
        total_confidence_filtered = []

        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            filtered_actions = 0
            while not done:
                action, confidence = self.predict_with_confidence(
                    obs, confidence_threshold)
                if confidence < confidence_threshold:
                    filtered_actions += 1
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

            total_pnls.append(info.get("total_pnl", 0.0))
            total_trades.append(info.get("n_trades", 0))
            total_wins.append(info.get("n_trades", 0) * info.get("win_rate", 0.0))
            total_confidence_filtered.append(filtered_actions)

        return {
            "mean_pnl": np.mean(total_pnls),
            "std_pnl": np.std(total_pnls),
            "mean_trades": np.mean(total_trades),
            "win_rate": np.sum(total_wins) / max(1, np.sum(total_trades)),
            "mean_filtered_actions": np.mean(total_confidence_filtered),
            "confidence_threshold": confidence_threshold,
        }
