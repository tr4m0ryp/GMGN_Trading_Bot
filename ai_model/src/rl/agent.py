"""
RL Trading Agent with custom policy network.

This module implements the RL trading agent using PPO with a custom
neural network policy. The policy can be an MLP (for single-step obs)
or integrated with the Transformer-LSTM (for sequence obs).

Dependencies:
    torch: Deep learning framework
    stable_baselines3: RL algorithms
    gymnasium: Environment interface

Author: Trading Team
Date: 2025-12-23
"""

from typing import Dict, List, Tuple, Type, Union, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym


class TradingFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for trading observations.

    Processes the raw observation (features + position info) through
    a neural network to produce a rich feature representation.

    Architecture:
        Input -> Linear -> LayerNorm -> GELU -> Dropout
              -> Linear -> LayerNorm -> GELU -> Dropout
              -> Output (features_dim)

    Args:
        observation_space: Gym observation space.
        features_dim: Output feature dimension. Default is 128.
        hidden_dim: Hidden layer dimension. Default is 256.
        dropout: Dropout probability. Default is 0.2.

    Example:
        >>> extractor = TradingFeaturesExtractor(env.observation_space, features_dim=128)
        >>> features = extractor(observations)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__(observation_space, features_dim)

        input_dim = int(np.prod(observation_space.shape))

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.GELU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Extract features from observations."""
        return self.network(observations)


class AdvancedTradingFeaturesExtractor(BaseFeaturesExtractor):
    """
    Advanced feature extractor with multi-scale pattern detection.

    Uses parallel processing paths at different scales to capture
    both short-term signals and longer-term patterns.

    Args:
        observation_space: Gym observation space.
        features_dim: Output feature dimension. Default is 256.

    Example:
        >>> extractor = AdvancedTradingFeaturesExtractor(env.observation_space)
        >>> features = extractor(observations)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 256,
    ):
        super().__init__(observation_space, features_dim)

        input_dim = int(np.prod(observation_space.shape))
        hidden = features_dim // 2

        # Multi-scale feature extraction
        self.scale1 = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )

        self.scale2 = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
        )

        # Combine scales
        self.combine = nn.Sequential(
            nn.Linear(hidden * 2, features_dim),
            nn.LayerNorm(features_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Position-aware processing
        # Last 4 dims are position info: in_position, entry_price, unrealized_pnl, time_in_position
        self.position_net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )

        # Final projection
        self.output = nn.Sequential(
            nn.Linear(features_dim + 32, features_dim),
            nn.LayerNorm(features_dim),
            nn.GELU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Extract features from observations."""
        # Split features and position info
        features = observations[..., :-4]
        position_info = observations[..., -4:]

        # Multi-scale processing of features
        s1 = self.scale1(observations)
        s2 = self.scale2(observations)
        combined = self.combine(torch.cat([s1, s2], dim=-1))

        # Position-aware features
        pos_features = self.position_net(position_info)

        # Combine all
        out = self.output(torch.cat([combined, pos_features], dim=-1))

        return out


class TradingTrainCallback(BaseCallback):
    """
    Custom callback for logging trading-specific metrics during training.

    Logs aggregate PnL, win rate, and trade statistics to TensorBoard
    and prints progress to console.

    Args:
        check_freq: Logging frequency in steps. Default is 1000.
        verbose: Verbosity level. Default is 1.

    Example:
        >>> callback = TradingTrainCallback(check_freq=1000)
        >>> model.learn(total_timesteps=100000, callback=callback)
    """

    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_pnls = []
        self.episode_trades = []
        self.episode_win_rates = []

    def _on_step(self) -> bool:
        """Called at each training step."""
        # Check for episode end info
        for info in self.locals.get("infos", []):
            if "total_pnl" in info:
                self.episode_pnls.append(info["total_pnl"])
                self.episode_trades.append(info.get("n_trades", 0))
                self.episode_win_rates.append(info.get("win_rate", 0.0))

        # Log periodically
        if self.n_calls % self.check_freq == 0 and len(self.episode_pnls) > 0:
            mean_pnl = np.mean(self.episode_pnls[-100:])
            mean_trades = np.mean(self.episode_trades[-100:])
            mean_win_rate = np.mean(self.episode_win_rates[-100:])

            if self.verbose >= 1:
                print(f"\n[Step {self.n_calls}] "
                      f"Mean PnL: {mean_pnl:.4f} | "
                      f"Mean Trades: {mean_trades:.1f} | "
                      f"Win Rate: {mean_win_rate:.2%}")

            # Log to TensorBoard
            if self.logger is not None:
                self.logger.record("trading/mean_pnl", mean_pnl)
                self.logger.record("trading/mean_trades", mean_trades)
                self.logger.record("trading/mean_win_rate", mean_win_rate)

        return True


class RLTradingAgent:
    """
    High-level interface for RL trading agent.

    Wraps Stable Baselines3 algorithms (PPO, A2C, DQN) with custom
    policy networks optimized for trading.

    Args:
        env: Trading environment.
        algorithm: RL algorithm to use ('ppo', 'a2c', 'dqn'). Default is 'ppo'.
        learning_rate: Learning rate. Default is 3e-4.
        features_extractor: Feature extractor class. Default is TradingFeaturesExtractor.
        features_dim: Feature extractor output dimension. Default is 256.
        net_arch: Policy/value network architecture. Default is [256, 128].
        device: Device to use ('cuda' or 'cpu'). Default is 'auto'.
        verbose: Verbosity level. Default is 1.

    Example:
        >>> env = TradingEnvironment(candles)
        >>> agent = RLTradingAgent(env, algorithm='ppo')
        >>> agent.train(total_timesteps=100000)
        >>> action = agent.predict(obs)
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

        # Policy kwargs
        policy_kwargs = {
            "features_extractor_class": features_extractor,
            "features_extractor_kwargs": {"features_dim": features_dim},
            "net_arch": net_arch,
        }

        # Create algorithm
        algorithm_class = self.ALGORITHMS.get(self.algorithm_name)
        if algorithm_class is None:
            raise ValueError(f"Unknown algorithm: {algorithm}. "
                           f"Choose from {list(self.ALGORITHMS.keys())}")

        # DQN doesn't use net_arch the same way
        if self.algorithm_name == 'dqn':
            policy_kwargs.pop("net_arch")
            self.model = algorithm_class(
                "MlpPolicy",
                env,
                learning_rate=learning_rate,
                policy_kwargs=policy_kwargs,
                device=device,
                verbose=verbose,
                buffer_size=100000,
                learning_starts=1000,
                batch_size=256,
                **kwargs
            )
        else:
            self.model = algorithm_class(
                "MlpPolicy",
                env,
                learning_rate=learning_rate,
                policy_kwargs=policy_kwargs,
                device=device,
                verbose=verbose,
                n_steps=2048,
                batch_size=256,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                **kwargs
            )

    def train(
        self,
        total_timesteps: int,
        callback: Optional[BaseCallback] = None,
        log_interval: int = 100,
    ) -> None:
        """
        Train the agent.

        Args:
            total_timesteps: Total number of training steps.
            callback: Optional callback for logging. If None, uses default.
            log_interval: Logging interval in episodes.
        """
        if callback is None:
            callback = TradingTrainCallback(check_freq=1000, verbose=self.verbose)

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
        )

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[int, Optional[np.ndarray]]:
        """
        Predict action for given observation.

        Args:
            observation: Current observation.
            deterministic: Whether to use deterministic policy.

        Returns:
            Tuple of (action, states).
        """
        action, states = self.model.predict(observation, deterministic=deterministic)
        return action, states

    def save(self, path: str) -> None:
        """Save model to disk."""
        self.model.save(path)

    def load(self, path: str) -> None:
        """Load model from disk."""
        algorithm_class = self.ALGORITHMS[self.algorithm_name]
        self.model = algorithm_class.load(path, env=self.env)

    def evaluate(
        self,
        n_episodes: int = 10,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate agent performance.

        Args:
            n_episodes: Number of evaluation episodes.
            deterministic: Whether to use deterministic policy.

        Returns:
            Dictionary with evaluation metrics.
        """
        total_pnls = []
        total_trades = []
        total_wins = []
        episode_lengths = []

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
