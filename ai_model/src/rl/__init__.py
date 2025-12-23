"""
Reinforcement Learning module for trading model - High Win Rate Edition.

This module provides RL-based trading agents optimized for 90%+ win rate
through binary rewards, LSTM temporal memory, and confidence-based trading.

Key Features (v2):
- RecurrentPPO (LSTM): Temporal pattern recognition via sb3-contrib
- Binary win-rate rewards: +1.0 for wins, -0.3 for losses
- Win-rate episode bonus: +0.5 for achieving 90%+ win rate
- Confidence-based trading: Only trades when model is confident
- SubprocVecEnv: 16 parallel environments for 3-4x speedup
- Curriculum learning: Gradually increases trading fees during training

Modules:
    environment: Trading environments with win-rate focused rewards
    agent: RL agent with custom policy networks and confidence prediction
    trainer: Training pipeline with RecurrentPPO/PPO support

Usage:
    from rl import train_rl_agent

    results = train_rl_agent(
        data_dir="./data",
        output_dir="./models",
        use_recurrent=True,  # Use LSTM
        use_subproc=True,    # Use parallel environments
        n_envs=16,           # 16 parallel environments
    )

Author: Trading Team
Date: 2025-12-23
"""

from .environment import TradingEnvironmentV2, CurriculumTradingEnvironment
from .agent import (
    RLTradingAgent,
    TradingFeaturesExtractor,
    AdvancedTradingFeaturesExtractor,
    TradingTrainCallback,
)
from .trainer import train_rl_agent

__all__ = [
    'TradingEnvironmentV2',
    'CurriculumTradingEnvironment',
    'RLTradingAgent',
    'TradingFeaturesExtractor',
    'AdvancedTradingFeaturesExtractor',
    'TradingTrainCallback',
    'train_rl_agent',
]
