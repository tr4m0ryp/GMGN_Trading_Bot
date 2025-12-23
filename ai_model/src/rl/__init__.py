"""
Reinforcement Learning module for trading model.

This module provides RL-based trading agents that learn optimal buy/sell
decisions through interaction with a simulated trading environment,
maximizing profit rather than fitting predetermined labels.

Key Features:
- Curriculum learning: Gradually increases trading fees during training
- Hindsight rewards: Penalizes missing profitable opportunities
- Advanced reward shaping: Prevents policy collapse
- PPO with optimized hyperparameters for exploration

Modules:
    environment_v2: Trading environments with curriculum learning
    agent: RL agent with custom policy networks
    trainer_v2: PPO training pipeline with curriculum

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
