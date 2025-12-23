"""
Reinforcement Learning module for trading model.

This module provides RL-based trading agents that learn optimal buy/sell
decisions through interaction with a simulated trading environment,
maximizing profit rather than fitting predetermined labels.

Modules:
    environment: Gym-compatible trading environment (v1)
    environment_v2: Improved environment with curriculum learning
    agent: RL agent with custom policy networks
    trainer: PPO training loop and utilities
    trainer_v2: Improved training with curriculum and better hyperparams

Author: Trading Team
Date: 2025-12-23
"""

from .environment import TradingEnvironment, MultiTokenTradingEnvironment
from .environment_v2 import TradingEnvironmentV2, CurriculumTradingEnvironment
from .agent import RLTradingAgent, TradingFeaturesExtractor, AdvancedTradingFeaturesExtractor
from .trainer import train_rl_agent
from .trainer_v2 import train_rl_agent_v2

__all__ = [
    # V1 (original)
    'TradingEnvironment',
    'MultiTokenTradingEnvironment',
    'RLTradingAgent',
    'train_rl_agent',
    # V2 (improved)
    'TradingEnvironmentV2',
    'CurriculumTradingEnvironment',
    'TradingFeaturesExtractor',
    'AdvancedTradingFeaturesExtractor',
    'train_rl_agent_v2',
]
