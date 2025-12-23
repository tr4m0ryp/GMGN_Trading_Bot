"""
Reinforcement Learning module for trading model.

This module provides RL-based trading agents that learn optimal buy/sell
decisions through interaction with a simulated trading environment,
maximizing profit rather than fitting predetermined labels.

Modules:
    environment: Gym-compatible trading environment
    agent: RL agent with Transformer-LSTM policy network
    trainer: PPO training loop and utilities

Author: Trading Team
Date: 2025-12-23
"""

from .environment import TradingEnvironment
from .agent import RLTradingAgent
from .trainer import train_rl_agent

__all__ = [
    'TradingEnvironment',
    'RLTradingAgent',
    'train_rl_agent',
]
