# AI Trading Model - Reinforcement Learning

This directory contains the reinforcement learning trading agent that learns optimal buy/sell strategies for cryptocurrency trading on GMGN.ai.

## Overview

The model uses **Proximal Policy Optimization (PPO)** with **curriculum learning** to discover profitable trading patterns without relying on hand-crafted labels.

### Key Innovation: Curriculum Learning

Instead of training directly with full transaction fees, the agent gradually experiences increasing difficulty:

1. **Early Training** (0-300 episodes): No fees → Learn basic patterns
2. **Mid Training** (300-700 episodes): Partial fees → Refine strategy
3. **Late Training** (700-1000 episodes): Full fees → Real-world conditions

This prevents **policy collapse** where the agent learns to "do nothing" to avoid losing money on fees.

## Quick Start

### 1. Prepare Data

```bash
# From ai_model directory
python src/data/preprocess.py
```

This loads raw CSV data and creates train/val/test splits.

### 2. Train in Google Colab

Upload `notebooks/train_rl_agent.ipynb` to Google Colab and run all cells:

```python
# Key training parameters
TOTAL_TIMESTEPS = 1_000_000     # 1M steps (~2-3 hours on T4 GPU)
CURRICULUM_EPISODES = 1000       # Episodes to reach full fees
LEARNING_RATE = 1e-4            # Conservative LR for stability
```

### 3. Download Trained Model

The notebook automatically downloads:
- `final_model.zip` - Trained PPO agent
- `training_results.json` - Performance metrics

## Architecture

### Environment (`src/rl/environment.py`)

Gym-compatible trading simulator:

**Observations** (19 features):
- 14 price features (returns, RSI, MACD, Bollinger Bands, etc.)
- 4 position features (in_position, entry_price, unrealized_pnl, time_held)
- 1 momentum hint (recent price trend)

**Actions**:
- 0: HOLD - Do nothing
- 1: BUY - Enter long position
- 2: SELL - Exit position

**Rewards**:
- Realized profit/loss (×100 scaling, ×1.5 multiplier for wins)
- Opportunity penalty (-0.005) for missing profitable moves
- Episode bonus (+0.05) for making ≥2 trades
- Momentum reward (±0.01) while holding positions

### Agent (`src/rl/agent.py`)

PPO agent with custom feature extractor:

- **Policy Network**: Multi-scale processing (256→128 dims)
- **Value Network**: Separate architecture for critic
- **Entropy**: 0.05 (high for exploration)
- **Clip Range**: 0.1 (conservative updates)

### Trainer (`src/rl/trainer.py`)

Complete training pipeline:

- Multi-token training (cycles through different price histories)
- Automatic checkpointing every 50K steps
- Evaluation every 10K steps
- TensorBoard logging

## Project Structure

```
ai_model/
├── data/
│   ├── raw/              # Raw CSV token data
│   └── processed/        # Preprocessed train/val/test splits
├── notebooks/
│   └── train_rl_agent.ipynb   # Google Colab training notebook
├── src/
│   ├── config/           # Trading constants and hyperparameters
│   ├── data/             # Data loading and feature extraction
│   ├── rl/               # RL environment, agent, trainer
│   └── utils/            # Helper functions
└── models/               # Saved models (gitignored)
```

## Configuration

Key parameters in `src/config/__init__.py`:

```python
# Trading
FIXED_POSITION_SIZE = 0.01      # 0.01 SOL per trade
TOTAL_FEE_PER_TX = 0.00035      # Jito + gas fees
DELAY_SECONDS = 1               # Transaction confirmation delay

# Features
MIN_HISTORY_LENGTH = 12         # Minimum candles before acting
```

## Training Metrics

Monitor during training:

- **Mean PnL**: Average profit per episode
- **Mean Trades**: Number of trades per episode
- **Win Rate**: Percentage of profitable trades
- **Curriculum Progress**: 0-100% fee difficulty

Example good training:
```
[Step 10000] PnL: 0.0015 | Trades: 4.2 | WinRate: 42% | Curriculum: 20%
[Step 50000] PnL: 0.0032 | Trades: 5.1 | WinRate: 51% | Curriculum: 60%
[Step 100000] PnL: 0.0028 | Trades: 4.8 | WinRate: 48% | Curriculum: 100%
```

## Evaluation

After training, the model is evaluated on held-out tokens:

```python
Mean PnL: 0.0025 ± 0.0018 SOL
Mean Trades: 4.5
Win Rate: 47%
```

Good results:
- Win rate > 45%
- Mean trades 3-8 per episode (active trading)
- Positive mean PnL with low std

## Common Issues

### Policy Collapse (Agent Stops Trading)

**Symptoms**: Mean trades drops to <1, mean PnL approaches 0

**Causes**:
- Curriculum too fast (reduce `curriculum_episodes`)
- Entropy too low (increase `ent_coef`)
- Not enough exploration bonus

**Fix**: The current configuration prevents this with:
- Curriculum learning (1000 episodes)
- High entropy (0.05)
- Opportunity penalties
- Minimum trade requirements

### Unstable Training

**Symptoms**: Reward oscillates wildly, doesn't converge

**Causes**:
- Learning rate too high
- Clip range too large
- Not enough training steps

**Fix**:
- Lower LR (try 5e-5)
- Reduce clip_range (try 0.05)
- Train longer (2M+ steps)

### Poor Evaluation Performance

**Symptoms**: Good training metrics, bad evaluation

**Causes**:
- Overfitting to training tokens
- Evaluation with different fee structure
- Not enough diverse training data

**Fix**:
- Add more training tokens
- Use regularization (higher weight_decay)
- Ensure eval uses full fees

## Advanced Usage

### Custom Reward Function

Edit `src/rl/environment.py` TradingEnvironmentV2.step():

```python
# Add custom reward shaping
if trade_pnl > threshold:
    reward += bonus
```

### Hyperparameter Tuning

Edit `src/rl/trainer.py` train_rl_agent():

```python
model = PPO(
    ...
    ent_coef=0.1,        # Higher exploration
    gamma=0.995,         # Longer-term planning
    n_steps=8192,        # Larger rollout buffer
)
```

### Multi-GPU Training

Use `SubprocVecEnv` instead of `DummyVecEnv`:

```python
from stable_baselines3.common.vec_env import SubprocVecEnv

env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
```

## Research Notes

### Why RL Over Classification?

We initially tried supervised learning with labels:
- BUY: Future profit ≥5% AND drawdown ≥-6%
- SELL: Drawdown ≤-6% OR price rollover
- HOLD: Neither condition

**Results**: 14-34% accuracy (worse than 33% random baseline)

**Problems**:
1. Labels created extreme class imbalance (64% SELL)
2. Features (past prices) can't predict 20-second future
3. Unrealistic thresholds for volatile crypto markets

**RL Solution**: No labels. Agent discovers patterns through trial/error with profit-based rewards.

### Reward Shaping Techniques

1. **Hindsight**: Penalize missing opportunities (future price increased >2%)
2. **Asymmetric**: Bigger reward for wins (×1.5) than losses (×1.0)
3. **Potential-based**: Reward holding profitable positions
4. **Activity bonus**: Reward making minimum number of trades

All designed to prevent "do nothing" strategy.

## Dependencies

```
torch>=2.0.0
stable-baselines3>=2.2.0
gymnasium>=0.29.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
tensorboard>=2.14.0
```

Install with:
```bash
pip install -r requirements.txt
```

## Citation

If you use this code for research, please cite:

```
@misc{gmgn_rl_trader,
  author = {Trading Team},
  title = {GMGN Trading Bot - Reinforcement Learning Approach},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/tr4m0ryp/GMGN_TradingBot}
}
```

## License

MIT License - See ../LICENSE for details
