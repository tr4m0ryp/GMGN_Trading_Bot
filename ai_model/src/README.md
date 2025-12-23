# AI Model Source Code

This directory contains the core implementation of the reinforcement learning trading agent.

## Overview

All core model logic and utilities are here. Notebooks should ONLY import from this directory, never define models directly.

## Module Structure

### `config/`
Trading constants and configuration management.

**Files**:
- `__init__.py` - Trading constants, fee structures, thresholds

**Key Configuration**:
```python
from config import (
    FIXED_POSITION_SIZE,  # 0.01 SOL per trade
    TOTAL_FEE_PER_TX,     # 0.00035 SOL (Jito + gas)
    MIN_HISTORY_LENGTH,   # 12 candles minimum
)
```

### `data/`
Data loading, preprocessing, and feature extraction.

**Files**:
- `preparation.py` - Feature extraction and dataset creation
- `preprocess.py` - CLI script to generate train/val/test splits

**Features** (14 total):
- Price features: log_close, returns (1s/3s/5s), range_ratio
- Volume: volume_log
- Technical indicators: RSI, MACD, Bollinger Bands, VWAP, momentum
- Readiness flags: indicator_ready_short, indicator_ready_long

**Usage**:
```python
from data import load_preprocessed_datasets, extract_features

# Load preprocessed data
train_ds, val_ds, test_ds, metadata = load_preprocessed_datasets('../data/processed')

# Extract features from candles
features = extract_features(candles)  # Returns (n_candles, 14) array
```

### `rl/`
Reinforcement learning environment, agent, and training pipeline.

**Files**:
- `environment.py` - Gym-compatible trading environment with curriculum learning
- `agent.py` - PPO agent with custom feature extractors
- `trainer.py` - Complete training pipeline

**Key Components**:

**TradingEnvironmentV2**:
- Gym environment for single-token trading
- Curriculum learning (gradual fee introduction)
- Hindsight rewards (penalize missed opportunities)
- Asymmetric rewards (×1.5 bonus for wins)

**CurriculumTradingEnvironment**:
- Multi-token training environment
- Automatic difficulty progression
- Tracks aggregate statistics

**Usage**:
```python
from rl import TradingEnvironmentV2, CurriculumTradingEnvironment, train_rl_agent

# Single token environment
env = TradingEnvironmentV2(candles, fee_multiplier=1.0)

# Multi-token curriculum
env = CurriculumTradingEnvironment(
    all_candles,
    initial_fee_mult=0.0,
    target_fee_mult=1.0,
    curriculum_episodes=1000
)

# Complete training
results = train_rl_agent(
    data_dir='../data',
    output_dir='../models/rl',
    total_timesteps=1_000_000,
    curriculum_episodes=1000
)
```

### `utils/`
Helper functions and utilities.

**Files**:
- `__init__.py` - Device management, checkpointing, seeding

**Functions**:
```python
from utils import get_device, set_seed, count_parameters, save_checkpoint

device = get_device()              # Auto-detect cuda/cpu
set_seed(42)                       # Reproducibility
n_params = count_parameters(model) # Count parameters
save_checkpoint(model, path)       # Save model
```

## Architecture

```
Training Flow:

Raw CSV Data
    ↓
data/preprocess.py  →  Train/Val/Test Splits
    ↓
data/preparation.py →  Feature Extraction (14 features)
    ↓
rl/environment.py   →  Trading Environment (Gym)
    ↓
rl/agent.py         →  PPO Agent (Policy + Value Networks)
    ↓
rl/trainer.py       →  Training Loop + Curriculum
    ↓
Trained Model (.zip)
```

## Usage Example

### Training

```python
# Simple training
from rl import train_rl_agent

results = train_rl_agent(
    data_dir='../data',
    output_dir='../models/rl',
    total_timesteps=1_000_000,
    learning_rate=1e-4,
    curriculum_episodes=1000,
    device='cuda'
)

print(f"Mean PnL: {results['final_metrics']['mean_pnl']:.4f}")
print(f"Win Rate: {results['final_metrics']['win_rate']:.1%}")
```

### Inference

```python
from stable_baselines3 import PPO
from rl import TradingEnvironmentV2

# Load model
model = PPO.load('../models/rl/best_model.zip')

# Create environment
env = TradingEnvironmentV2(candles, fee_multiplier=1.0)
obs, _ = env.reset()

# Trade
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

print(f"Episode PnL: {info['total_pnl']:.6f} SOL")
print(f"Trades: {info['n_trades']}")
print(f"Win Rate: {info['win_rate']:.1%}")
```

## CLI Scripts

### Preprocess Data

```bash
cd src
python -m data.preprocess --csv-path ../data/raw/rawdata.csv --output-dir ../data/processed
```

This creates:
- `train_samples.pkl` - Training samples
- `val_samples.pkl` - Validation samples
- `test_samples.pkl` - Test samples
- `metadata.pkl` - Metadata (splits, seed, etc.)

## Development Guidelines

1. **All model logic goes here** (NOT in notebooks)
2. **Use type hints** for all function signatures
3. **Google-style docstrings** for all public functions
4. **Single responsibility** - Keep functions focused
5. **Unit tests** for data processing (future)

### Code Style Example

```python
def calculate_pnl(
    buy_price: float,
    sell_price: float,
    position_size: float = FIXED_POSITION_SIZE,
    fee_per_trade: float = TOTAL_FEE_PER_TX
) -> float:
    """
    Calculate net profit/loss from a trade.

    Args:
        buy_price: Entry price in SOL.
        sell_price: Exit price in SOL.
        position_size: Trade size in SOL. Default from config.
        fee_per_trade: Fee per transaction side. Default from config.

    Returns:
        Net PnL in SOL after fees.

    Example:
        >>> pnl = calculate_pnl(buy_price=1.0, sell_price=1.05)
        >>> print(f"{pnl:.6f} SOL")
        0.000325 SOL
    """
    tokens = position_size / buy_price
    sell_value = tokens * sell_price
    net = sell_value - (2 * fee_per_trade)
    return net - position_size
```

## Debugging

### Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

results = train_rl_agent(..., verbose=2)
```

### TensorBoard Monitoring

```bash
tensorboard --logdir ../models/rl/logs
```

### Key Debug Points

- `environment.py:step()` - Reward calculation
- `trainer.py:train_rl_agent()` - Episode statistics
- `agent.py:predict()` - Action selection

## Testing

Future test structure:

```bash
pytest tests/test_data.py      # Data loading tests
pytest tests/test_env.py       # Environment tests
pytest tests/test_training.py  # Training integration tests
```

## Performance Benchmarks

On T4 GPU:
- **Data preprocessing**: 2-5 seconds (286 tokens)
- **Training**: 1M steps in ~2-3 hours
- **Evaluation**: 20 episodes in ~30 seconds

Memory:
- **PPO model**: ~150MB
- **Rollout buffer**: ~500MB (4096 steps × 4 envs)
- **Total GPU**: ~2GB

## Migration from Classification

Previous classification approach removed. See git history for:
- `models/` - LSTM, Transformer-LSTM architectures
- `training/` - Classification training loop
- `evaluation/` - Classification metrics

Replaced with RL because:
- Classification accuracy: 14-34% (worse than random)
- RL learns directly from profit feedback
- No need for hand-crafted labels

## Future Enhancements

- [ ] Recurrent policies (LSTM/GRU)
- [ ] Multi-asset trading
- [ ] Hierarchical RL
- [ ] Offline RL from historical data
- [ ] Distributional value functions

## License

MIT License - See ../../LICENSE
