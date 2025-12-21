# AI Trading Model

Variable-length LSTM/Transformer model for predicting optimal BUY/SELL/HOLD signals on Solana tokens.

## Overview

This model uses a revolutionary **variable-length sequence approach** where it sees the ENTIRE price history from token discovery to current time, exactly simulating how a real trader watches a live chart evolve.

## Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scikit-learn
```

### 2. Prepare Data

```bash
python src/data_preparation.py --input ../ai_data/data/tokens_2025-12-21.csv --output data/processed/
```

### 3. Train Model (with GPU)

Open `notebooks/train_gpu.ipynb` in Jupyter and run all cells.

The notebook will:
- Import model from `src/model_lstm.py`
- Load processed data
- Train on GPU
- Save best model to `models/best_model.pth`

### 4. Evaluate

```bash
python src/evaluate.py --model models/best_model.pth --test-data data/test/
```

## Directory Structure

```
ai_model/
├── README.md              # This file
├── CLAUDE.md              # Development guidelines and coding rules
├── MODEL_DESIGN.md        # Model architecture specification
├── data/                  # Data directory
│   ├── raw/               # Raw CSV files
│   ├── processed/         # Preprocessed features and labels
│   └── test/              # Test dataset
├── models/                # Saved models
│   ├── checkpoints/       # Training checkpoints
│   └── best_model.pth     # Best performing model
├── src/                   # Python source code (CORE LOGIC HERE)
│   ├── data_preparation.py
│   ├── model_lstm.py
│   ├── model_transformer.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
└── notebooks/             # Jupyter notebooks (GPU ACCESS ONLY)
    └── train_gpu.ipynb
```

## Key Design Principles

### Variable-Length Sequences
Unlike fixed-window approaches (e.g., last 30 seconds), this model sees ALL historical candles:
- At T=31s: sees 31 candles [0-30]
- At T=100s: sees 100 candles [0-99]
- At T=300s: sees 300 candles [0-299]

This provides full lifecycle context, just like a real trader.

### Realistic Training
- Simulates 1-second Jito transaction delay
- Uses worst-case slippage (highest buy, lowest sell)
- Accounts for all fees (~7% per round trip)
- Labels based on NET profit, not gross price movement

### Clean Architecture
- **All model code in `src/` Python files**
- **Notebooks ONLY for GPU access**
- Type hints on all functions
- Comprehensive docstrings

## Model Architecture

### Input
- **Shape**: `(batch, variable_seq_len, 11)`
- **Features**: OHLCV + RSI + MACD + Bollinger Bands + VWAP + Momentum

### Output
- **Predictions**: `(batch, 3)` - BUY/HOLD/SELL probabilities
- **Confidence**: `(batch,)` - max probability

### Architecture Options
1. **LSTM** (recommended): 2 layers, 128 hidden units, dropout 0.3
2. **Transformer**: 4 layers, 8 attention heads, d_model=128

## Performance Targets

| Metric | Target |
|--------|--------|
| Win rate | >60% |
| Avg profit/trade | >18% NET |
| Sharpe ratio | >1.5 |
| Max drawdown | <15% |

## Development Guidelines

See [CLAUDE.md](CLAUDE.md) for complete coding standards and best practices.

**Critical Rules:**
- All model logic goes in `src/` Python files
- Notebooks are ONLY for GPU access, not model definitions
- Use type hints and docstrings everywhere
- Set random seeds for reproducibility

## Documentation

- [CLAUDE.md](CLAUDE.md) - Development guidelines and coding standards
- [MODEL_DESIGN.md](MODEL_DESIGN.md) - Detailed model architecture specification

---

*Last Updated: December 21, 2025*
