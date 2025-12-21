# Source Code Directory

## Overview

This directory contains ALL core model logic and utilities. Notebooks should ONLY import from here, never define models directly.

## Package Layout

### config/
Configuration management:
- `DEFAULT_CONFIG` - Default configuration dictionary
- `get_config()` - Get configuration with all hyperparameters
- Trading constants (fees, position sizes, thresholds)

### data/
Data preprocessing and feature extraction:
- `preparation.py` - Load raw data, extract features, build datasets
- `preprocess.py` - CLI preprocessing script for cached datasets

### models/
LSTM model architecture:
- `lstm.py` - `VariableLengthLSTMTrader`
- Handles variable-length sequences with pack_padded_sequence
- Outputs HOLD/BUY/SELL (0/1/2) predictions with confidence scores

### training/
Training logic:
- `train.py` - Training loop, validation, checkpointing

### evaluation/
Evaluation and backtesting:
- `evaluate.py` - Metrics, backtests, reporting

### utils/
Helper functions:
- `set_seed()` - Set random seeds for reproducibility
- `save_checkpoint()` - Save model checkpoints
- `load_checkpoint()` - Load model checkpoints
- `get_device()` - Get available device (CPU/CUDA)
- Logging utilities

## Usage Example

```python
# In notebook: notebooks/train_gpu.ipynb
import torch
from torch.utils.data import DataLoader

from config import get_config
from data import load_preprocessed_datasets, collate_variable_length
from models import VariableLengthLSTMTrader
from training import train_model

# Load config and data
config = get_config()
train_ds, val_ds, _ = load_preprocessed_datasets('../data/processed')

train_loader = DataLoader(
    train_ds,
    batch_size=config['training']['batch_size'],
    shuffle=True,
    collate_fn=collate_variable_length,
)
val_loader = DataLoader(
    val_ds,
    batch_size=config['training']['batch_size'],
    shuffle=False,
    collate_fn=collate_variable_length,
)

# Initialize model
model = VariableLengthLSTMTrader().cuda()

# Train
train_model(model, train_loader, val_loader, config, device='cuda')
```

## CLI Scripts

Run the preprocessing script from `ai_model/src`:

```bash
python -m data.preprocess --csv-path ../data/raw/rawdata.csv --output-dir ../data/processed
```

## Development Rules

1. All model definitions go here (NOT in notebooks)
2. Use type hints for all function signatures
3. Add Google-style docstrings to all functions
4. Keep functions focused and single-purpose
5. Write unit tests for data processing functions
