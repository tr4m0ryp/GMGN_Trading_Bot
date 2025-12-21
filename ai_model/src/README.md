# Source Code Directory

## Overview

This directory contains ALL core model logic and utilities. Notebooks should ONLY import from here, never define models directly.

## File Structure

### data_preparation.py
Data preprocessing and feature extraction:
- `load_raw_data()` - Load CSV token data
- `extract_features()` - Extract OHLCV + technical indicators
- `prepare_realistic_training_data()` - Create training samples with full historical context
- `split_data()` - Split into train/val/test sets

### model_lstm.py
LSTM model architecture:
- `VariableLengthLSTMTrader` - Main model class
- Handles variable-length sequences with pack_padded_sequence

### model_transformer.py
Transformer model architecture:
- `PositionalEncoding` - Positional encoding layer
- `TransformerTrader` - Main model class
- Handles variable-length with attention masks

### train.py
Training logic:
- `train_model()` - Main training loop
- `validate()` - Validation function
- `EarlyStopping` - Early stopping callback
- Checkpoint saving

### evaluate.py
Evaluation and backtesting:
- `evaluate_model()` - Calculate metrics on test set
- `backtest()` - Realistic backtesting with fees
- `calculate_sharpe_ratio()` - Risk-adjusted returns
- Performance reporting

### utils.py
Helper functions:
- `set_seed()` - Set random seeds for reproducibility
- `collate_variable_length()` - DataLoader collate function
- `create_padding_mask()` - Attention mask creation
- Logging utilities

## Usage Example

```python
# In notebook: notebooks/train_gpu.ipynb
from src.model_lstm import VariableLengthLSTMTrader
from src.train import train_model
from src.data_preparation import load_training_data

# Load data
train_data, val_data = load_training_data()

# Initialize model
model = VariableLengthLSTMTrader().cuda()

# Train
train_model(model, train_data, val_data, device='cuda')
```

## Development Rules

1. All model definitions go here (NOT in notebooks)
2. Use type hints for all function signatures
3. Add Google-style docstrings to all functions
4. Keep functions focused and single-purpose
5. Write unit tests for data processing functions
