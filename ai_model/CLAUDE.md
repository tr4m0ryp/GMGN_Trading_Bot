# AI Model Development Guidelines

## Primary Language: Python

This project focuses on **Python with PyTorch** for deep learning model development. Python provides the best ecosystem for machine learning with extensive libraries and GPU acceleration support.

## Hardware Specification

**GPU**: NVIDIA Tesla T4 (16GB VRAM)
- CUDA Compute Capability: 7.5
- Tensor Cores: Available for mixed precision training
- Memory Bandwidth: 320 GB/s

**Performance Optimization**:
- Maximize GPU utilization with proper batch sizing
- Use mixed precision training (FP16) for faster computation
- Enable TensorCore acceleration where possible
- Optimize DataLoader with multiple workers and pin memory
- Use GPU-accelerated libraries (cuDNN, cuBLAS)

## Project Overview

This directory contains the **AI Trading Model** for GMGN token trading. The model uses variable-length LSTM/Transformer architectures to predict optimal BUY/SELL/HOLD signals based on full historical price data.

## Directory Structure

```
ai_model/
├── CLAUDE.md              # This file - development guidelines
├── MODEL_DESIGN.md        # Model architecture and design specification
├── data/                  # Input data directory
│   ├── raw/               # Raw CSV token data
│   ├── processed/         # Processed training data (features, labels)
│   └── test/              # Test dataset
├── models/                # Saved model weights and checkpoints
│   ├── checkpoints/       # Training checkpoints
│   └── best_model.pth     # Best performing model
├── src/                   # Python source files (CORE MODEL CODE)
│   ├── data_preparation.py    # Data preprocessing and feature extraction
│   ├── model_lstm.py          # LSTM model architecture
│   ├── model_transformer.py   # Transformer model architecture
│   ├── train.py               # Training logic
│   ├── evaluate.py            # Evaluation and backtesting
│   └── utils.py               # Helper functions
└── notebooks/             # Jupyter notebooks (GPU ACCESS ONLY)
    └── train_gpu.ipynb    # Notebook wrapper for GPU training
```

## Critical Architecture Rules

### 1. Notebook Usage Policy

**IMPORTANT: Notebooks are ONLY for GPU access, NOT for model code.**

- Notebooks should contain MINIMAL code
- Use notebooks ONLY to:
  - Import model classes from `src/`
  - Load data
  - Call training functions
  - Access GPU resources
- ALL model architecture, training logic, and utilities MUST be in `.py` files in `src/`

**Example (CORRECT):**
```python
# notebooks/train_gpu.ipynb
from src.model_lstm import VariableLengthLSTMTrader
from src.train import train_model
from src.data_preparation import load_training_data

# Load data
train_data, val_data = load_training_data()

# Initialize model
model = VariableLengthLSTMTrader().cuda()

# Train (function defined in src/train.py)
train_model(model, train_data, val_data, device='cuda')
```

**Example (WRONG):**
```python
# notebooks/train_gpu.ipynb
# DO NOT define model classes directly in notebooks!
class LSTMModel(nn.Module):  # ❌ WRONG - should be in src/
    def __init__(self):
        ...
```

## Code Quality Standards

### Professional Development Practices
- Act as a professional high-end developer
- Follow strict code rules and best practices
- Thoroughly check all code before submission
- Ensure code is production-ready and maintainable
- Verify proper GPU memory management
- Check for tensor shape mismatches and gradient issues
- Ensure reproducibility with proper seed setting

### 2. Python Coding Standards

#### Language and Style
- **Primary Language**: Python 3.9+
- **Style Guide**: PEP 8 (strictly enforced)
- **Type Hints**: Required for all function signatures
- **Docstrings**: Google-style docstrings for all functions/classes
- **Indentation**: 4 spaces (no tabs)
- **Line length**: Maximum 100 characters
- **NO emojis** in code, comments, docstrings, or commit messages

#### Naming Conventions
- **Functions**: lowercase with underscores: `prepare_training_data()`
- **Classes**: PascalCase: `VariableLengthLSTMTrader`
- **Constants**: UPPERCASE: `FIXED_POSITION_SIZE = 0.01`
- **Private methods**: prefix with underscore: `_calculate_loss()`
- **Module-level variables**: lowercase: `default_config = {...}`
- **Temporary variables**: descriptive names (no single letters except i, j, k in loops)

#### Code Quality Standards
- Maximum line length: 100 characters
- Use meaningful variable names (no single letters except loop indices)
- One import per line
- Group imports: stdlib, third-party, local
- NO emojis in code, comments, or docstrings
- Add inline comments only when logic is not self-evident
- Avoid magic numbers - use named constants

### 3. Machine Learning Best Practices

#### Model Architecture
- Define all models as `nn.Module` subclasses in `src/`
- Separate architecture definition from training logic
- Use configuration dictionaries for hyperparameters
- Support both CPU and GPU execution

#### Data Pipeline
- Use PyTorch `Dataset` and `DataLoader` classes
- Implement custom collate functions for variable-length sequences
- Normalize/standardize features appropriately
- Split data: 80% train, 10% validation, 10% test

#### Training
- Use separate train/validation/test sets
- Implement early stopping
- Save checkpoints regularly
- Log metrics to TensorBoard or Weights & Biases
- Use gradient clipping for stability
- Set random seeds for reproducibility

#### Evaluation
- Never evaluate on training data
- Use proper metrics: accuracy, precision, recall, F1
- Implement backtesting with realistic fee simulation
- Calculate Sharpe ratio and max drawdown

### 4. File Organization Rules

#### src/data_preparation.py
```python
"""Data preprocessing and feature extraction."""

def load_raw_data(csv_path: str) -> List[Dict]:
    """Load raw token data from CSV."""
    pass

def extract_features(candles: List[Dict]) -> np.ndarray:
    """Extract OHLCV and technical indicators."""
    pass

def prepare_realistic_training_data(token_candles: List[Dict]) -> List[Dict]:
    """Create training samples with full historical context."""
    pass
```

#### src/model_lstm.py
```python
"""LSTM model for variable-length sequence trading."""

import torch.nn as nn

class VariableLengthLSTMTrader(nn.Module):
    """LSTM-based trading model."""

    def __init__(self, input_size: int = 11, hidden_size: int = 128):
        super().__init__()
        # Model definition

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        # Forward pass
        pass
```

#### src/train.py
```python
"""Training logic and utilities."""

def train_model(model: nn.Module, train_loader, val_loader,
                device: str = 'cuda', epochs: int = 100):
    """Train the model with early stopping."""
    pass

def validate(model: nn.Module, val_loader, criterion, device: str):
    """Validate model on validation set."""
    pass
```

## Code Documentation Requirements

### File Header Comments

Every Python file must include a module-level docstring at the top:

**Sample Module Docstring (Data Preparation):**
```python
"""
Data preprocessing and feature extraction for trading model.

This module implements data loading, feature extraction, and training sample
generation with full historical context. Each sample contains all price history
from token discovery to current timestamp, simulating real market observation.

Dependencies:
    numpy: Numerical computations
    pandas: Data manipulation
    torch: Tensor operations

Author: Trading Team
Date: 2025-12-21
"""
```

**Sample Module Docstring (Model Architecture):**
```python
"""
LSTM model architecture for variable-length sequence trading.

This module implements the VariableLengthLSTMTrader class, which processes
entire price histories from token discovery using packed sequences for
efficiency. The model outputs BUY/HOLD/SELL predictions with confidence scores.

Dependencies:
    torch: Deep learning framework
    torch.nn: Neural network modules
    torch.nn.utils.rnn: Sequence packing utilities

Author: Trading Team
Date: 2025-12-21
"""
```

### Function Docstrings

Every function must include a Google-style docstring explaining its purpose, parameters, and return value:

**Sample Function Docstring (Complex Function):**
```python
def prepare_realistic_training_data(token_candles: List[Dict],
                                   min_history: int = 30) -> List[Dict]:
    """
    Prepare training data with full historical context.

    Creates training samples where each sample contains ALL price history
    from token discovery to current timestamp, simulating real market
    conditions. Includes realistic Jito fee simulation and 1-second
    execution delay.

    Args:
        token_candles: List of OHLCV candles for a single token.
            Each candle dict must contain: 'o', 'h', 'l', 'c', 'v' keys.
        min_history: Minimum number of candles required before generating
            samples. Default is 30 (30 seconds of history).

    Returns:
        List of training samples, each containing:
            - features: np.ndarray of shape (seq_len, 11)
            - label: int (0=HOLD, 1=BUY, 2=SELL)
            - seq_length: int (actual sequence length)
            - timestamp: int (seconds since discovery)
            - buy_price: float (simulated execution price)
            - potential_profit_pct: float (NET profit potential)

    Raises:
        ValueError: If token_candles is empty or contains invalid data.
        KeyError: If required OHLCV keys are missing from candles.

    Example:
        >>> candles = load_token_data('token_ABC.csv')
        >>> samples = prepare_realistic_training_data(candles, min_history=30)
        >>> print(f"Generated {len(samples)} training samples")
        Generated 285 training samples

    Note:
        This function simulates worst-case execution with 1-second delay.
        Buy orders use highest price in delay window, sell orders use lowest.
        All profit calculations include Jito fees (~7% per round trip).
    """
    pass
```

**Sample Function Docstring (Simple Function):**
```python
def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.

    Sets seeds for torch, numpy, and Python random to ensure deterministic
    behavior across runs. Also configures cuDNN for deterministic execution.

    Args:
        seed: Random seed value. Default is 42.

    Example:
        >>> set_seed(42)
        >>> # All random operations will now be reproducible
    """
    pass
```

**Sample Class Docstring:**
```python
class VariableLengthLSTMTrader(nn.Module):
    """
    LSTM-based trading model for variable-length sequences.

    This model processes the entire price history from token discovery
    to current time, using packed sequences for efficiency. Outputs
    BUY/SELL/HOLD predictions with confidence scores.

    The architecture uses 2 LSTM layers with 128 hidden units, followed
    by fully connected layers for classification. Dropout is applied for
    regularization.

    Architecture:
        - LSTM layers: 2 layers with 128 hidden units each
        - Dropout: 0.3 between LSTM and FC layers
        - FC layers: 128 -> 64 -> 3 classes
        - Output: 3-class softmax (BUY/HOLD/SELL)

    Attributes:
        hidden_size: LSTM hidden dimension
        num_layers: Number of LSTM layers
        lstm: LSTM module
        fc1: First fully connected layer
        fc2: Output layer

    Args:
        input_size: Number of features per timestep. Default is 11.
            Features: OHLCV (5) + RSI + MACD + BB_upper + BB_lower + VWAP + Momentum
        hidden_size: LSTM hidden dimension. Default is 128.
        num_layers: Number of LSTM layers. Default is 2.
        num_classes: Number of output classes. Default is 3 (BUY/HOLD/SELL).

    Example:
        >>> model = VariableLengthLSTMTrader(input_size=11, hidden_size=256)
        >>> model = model.cuda()
        >>> predictions, confidence = model(features, lengths)
        >>> print(predictions.shape)  # (batch_size, 3)
        >>> print(confidence.shape)   # (batch_size,)

    Note:
        This model requires packed sequences for variable-length inputs.
        Use pack_padded_sequence before passing to forward().
    """
    pass
```

### Inline Comments

Use inline comments sparingly, only when the code logic is not self-evident:

**Good inline comments:**
```python
# Pack sequences for efficient LSTM processing with variable lengths
packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

# Use highest price in delay window to simulate worst-case buy slippage
buy_price = max(c['high'] for c in delay_candles)

# Calculate NET profit after Jito fees (~7% round trip)
net_profit = gross_profit - (2 * TOTAL_FEE_PER_TX)
```

**Bad inline comments (stating the obvious):**
```python
# Increment i
i += 1

# Set x to 5
x = 5

# Loop through items
for item in items:
    pass
```

## GPU Optimization for Tesla T4

### Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

# Initialize gradient scaler for mixed precision
scaler = GradScaler()

def train_step(model, batch, optimizer):
    """Training step with automatic mixed precision."""
    optimizer.zero_grad()

    # Enable autocasting for FP16
    with autocast():
        predictions, _ = model(batch['features'], batch['seq_lengths'])
        loss = criterion(predictions, batch['labels'])

    # Scale loss and backward pass
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return loss.item()
```

### Optimized DataLoader Configuration
```python
train_loader = DataLoader(
    train_dataset,
    batch_size=64,              # Maximize T4 16GB VRAM utilization
    shuffle=True,
    num_workers=4,              # Parallel data loading
    pin_memory=True,            # Faster CPU-to-GPU transfer
    persistent_workers=True,    # Keep workers alive between epochs
    prefetch_factor=2,          # Prefetch 2 batches per worker
)
```

### Memory-Efficient Gradient Accumulation
```python
# Simulate larger batch size with gradient accumulation
ACCUMULATION_STEPS = 4  # Effective batch = 64 * 4 = 256

for i, batch in enumerate(train_loader):
    with autocast():
        loss = model(batch) / ACCUMULATION_STEPS

    scaler.scale(loss).backward()

    if (i + 1) % ACCUMULATION_STEPS == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

### 6. Dependencies

**Core Libraries:**
```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
```

**Optional (for advanced features):**
```
tensorboard>=2.14.0
wandb>=0.15.0
tqdm>=4.65.0
```

**Installation:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scikit-learn
```

### 7. Configuration Management

Use configuration dictionaries or YAML files:

```python
# src/config.py
DEFAULT_CONFIG = {
    'model': {
        'type': 'lstm',  # or 'transformer'
        'input_size': 11,
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.3,
    },
    'training': {
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 100,
        'early_stopping_patience': 10,
    },
    'data': {
        'min_history_length': 30,
        'train_split': 0.8,
        'val_split': 0.1,
        'test_split': 0.1,
    },
}
```

### 8. Testing Requirements

- Write unit tests for data preprocessing functions
- Test model forward pass with dummy data
- Validate output shapes and value ranges
- Test variable-length sequence handling
- Verify gradient flow (no NaN/Inf)

### 9. Version Control

**Git Commit Messages:**
```
feat: add LSTM model implementation
fix: correct sequence padding in collate_fn
refactor: separate feature extraction logic
docs: update model architecture documentation
```

**What to Commit:**
- All `.py` files in `src/`
- Configuration files
- Documentation (`.md` files)
- Requirements file

**What NOT to Commit:**
- Model weights (`models/*.pth`)
- Processed data (`data/processed/`)
- Checkpoint files
- Jupyter notebook outputs

### 10. Performance Optimization

- Use DataLoader with `num_workers > 0` for parallel data loading
- Implement gradient accumulation for large models
- Use mixed precision training (torch.cuda.amp)
- Profile code to identify bottlenecks
- Batch sequences by similar lengths to reduce padding

### 11. Reproducibility

```python
# Set seeds at the start of every script
import torch
import numpy as np
import random

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### 12. Error Handling

- Check for NaN/Inf in loss values
- Validate input shapes before forward pass
- Handle empty batches gracefully
- Log errors with context information
- Implement proper exception handling

## Model Development Workflow

1. **Data Preparation** (`src/data_preparation.py`)
   - Load raw CSV data
   - Extract features with full historical context
   - Create train/val/test splits
   - Save processed data

2. **Model Implementation** (`src/model_lstm.py` or `src/model_transformer.py`)
   - Define model architecture
   - Implement forward pass
   - Add helper methods

3. **Training Setup** (`src/train.py`)
   - Define training loop
   - Implement validation
   - Add checkpointing and early stopping

4. **Notebook Wrapper** (`notebooks/train_gpu.ipynb`)
   - Import from `src/`
   - Initialize model and data
   - Call training functions
   - Monitor on GPU

5. **Evaluation** (`src/evaluate.py`)
   - Backtest on test set
   - Calculate performance metrics
   - Generate reports

## Critical Success Factors

1. **Variable-Length Sequences**: Model sees full history from token discovery
2. **GPU Optimization**: Maximize Tesla T4 with mixed precision, proper batching
3. **Clean Code**: All logic in `.py` files, notebooks only for GPU
4. **Type Safety**: Use type hints everywhere
5. **Documentation**: Comprehensive docstrings with Google-style format
6. **NO Emojis**: Strictly forbidden in code, comments, and documentation
7. **Reproducibility**: Set seeds, log hyperparameters
8. **Validation**: Separate val/test sets, realistic backtesting
9. **Professional Standards**: Follow strict coding rules and best practices

## Summary

This AI model development follows professional software engineering practices with:
- **Python** as primary language with strict PEP 8 compliance
- **NVIDIA Tesla T4 GPU** optimization for maximum performance
- **NO emojis** policy across all code and documentation
- **Comprehensive documentation** with module-level and function docstrings
- **Clean architecture** separating notebooks (GPU only) from model code (src/)
- **Variable-length sequences** for realistic market simulation

---

*Last Updated: December 21, 2025*
*Hardware: NVIDIA Tesla T4 (16GB VRAM)*
