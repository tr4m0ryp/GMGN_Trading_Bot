"""
Utility functions for trading model.

This module provides helper functions for reproducibility, logging,
checkpoint management, and other common operations used throughout
the trading model codebase.

Dependencies:
    torch: Deep learning framework
    numpy: Numerical computations
    random: Python random number generation

Author: Trading Team
Date: 2025-12-21
"""

import random
import logging
from typing import Optional, Dict, Any
from pathlib import Path

import torch
import numpy as np


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
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_file: Optional[str] = None,
                  level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging configuration.

    Configures logging with console and optional file output.

    Args:
        log_file: Path to log file. If None, only console logging is enabled.
        level: Logging level. Default is logging.INFO.

    Returns:
        Configured logger instance.

    Example:
        >>> logger = setup_logging('training.log')
        >>> logger.info('Training started')
    """
    logger = logging.getLogger('trading_model')
    logger.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def save_checkpoint(model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   loss: float,
                   filepath: str,
                   **kwargs: Any) -> None:
    """
    Save model checkpoint.

    Saves model state, optimizer state, and training metadata to disk.

    Args:
        model: PyTorch model to save.
        optimizer: Optimizer to save.
        epoch: Current epoch number.
        loss: Current loss value.
        filepath: Path where checkpoint will be saved.
        **kwargs: Additional metadata to save (e.g., metrics, config).

    Example:
        >>> save_checkpoint(model, optimizer, epoch=10, loss=0.5,
        ...                 filepath='checkpoints/model_epoch10.pth',
        ...                 val_accuracy=0.85)
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        **kwargs
    }

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath: str,
                   model: torch.nn.Module,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   device: str = 'cpu') -> Dict[str, Any]:
    """
    Load model checkpoint.

    Loads model state, optimizer state, and training metadata from disk.

    Args:
        filepath: Path to checkpoint file.
        model: PyTorch model to load state into.
        optimizer: Optional optimizer to load state into.
        device: Device to load checkpoint to ('cpu' or 'cuda').

    Returns:
        Dictionary containing checkpoint metadata (epoch, loss, etc.).

    Example:
        >>> metadata = load_checkpoint('checkpoints/best_model.pth',
        ...                            model, optimizer, device='cuda')
        >>> print(f"Loaded model from epoch {metadata['epoch']}")
    """
    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint


def get_device() -> str:
    """
    Get the appropriate device for training.

    Returns:
        'cuda' if GPU is available, 'cpu' otherwise.

    Example:
        >>> device = get_device()
        >>> model = model.to(device)
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model.

    Returns:
        Total number of trainable parameters.

    Example:
        >>> model = VariableLengthLSTMTrader()
        >>> print(f"Total parameters: {count_parameters(model):,}")
        Total parameters: 245,123
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
