"""
Model 2: Entry Timing Optimizer - Factory and IO functions.

Provides model creation, initialization, save/load utilities
for the Entry timing model.

Author: Trading Team
Date: 2025-12-29
"""

from typing import Dict, Any, Tuple, Optional
from pathlib import Path

import torch
import torch.nn as nn

from ..config import EntryConfig, DEFAULT_ENTRY_CONFIG
from .entry_model import AttentionLayer, EntryEncoder, EntryModel


def create_entry_model(
    config: Optional[EntryConfig] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> EntryModel:
    """
    Create and initialize entry model.

    Args:
        config: Model configuration.
        device: Device to place model on.

    Returns:
        Initialized EntryModel.
    """
    model = EntryModel(config)
    model = model.to(device)

    # Initialize weights
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

    return model


def save_entry_model(
    model: EntryModel,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save entry model checkpoint.

    Args:
        model: Model to save.
        path: Save path.
        optimizer: Optional optimizer state.
        epoch: Current epoch.
        metrics: Optional metrics dictionary.
    """
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "config": model.config,
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics or {},
    }

    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(state, save_path)


def load_entry_model(
    path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[EntryModel, Dict[str, Any]]:
    """
    Load entry model from checkpoint.

    Args:
        path: Path to checkpoint.
        device: Device to load model to.

    Returns:
        Tuple of (model, checkpoint_info).
    """
    checkpoint = torch.load(path, map_location=device)

    config = checkpoint.get("config", DEFAULT_ENTRY_CONFIG)
    model = EntryModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    info = {
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {}),
    }

    return model, info
