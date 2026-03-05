"""
CPC Trainer Utilities: encoder loading for Phase 2 fine-tuning.

Extracted from cpc_trainer.py to keep files under 300 lines.

Dependencies:
    torch

Date: 2025-12-25
"""

from typing import Tuple

import torch

from cpc_regression.encoder import CPCEncoder
from cpc_regression.config import CPCConfig


def load_pretrained_encoder(
    model_path: str,
    device: str = 'cuda',
) -> Tuple[CPCEncoder, CPCConfig]:
    """
    Load pretrained encoder for Phase 2 fine-tuning.

    Args:
        model_path: Path to saved CPC model
        device: Device to load model on

    Returns:
        Tuple of (encoder, config)
    """
    checkpoint = torch.load(model_path, map_location=device)
    config_dict = checkpoint['config']
    config = CPCConfig(**config_dict)

    encoder = CPCEncoder(
        input_dim=config.input_dim, hidden_dim=config.hidden_dim,
        embed_dim=config.embed_dim, lstm_layers=config.lstm_layers,
        n_heads=config.n_heads, ff_dim=config.ff_dim, dropout=config.dropout,
    )

    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder = encoder.to(device)

    print(f"Loaded pretrained encoder from {model_path}")
    print(f"  Embed dim: {config.embed_dim}")
    print(f"  Hidden dim: {config.hidden_dim}")

    return encoder, config
