"""
CPCEncoder: Shared encoder for CPC pretraining and return regression.

This module implements a Bidirectional LSTM + Multi-Head Attention encoder
that outputs per-timestep embeddings for contrastive learning.

Architecture:
    Input [batch, seq_len, 14] -> Input Projection
        -> Bidirectional LSTM (captures temporal patterns)
        -> Multi-Head Self-Attention (long-range dependencies)
        -> Feed-Forward + Residual
        -> Output [batch, seq_len, embed_dim]

Dependencies:
    torch, numpy

Date: 2025-12-25
"""

import math
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CPCEncoder(nn.Module):
    """
    Encoder for CPC pretraining and return regression.

    Outputs per-timestep embeddings suitable for:
    1. CPC pretraining: Predict future embeddings from context
    2. Return regression: Pool embeddings for prediction

    Unlike the RL feature extractor, this processes full sequences
    and outputs embeddings for each timestep.

    Args:
        input_dim: Input feature dimension. Default is 14.
        hidden_dim: LSTM hidden dimension. Default is 256.
        embed_dim: Output embedding dimension. Default is 512.
        lstm_layers: Number of LSTM layers. Default is 2.
        n_heads: Number of attention heads. Default is 8.
        ff_dim: Feed-forward hidden dimension. Default is 2048.
        dropout: Dropout probability. Default is 0.1.

    Example:
        >>> encoder = CPCEncoder(input_dim=14, hidden_dim=256)
        >>> x = torch.randn(32, 128, 14)  # [batch, seq_len, features]
        >>> seq_lengths = torch.ones(32) * 128
        >>> z = encoder(x, seq_lengths)  # [batch, seq_len, 512]
    """

    def __init__(
        self,
        input_dim: int = 14,
        hidden_dim: int = 256,
        embed_dim: int = 512,
        lstm_layers: int = 2,
        n_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.lstm_layers = lstm_layers

        # Input projection: 14 features -> hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Bidirectional LSTM for temporal patterns
        # Output: hidden_dim * 2 (bidirectional)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # LSTM output dimension (bidirectional doubles it)
        lstm_out_dim = hidden_dim * 2

        # Multi-head self-attention for long-range dependencies
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_out_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(lstm_out_dim)

        # Feed-forward network with residual connection
        self.ffn = nn.Sequential(
            nn.Linear(lstm_out_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, lstm_out_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(lstm_out_dim)

        # Output projection to embedding dimension
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_out_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def _create_padding_mask(
        self,
        seq_lengths: torch.Tensor,
        max_len: int,
    ) -> torch.Tensor:
        """
        Create attention mask for padded sequences.

        Args:
            seq_lengths: Actual sequence lengths [batch]
            max_len: Maximum sequence length

        Returns:
            Boolean mask [batch, max_len] where True = padded (ignore)
        """
        batch_size = seq_lengths.size(0)
        # Create range tensor [1, max_len]
        range_tensor = torch.arange(max_len, device=seq_lengths.device)
        # Expand to [batch, max_len] and compare
        mask = range_tensor.unsqueeze(0) >= seq_lengths.unsqueeze(1)
        return mask

    def forward(
        self,
        x: torch.Tensor,
        seq_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode input sequences to per-timestep embeddings.

        Args:
            x: Input tensor [batch, seq_len, input_dim]
            seq_lengths: Actual sequence lengths [batch] (optional)
                        If None, assumes no padding.

        Returns:
            Embeddings [batch, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Default: assume no padding
        if seq_lengths is None:
            seq_lengths = torch.full(
                (batch_size,), seq_len, dtype=torch.long, device=device
            )

        # Input projection
        x = self.input_proj(x)  # [batch, seq_len, hidden_dim]

        # Pack sequences for efficient LSTM processing
        # Sort by length (required for pack_padded_sequence)
        sorted_lengths, sort_idx = seq_lengths.sort(descending=True)
        sorted_x = x[sort_idx]

        # Pack padded sequence
        packed = pack_padded_sequence(
            sorted_x,
            sorted_lengths.cpu(),
            batch_first=True,
            enforce_sorted=True,
        )

        # LSTM forward pass
        lstm_out, _ = self.lstm(packed)

        # Unpack sequence
        lstm_out, _ = pad_packed_sequence(
            lstm_out, batch_first=True, total_length=seq_len
        )

        # Unsort to original order
        _, unsort_idx = sort_idx.sort()
        lstm_out = lstm_out[unsort_idx]  # [batch, seq_len, hidden*2]

        # Create attention mask for padded positions
        attn_mask = self._create_padding_mask(seq_lengths, seq_len)

        # Self-attention with residual connection
        attn_out, _ = self.attention(
            lstm_out, lstm_out, lstm_out,
            key_padding_mask=attn_mask,
        )
        attn_out = self.attn_norm(lstm_out + attn_out)

        # Feed-forward with residual connection
        ffn_out = self.ffn(attn_out)
        ffn_out = self.ffn_norm(attn_out + ffn_out)

        # Output projection
        z = self.output_proj(ffn_out)  # [batch, seq_len, embed_dim]

        return z

    def get_last_embedding(
        self,
        x: torch.Tensor,
        seq_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get embedding at the last valid timestep for each sequence.

        Useful for regression when you want a single embedding per sequence.

        Args:
            x: Input tensor [batch, seq_len, input_dim]
            seq_lengths: Actual sequence lengths [batch]

        Returns:
            Last valid embeddings [batch, embed_dim]
        """
        batch_size = x.shape[0]
        device = x.device

        # Get all embeddings
        z = self.forward(x, seq_lengths)  # [batch, seq_len, embed_dim]

        if seq_lengths is None:
            # No padding: return last timestep
            return z[:, -1, :]

        # Extract embedding at last valid position for each sequence
        # seq_lengths - 1 gives the index of last valid position
        indices = (seq_lengths - 1).long().clamp(min=0)
        indices = indices.view(batch_size, 1, 1).expand(-1, -1, z.size(-1))
        last_z = z.gather(1, indices).squeeze(1)  # [batch, embed_dim]

        return last_z


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for sequence positions.

    Can be optionally added to encoder for explicit position information.

    Args:
        d_model: Embedding dimension
        max_len: Maximum sequence length
        dropout: Dropout probability

    Example:
        >>> pe = PositionalEncoding(d_model=512, max_len=128)
        >>> x = torch.randn(32, 128, 512)
        >>> x = pe(x)
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
