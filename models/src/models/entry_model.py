"""
Entry Model: Network architecture for entry timing.

Contains AttentionLayer, EntryEncoder, and EntryModel classes
for the LSTM/Hybrid entry timing optimizer.

Author: Trading Team
Date: 2025-12-29
"""

from typing import Dict, Any, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..config import EntryConfig, DEFAULT_ENTRY_CONFIG


class AttentionLayer(nn.Module):
    """
    Multi-head self-attention layer for sequence modeling.

    Applies attention over the sequence to focus on important timesteps.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        """
        Initialize attention layer.

        Args:
            hidden_dim: Dimension of hidden states.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply self-attention with residual connection."""
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        return self.norm(x + attn_out)


class EntryEncoder(nn.Module):
    """
    Bidirectional LSTM encoder with attention for entry timing.

    Architecture:
    1. Input projection
    2. Bidirectional LSTM layers
    3. Self-attention
    4. Output projection
    """

    def __init__(
        self, input_dim: int = 14, hidden_dim: int = 128,
        num_layers: int = 2, bidirectional: bool = True,
        dropout: float = 0.2, use_attention: bool = True,
        attention_heads: int = 4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )

        lstm_out_dim = hidden_dim * self.num_directions

        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionLayer(lstm_out_dim, attention_heads, dropout)

        self.output_proj = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.output_dim = hidden_dim

    def forward(
        self, x: torch.Tensor, seq_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode sequence.

        Returns:
            Tuple of (sequence_output, final_embedding).
        """
        batch_size = x.size(0)
        x = self.input_proj(x)

        if seq_lengths is not None:
            seq_lengths = seq_lengths.cpu().clamp(min=1, max=x.size(1))
            x_packed = pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)
            lstm_out, (h_n, c_n) = self.lstm(x_packed)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, (h_n, c_n) = self.lstm(x)

        if self.use_attention:
            if seq_lengths is not None:
                max_len = lstm_out.size(1)
                mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= seq_lengths.unsqueeze(1)
            else:
                mask = None
            lstm_out = self.attention(lstm_out, mask)

        if seq_lengths is not None:
            idx = (seq_lengths - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, lstm_out.size(2))
            idx = idx.to(lstm_out.device)
            final = lstm_out.gather(1, idx).squeeze(1)
        else:
            final = lstm_out[:, -1, :]

        final = self.output_proj(final)
        return lstm_out, final


class EntryModel(nn.Module):
    """
    Complete Model 2: Entry Timing Optimizer.

    Combines encoder with classification head for ENTER_NOW/WAIT/ABORT.
    """

    def __init__(self, config: Optional[EntryConfig] = None):
        super().__init__()
        self.config = config or DEFAULT_ENTRY_CONFIG

        self.encoder = EntryEncoder(
            input_dim=self.config.input_dim, hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers, bidirectional=self.config.bidirectional,
            dropout=self.config.dropout, use_attention=self.config.use_attention,
            attention_heads=self.config.attention_heads,
        )

        head_layers = []
        prev_dim = self.encoder.output_dim
        for dim in self.config.head_hidden_dims:
            head_layers.extend([
                nn.Linear(prev_dim, dim), nn.LayerNorm(dim),
                nn.ReLU(), nn.Dropout(self.config.dropout),
            ])
            prev_dim = dim
        head_layers.append(nn.Linear(prev_dim, self.config.num_classes))
        self.classifier = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor, seq_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass. Returns class logits."""
        _, embedding = self.encoder(x, seq_lengths)
        return self.classifier(embedding)

    def predict(self, x: torch.Tensor, seq_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get predicted class labels."""
        return self.forward(x, seq_lengths).argmax(dim=-1)

    def predict_proba(self, x: torch.Tensor, seq_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get predicted class probabilities."""
        return F.softmax(self.forward(x, seq_lengths), dim=-1)

    def should_enter(
        self, x: torch.Tensor, seq_lengths: Optional[torch.Tensor] = None,
        threshold: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Check if should enter based on confidence threshold."""
        threshold = threshold or self.config.confidence_threshold
        proba = self.predict_proba(x, seq_lengths)
        enter_proba = proba[:, 1]
        return enter_proba >= threshold, enter_proba
