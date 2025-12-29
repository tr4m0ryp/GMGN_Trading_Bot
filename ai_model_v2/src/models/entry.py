"""
Model 2: Entry Timing Optimizer (LSTM/Hybrid).

Time-series model to determine the optimal entry point for tokens
that passed the screener. Uses bidirectional LSTM with attention.

Author: Trading Team
Date: 2025-12-29
"""

from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path

import numpy as np
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
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply self-attention.

        Args:
            x: Input tensor (batch, seq_len, hidden_dim).
            mask: Optional attention mask.

        Returns:
            Attended output (batch, seq_len, hidden_dim).
        """
        # Self-attention with residual connection
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
        self,
        input_dim: int = 14,
        hidden_dim: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.2,
        use_attention: bool = True,
        attention_heads: int = 4,
    ):
        """
        Initialize encoder.

        Args:
            input_dim: Number of input features.
            hidden_dim: LSTM hidden dimension.
            num_layers: Number of LSTM layers.
            bidirectional: Use bidirectional LSTM.
            dropout: Dropout probability.
            use_attention: Whether to use attention.
            attention_heads: Number of attention heads.
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Output dimension after LSTM
        lstm_out_dim = hidden_dim * self.num_directions

        # Attention layer
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionLayer(lstm_out_dim, attention_heads, dropout)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.output_dim = hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        seq_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode sequence.

        Args:
            x: Input tensor (batch, seq_len, input_dim).
            seq_lengths: Actual sequence lengths (batch,).

        Returns:
            Tuple of (sequence_output, final_embedding).
            - sequence_output: (batch, seq_len, hidden_dim)
            - final_embedding: (batch, hidden_dim)
        """
        batch_size = x.size(0)

        # Input projection
        x = self.input_proj(x)

        # Pack sequences for efficient LSTM
        if seq_lengths is not None:
            # Clamp lengths to valid range
            seq_lengths = seq_lengths.cpu().clamp(min=1, max=x.size(1))
            x_packed = pack_padded_sequence(
                x, seq_lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, (h_n, c_n) = self.lstm(x_packed)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, (h_n, c_n) = self.lstm(x)

        # Apply attention
        if self.use_attention:
            # Create mask for attention (True = masked/padded)
            if seq_lengths is not None:
                max_len = lstm_out.size(1)
                mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= seq_lengths.unsqueeze(1)
            else:
                mask = None
            lstm_out = self.attention(lstm_out, mask)

        # Get final embedding (last valid timestep for each sequence)
        if seq_lengths is not None:
            # Gather last valid outputs
            idx = (seq_lengths - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, lstm_out.size(2))
            idx = idx.to(lstm_out.device)
            final = lstm_out.gather(1, idx).squeeze(1)
        else:
            final = lstm_out[:, -1, :]

        # Output projection
        final = self.output_proj(final)

        return lstm_out, final


class EntryModel(nn.Module):
    """
    Complete Model 2: Entry Timing Optimizer.

    Combines encoder with classification head for ENTER_NOW/WAIT/ABORT.
    """

    def __init__(self, config: Optional[EntryConfig] = None):
        """
        Initialize entry model.

        Args:
            config: EntryConfig with hyperparameters.
        """
        super().__init__()
        self.config = config or DEFAULT_ENTRY_CONFIG

        # Encoder
        self.encoder = EntryEncoder(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            bidirectional=self.config.bidirectional,
            dropout=self.config.dropout,
            use_attention=self.config.use_attention,
            attention_heads=self.config.attention_heads,
        )

        # Classification head
        head_layers = []
        prev_dim = self.encoder.output_dim

        for dim in self.config.head_hidden_dims:
            head_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Dropout(self.config.dropout),
            ])
            prev_dim = dim

        head_layers.append(nn.Linear(prev_dim, self.config.num_classes))
        self.classifier = nn.Sequential(*head_layers)

    def forward(
        self,
        x: torch.Tensor,
        seq_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, input_dim).
            seq_lengths: Actual sequence lengths (batch,).

        Returns:
            Class logits (batch, num_classes).
        """
        _, embedding = self.encoder(x, seq_lengths)
        logits = self.classifier(embedding)
        return logits

    def predict(
        self,
        x: torch.Tensor,
        seq_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get predicted class labels.

        Args:
            x: Input tensor.
            seq_lengths: Sequence lengths.

        Returns:
            Predicted class indices (batch,).
        """
        logits = self.forward(x, seq_lengths)
        return logits.argmax(dim=-1)

    def predict_proba(
        self,
        x: torch.Tensor,
        seq_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get predicted class probabilities.

        Args:
            x: Input tensor.
            seq_lengths: Sequence lengths.

        Returns:
            Class probabilities (batch, num_classes).
        """
        logits = self.forward(x, seq_lengths)
        return F.softmax(logits, dim=-1)

    def should_enter(
        self,
        x: torch.Tensor,
        seq_lengths: Optional[torch.Tensor] = None,
        threshold: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Check if should enter based on confidence threshold.

        Args:
            x: Input tensor.
            seq_lengths: Sequence lengths.
            threshold: Confidence threshold.

        Returns:
            Tuple of (should_enter, confidence).
        """
        threshold = threshold or self.config.confidence_threshold
        proba = self.predict_proba(x, seq_lengths)

        # ENTER_NOW is class 1
        enter_proba = proba[:, 1]
        should_enter = enter_proba >= threshold

        return should_enter, enter_proba


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
