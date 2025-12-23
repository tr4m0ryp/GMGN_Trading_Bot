"""
Advanced Transformer-LSTM hybrid model for trading signal prediction.

This module implements a powerful hybrid architecture combining:
- Bidirectional LSTM for sequential feature extraction
- Multi-head self-attention transformer encoder
- Positional encoding for temporal awareness
- Residual connections and layer normalization
- Multi-scale temporal convolutions

Designed to utilize full GPU capacity (T4 16GB) for maximum accuracy.

Dependencies:
    torch: Deep learning framework
    torch.nn: Neural network modules
    math: Mathematical functions for positional encoding

Author: Trading Team
Date: 2025-12-23
"""

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for sequence position awareness.

    Adds positional information to input embeddings using sinusoidal functions
    at different frequencies. This allows the model to understand the temporal
    position of each timestep in the sequence.

    Args:
        d_model: Embedding dimension.
        max_len: Maximum sequence length. Default is 1000.
        dropout: Dropout probability. Default is 0.1.
    """

    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor."""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class MultiScaleConv(nn.Module):
    """
    Multi-scale temporal convolution for capturing patterns at different time scales.

    Applies parallel convolutions with different kernel sizes to capture both
    short-term patterns (rapid price changes) and longer-term trends.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels per scale.
        dropout: Dropout probability. Default is 0.1.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)

        self.norm = nn.LayerNorm(out_channels * 4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-scale convolutions.

        Args:
            x: Input tensor of shape (batch, seq_len, channels).

        Returns:
            Concatenated multi-scale features of shape (batch, seq_len, out_channels * 4).
        """
        x = x.transpose(1, 2)  # (batch, channels, seq_len)

        c1 = F.gelu(self.conv1(x))
        c3 = F.gelu(self.conv3(x))
        c5 = F.gelu(self.conv5(x))
        c7 = F.gelu(self.conv7(x))

        out = torch.cat([c1, c3, c5, c7], dim=1)  # (batch, out_channels * 4, seq_len)
        out = out.transpose(1, 2)  # (batch, seq_len, out_channels * 4)

        return self.dropout(self.norm(out))


class TransformerEncoderBlock(nn.Module):
    """
    Single transformer encoder block with multi-head attention and feed-forward network.

    Implements pre-layer normalization variant for more stable training.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        d_ff: Feed-forward network dimension.
        dropout: Dropout probability. Default is 0.1.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through transformer encoder block.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            mask: Attention mask for padding positions.

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        # Pre-norm multi-head attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, key_padding_mask=mask)
        x = x + self.dropout1(attn_out)

        # Pre-norm feed-forward with residual
        x = x + self.ff(self.norm2(x))

        return x


class AdvancedTransformerLSTMTrader(nn.Module):
    """
    Advanced hybrid Transformer-LSTM trading model.

    This model combines the best of both architectures:
    - Multi-scale convolutions for pattern detection
    - Bidirectional LSTM for sequential modeling
    - Transformer encoder for global attention
    - Sophisticated classification head

    Architecture:
        Input (batch, seq_len, 14 features)
            -> Multi-scale Conv (captures patterns at different timescales)
            -> Bidirectional LSTM (sequential encoding)
            -> Positional Encoding
            -> Transformer Encoder Blocks (global attention)
            -> Temporal Attention Pooling
            -> Classification Head (with residual)
            -> Output (3 classes: HOLD/BUY/SELL)

    This architecture can effectively utilize 4-8GB GPU memory
    depending on batch size and sequence length.

    Args:
        input_size: Number of input features per timestep. Default is 14.
        hidden_size: LSTM and transformer hidden dimension. Default is 512.
        num_lstm_layers: Number of bidirectional LSTM layers. Default is 3.
        num_transformer_layers: Number of transformer encoder layers. Default is 4.
        num_heads: Number of attention heads. Default is 8.
        num_classes: Number of output classes. Default is 3.
        dropout: Dropout probability. Default is 0.3.
        ff_mult: Feed-forward dimension multiplier. Default is 4.

    Example:
        >>> model = AdvancedTransformerLSTMTrader(input_size=14, hidden_size=512)
        >>> model = model.cuda()
        >>> logits, confidence = model(features, lengths)
        >>> print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    """

    def __init__(
        self,
        input_size: int = 14,
        hidden_size: int = 512,
        num_lstm_layers: int = 3,
        num_transformer_layers: int = 4,
        num_heads: int = 8,
        num_classes: int = 3,
        dropout: float = 0.3,
        ff_mult: int = 4,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # Multi-scale convolution for pattern detection
        conv_out_channels = hidden_size // 4
        self.multi_scale_conv = MultiScaleConv(input_size, conv_out_channels, dropout)

        # Input projection
        self.input_proj = nn.Linear(conv_out_channels * 4, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)

        # Bidirectional LSTM for sequential encoding
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,  # Bidirectional doubles this
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.lstm_norm = nn.LayerNorm(hidden_size)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            hidden_size, max_len=1000, dropout=dropout
        )

        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    d_model=hidden_size,
                    n_heads=num_heads,
                    d_ff=hidden_size * ff_mult,
                    dropout=dropout,
                )
                for _ in range(num_transformer_layers)
            ]
        )

        # Temporal attention pooling
        self.temporal_attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

        # Classification head with residual
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Kaiming initialization."""
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                if "lstm" in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            x: Padded input sequences of shape (batch, max_seq_len, input_size).
            lengths: Actual sequence lengths of shape (batch,).

        Returns:
            Tuple of (logits, confidence):
                - logits: Raw class scores of shape (batch, num_classes).
                - confidence: Maximum softmax probability per sample of shape (batch,).
        """
        batch_size = x.size(0)
        max_seq_len = x.size(1)
        device = x.device

        # Move lengths to device for mask creation, keep CPU copy for pack_padded_sequence
        lengths_device = lengths.to(device)
        lengths_cpu = lengths.cpu().clamp(min=1, max=max_seq_len)

        # Multi-scale convolution
        x = self.multi_scale_conv(x)

        # Input projection
        x = self.input_proj(x)
        x = self.input_norm(x)

        # Pack for LSTM - pack_padded_sequence requires CPU lengths
        packed = pack_padded_sequence(
            x,
            lengths_cpu,
            batch_first=True,
            enforce_sorted=False,
        )

        # Bidirectional LSTM
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

        # Normalize and add positional encoding
        x = self.lstm_norm(lstm_out)
        x = self.pos_encoder(x)

        # Create padding mask for transformer (use device lengths)
        mask = torch.arange(x.size(1), device=device).unsqueeze(
            0
        ) >= lengths_device.unsqueeze(1)

        # Transformer encoder layers
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, mask)

        # Temporal attention pooling
        attn_scores = self.temporal_attn(x).squeeze(-1)  # (batch, seq_len)
        attn_scores = attn_scores.masked_fill(mask, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=1)

        # Weighted sum
        context = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(
            1
        )  # (batch, hidden_size)

        # Classification
        logits = self.classifier(context)

        # Confidence scores
        probs = torch.softmax(logits, dim=1)
        confidence = torch.max(probs, dim=1)[0]

        return logits, confidence

    def predict(
        self, x: torch.Tensor, lengths: torch.Tensor, confidence_threshold: float = 0.7
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with confidence filtering.

        Args:
            x: Padded input sequences of shape (batch, max_seq_len, input_size).
            lengths: Actual sequence lengths of shape (batch,).
            confidence_threshold: Minimum confidence for action. Default is 0.7.

        Returns:
            Tuple of (actions, confidence):
                - actions: Predicted actions (0=HOLD, 1=BUY, 2=SELL) of shape (batch,).
                - confidence: Confidence scores of shape (batch,).
        """
        self.eval()

        with torch.no_grad():
            logits, confidence = self.forward(x, lengths)
            actions = torch.argmax(logits, dim=1)

            low_confidence = confidence < confidence_threshold
            actions[low_confidence] = 0

        return actions, confidence

    def get_attention_weights(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Get temporal attention weights for visualization.

        Args:
            x: Padded input sequences of shape (batch, max_seq_len, input_size).
            lengths: Actual sequence lengths of shape (batch,).

        Returns:
            Attention weights of shape (batch, max_seq_len).
        """
        self.eval()

        with torch.no_grad():
            batch_size = x.size(0)
            max_seq_len = x.size(1)
            device = x.device

            # Move lengths to device for mask creation, keep CPU copy for pack_padded_sequence
            lengths_device = lengths.to(device)
            lengths_cpu = lengths.cpu().clamp(min=1, max=max_seq_len)

            x = self.multi_scale_conv(x)
            x = self.input_proj(x)
            x = self.input_norm(x)

            packed = pack_padded_sequence(
                x,
                lengths_cpu,
                batch_first=True,
                enforce_sorted=False,
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

            x = self.lstm_norm(lstm_out)
            x = self.pos_encoder(x)

            mask = torch.arange(x.size(1), device=device).unsqueeze(
                0
            ) >= lengths_device.unsqueeze(1)

            for transformer_layer in self.transformer_layers:
                x = transformer_layer(x, mask)

            attn_scores = self.temporal_attn(x).squeeze(-1)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))
            attn_weights = F.softmax(attn_scores, dim=1)

        return attn_weights


class LightweightTransformerTrader(nn.Module):
    """
    Lightweight transformer-only model for faster training.

    Uses only transformer architecture without LSTM for situations
    where training speed is prioritized over maximum accuracy.

    Args:
        input_size: Number of input features per timestep. Default is 14.
        hidden_size: Model dimension. Default is 256.
        num_layers: Number of transformer layers. Default is 6.
        num_heads: Number of attention heads. Default is 8.
        num_classes: Number of output classes. Default is 3.
        dropout: Dropout probability. Default is 0.2.

    Example:
        >>> model = LightweightTransformerTrader(input_size=14, hidden_size=256)
        >>> model = model.cuda()
        >>> logits, confidence = model(features, lengths)
    """

    def __init__(
        self,
        input_size: int = 14,
        hidden_size: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        num_classes: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            hidden_size, max_len=1000, dropout=dropout
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Temporal attention pooling
        self.temporal_attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model."""
        device = x.device
        max_seq_len = x.size(1)

        # Input projection
        x = self.input_proj(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Create padding mask
        mask = torch.arange(max_seq_len, device=device).unsqueeze(
            0
        ) >= lengths.unsqueeze(1)

        # Transformer encoder
        x = self.transformer(x, src_key_padding_mask=mask)

        # Temporal attention pooling
        attn_scores = self.temporal_attn(x).squeeze(-1)
        attn_scores = attn_scores.masked_fill(mask, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=1)

        context = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)

        # Classification
        logits = self.classifier(context)

        probs = torch.softmax(logits, dim=1)
        confidence = torch.max(probs, dim=1)[0]

        return logits, confidence

    def predict(
        self, x: torch.Tensor, lengths: torch.Tensor, confidence_threshold: float = 0.7
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make predictions with confidence filtering."""
        self.eval()

        with torch.no_grad():
            logits, confidence = self.forward(x, lengths)
            actions = torch.argmax(logits, dim=1)

            low_confidence = confidence < confidence_threshold
            actions[low_confidence] = 0

        return actions, confidence
