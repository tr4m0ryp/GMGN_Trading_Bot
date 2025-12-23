"""
Attention-enhanced LSTM model for trading signal prediction.

This module implements an LSTM with self-attention mechanism that weighs
the importance of different timesteps in the sequence. The attention layer
learns which parts of the price history are most relevant for predicting
trading signals.

Dependencies:
    torch: Deep learning framework
    torch.nn: Neural network modules
    torch.nn.utils.rnn: Sequence packing utilities

Author: Trading Team
Date: 2025-12-22
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class AttentionLSTMTrader(nn.Module):
    """
    LSTM with self-attention for variable-length sequence trading.

    This model processes the entire price history from token discovery
    using an LSTM, then applies self-attention to identify the most
    relevant timesteps for prediction. This allows the model to focus
    on critical price movements (e.g., breakouts, reversals) rather
    than treating all timesteps equally.

    Architecture:
        - LSTM layers: 2 layers with configurable hidden units
        - Attention: Learned attention weights over LSTM outputs
        - FC layers: hidden_size -> 64 -> 3 classes
        - Output: 3-class softmax (0=HOLD, 1=BUY, 2=SELL)

    Attributes:
        input_size: Number of input features per timestep.
        hidden_size: LSTM hidden dimension.
        num_layers: Number of LSTM layers.
        num_classes: Number of output classes.
        dropout: Dropout probability.
        lstm: LSTM module.
        attention: Attention scoring network.
        fc: Output classification network.

    Args:
        input_size: Number of features per timestep. Default is 15.
        hidden_size: LSTM hidden dimension. Default is 128.
        num_layers: Number of LSTM layers. Default is 2.
        num_classes: Number of output classes. Default is 3.
        dropout: Dropout probability. Default is 0.3.

    Example:
        >>> model = AttentionLSTMTrader(input_size=15, hidden_size=128)
        >>> model = model.cuda()
        >>> predictions, confidence = model(features, lengths)
        >>> print(predictions.shape)
        torch.Size([32, 3])
    """

    def __init__(self,
                 input_size: int = 15,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 num_classes: int = 3,
                 dropout: float = 0.3):
        """Initialize the attention LSTM trading model."""
        super(AttentionLSTMTrader, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Attention mechanism: learns to score each timestep
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Output classification network
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self,
                x: torch.Tensor,
                lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention mechanism.

        Processes variable-length sequences using packed sequences for efficiency,
        applies self-attention to weight important timesteps, and outputs logits.

        Args:
            x: Padded input sequences of shape (batch, max_seq_len, input_size).
            lengths: Actual sequence lengths of shape (batch,).

        Returns:
            Tuple of (logits, confidence):
                - logits: Raw class scores of shape (batch, num_classes).
                - confidence: Maximum softmax probability per sample of shape (batch,).

        Example:
            >>> x = torch.randn(32, 150, 15)
            >>> lengths = torch.randint(12, 150, (32,))
            >>> predictions, confidence = model(x, lengths)
            >>> print(predictions.shape, confidence.shape)
            torch.Size([32, 3]) torch.Size([32])
        """
        batch_size = x.size(0)

        # Encode with LSTM
        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

        # Compute attention scores for each timestep
        attn_scores = self.attention(lstm_out).squeeze(-1)  # (batch, seq_len)

        # Mask padding positions to prevent attention on padding
        mask = torch.arange(lstm_out.size(1), device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch, seq_len)

        # Compute weighted sum of LSTM outputs (context vector)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)  # (batch, hidden_size)

        # Classification
        logits = self.fc(context)

        # Confidence scores
        probs = torch.softmax(logits, dim=1)
        confidence = torch.max(probs, dim=1)[0]

        return logits, confidence

    def predict(self,
               x: torch.Tensor,
               lengths: torch.Tensor,
               confidence_threshold: float = 0.7) -> Tuple[torch.Tensor,
                                                            torch.Tensor]:
        """
        Make predictions with confidence filtering.

        Only returns predictions where confidence exceeds the threshold.
        Low-confidence predictions are set to HOLD (class 0).

        Args:
            x: Padded input sequences of shape (batch, max_seq_len, input_size).
            lengths: Actual sequence lengths of shape (batch,).
            confidence_threshold: Minimum confidence for action. Default is 0.7.

        Returns:
            Tuple of (actions, confidence):
                - actions: Predicted actions (0=HOLD, 1=BUY, 2=SELL) of shape (batch,).
                - confidence: Confidence scores of shape (batch,).

        Example:
            >>> actions, conf = model.predict(x, lengths, confidence_threshold=0.75)
            >>> buy_signals = (actions == 1) & (conf > 0.75)
            >>> print(f"High-confidence buy signals: {buy_signals.sum()}")
        """
        self.eval()

        with torch.no_grad():
            predictions, confidence = self.forward(x, lengths)

            actions = torch.argmax(predictions, dim=1)

            # Filter low-confidence predictions to HOLD
            low_confidence = confidence < confidence_threshold
            actions[low_confidence] = 0

        return actions, confidence

    def get_attention_weights(self,
                             x: torch.Tensor,
                             lengths: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for visualization and analysis.

        Returns the learned attention weights showing which timesteps
        the model focuses on for each prediction.

        Args:
            x: Padded input sequences of shape (batch, max_seq_len, input_size).
            lengths: Actual sequence lengths of shape (batch,).

        Returns:
            Attention weights of shape (batch, max_seq_len).

        Example:
            >>> attn = model.get_attention_weights(x, lengths)
            >>> # Visualize attention for first sample
            >>> import matplotlib.pyplot as plt
            >>> plt.plot(attn[0].cpu().numpy())
            >>> plt.title("Attention weights over time")
        """
        self.eval()

        with torch.no_grad():
            batch_size = x.size(0)

            # Encode with LSTM
            packed = pack_padded_sequence(
                x,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False
            )

            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

            # Compute attention scores
            attn_scores = self.attention(lstm_out).squeeze(-1)

            # Mask padding
            mask = torch.arange(lstm_out.size(1), device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

            # Softmax to get weights
            attn_weights = F.softmax(attn_scores, dim=1)

        return attn_weights
