"""
LSTM model architecture for variable-length sequence trading.

This module implements the VariableLengthLSTMTrader class, which processes
entire price histories from token discovery using packed sequences for
efficiency. The model outputs HOLD/BUY/SELL (0/1/2) predictions with confidence
scores.

Dependencies:
    torch: Deep learning framework
    torch.nn: Neural network modules
    torch.nn.utils.rnn: Sequence packing utilities

Author: Trading Team
Date: 2025-12-21
"""

from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class VariableLengthLSTMTrader(nn.Module):
    """
    LSTM-based trading model for variable-length sequences.

    This model processes the entire price history from token discovery
    to current time, using packed sequences for efficiency. Outputs
    HOLD/BUY/SELL (0/1/2) predictions with confidence scores.

    The architecture uses 2 LSTM layers with 128 hidden units, followed
    by fully connected layers for classification. Dropout is applied after
    the first fully connected layer for regularization.

    Architecture:
        - LSTM layers: 2 layers with 128 hidden units each
        - Dropout: 0.3 after FC1
        - FC layers: 128 -> 64 -> 3 classes
        - Output: 3-class softmax (0=HOLD, 1=BUY, 2=SELL)

    Attributes:
        input_size: Number of input features per timestep.
        hidden_size: LSTM hidden dimension.
        num_layers: Number of LSTM layers.
        num_classes: Number of output classes.
        dropout: Dropout probability.
        lstm: LSTM module.
        dropout_layer: Dropout layer applied after FC1.
        fc1: First fully connected layer.
        relu: ReLU activation.
        fc2: Output layer.

    Args:
        input_size: Number of features per timestep. Default is 15.
            Features: engineered price/volume returns, indicator deltas, readiness masks, position flag.
        hidden_size: LSTM hidden dimension. Default is 128.
        num_layers: Number of LSTM layers. Default is 2.
        num_classes: Number of output classes. Default is 3 (0=HOLD, 1=BUY, 2=SELL).
        dropout: Dropout probability. Default is 0.3.

    Example:
        >>> model = VariableLengthLSTMTrader(input_size=15, hidden_size=256)
        >>> model = model.cuda()
        >>> predictions, confidence = model(features, lengths)
        >>> print(predictions.shape)
        torch.Size([32, 3])
        >>> print(confidence.shape)
        torch.Size([32])

    Note:
        This model requires packed sequences for variable-length inputs.
        Sequences should be padded before passing to forward().
    """

    def __init__(self,
                 input_size: int = 15,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 num_classes: int = 3,
                 dropout: float = 0.3):
        """Initialize the LSTM trading model."""
        super(VariableLengthLSTMTrader, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout_layer = nn.Dropout(dropout)

        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self,
                x: torch.Tensor,
                lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.

        Processes variable-length sequences using packed sequences for efficiency.
        Returns logits (for CE/focal) and confidence scores.

        Args:
            x: Padded input sequences of shape (batch, max_seq_len, input_size).
            lengths: Actual sequence lengths of shape (batch,).

        Returns:
            Tuple of (logits, confidence):
                - logits: Raw class scores of shape (batch, num_classes).
                - confidence: Maximum softmax probability per sample of shape (batch,).

        Example:
            >>> x = torch.randn(32, 150, 11)
            >>> lengths = torch.randint(30, 150, (32,))
            >>> predictions, confidence = model(x, lengths)
            >>> print(predictions.shape, confidence.shape)
            torch.Size([32, 3]) torch.Size([32])
        """
        batch_size = x.size(0)

        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_output, (hidden, cell) = self.lstm(packed)

        last_hidden = hidden[-1]

        x = self.fc1(last_hidden)
        x = self.relu(x)
        x = self.dropout_layer(x)

        logits = self.fc2(x)

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

            low_confidence = confidence < confidence_threshold
            actions[low_confidence] = 0

        return actions, confidence
