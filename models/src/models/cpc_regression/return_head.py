"""
ProbabilisticReturnHead: Multi-task return and drawdown prediction.

This module implements the prediction head that outputs:
1. Expected return (mean + variance) using Gaussian NLL
2. Expected drawdown (auxiliary task)

The variance output enables uncertainty quantification for
Kelly criterion position sizing.

Dependencies:
    torch

Date: 2025-12-25
"""

from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-export loss and calibration classes for backward compatibility
from .return_layers import gaussian_nll_loss, MultiTaskLoss, CalibrationMetrics


class AttentionPooling(nn.Module):
    """
    Attention-weighted pooling over sequence dimension.

    Learns to weight different timesteps differently when
    aggregating sequence embeddings.

    Args:
        embed_dim: Embedding dimension
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.Tanh(),
            nn.Linear(embed_dim // 4, 1),
        )

    def forward(
        self, z: torch.Tensor, mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pool sequence using learned attention weights.

        Args:
            z: Embeddings [batch, seq_len, embed_dim]
            mask: Boolean mask [batch, seq_len] where True = ignore

        Returns:
            Pooled embeddings [batch, embed_dim]
        """
        scores = self.attention(z).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        pooled = torch.bmm(weights.unsqueeze(1), z).squeeze(1)
        return pooled


class ProbabilisticReturnHead(nn.Module):
    """
    Multi-task prediction head for return and drawdown.

    Outputs:
        - mu: Predicted mean return (unbounded)
        - log_var: Predicted log-variance (for uncertainty)
        - drawdown: Predicted drawdown (optional auxiliary task)

    Architecture:
        Encoder output -> Attention Pooling -> Shared MLP
            -> Mean head (1 output)
            -> Variance head (1 output)
            -> Drawdown head (1 output, optional)

    Args:
        embed_dim: Input embedding dimension. Default is 512.
        hidden_dims: Hidden layer dimensions. Default [256, 128].
        predict_drawdown: Whether to predict drawdown. Default True.
        dropout: Dropout probability. Default 0.1.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dims: list = None,
        predict_drawdown: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.embed_dim = embed_dim
        self.predict_drawdown = predict_drawdown

        self.attention_pool = AttentionPooling(embed_dim)

        layers = []
        in_dim = embed_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                nn.GELU(), nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        self.shared = nn.Sequential(*layers)
        final_dim = hidden_dims[-1]

        self.mean_head = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2), nn.GELU(),
            nn.Linear(final_dim // 2, 1),
        )

        self.var_head = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2), nn.GELU(),
            nn.Linear(final_dim // 2, 1),
        )

        if predict_drawdown:
            self.drawdown_head = nn.Sequential(
                nn.Linear(final_dim, final_dim // 2), nn.GELU(),
                nn.Linear(final_dim // 2, 1),
            )

        self._init_output_layers()

    def _init_output_layers(self):
        """Initialize output layer weights."""
        for head in [self.mean_head, self.var_head]:
            for layer in head:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    nn.init.zeros_(layer.bias)

        if self.predict_drawdown:
            for layer in self.drawdown_head:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    nn.init.zeros_(layer.bias)

    def forward(
        self, z: torch.Tensor, seq_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass for return prediction.

        Args:
            z: Encoder embeddings [batch, seq_len, embed_dim]
            seq_lengths: Actual sequence lengths [batch]

        Returns:
            mu: Predicted mean return [batch, 1]
            log_var: Predicted log-variance [batch, 1]
            drawdown: Predicted drawdown [batch, 1] (if enabled)
        """
        batch_size, seq_len, _ = z.shape

        mask = None
        if seq_lengths is not None:
            device = z.device
            range_tensor = torch.arange(seq_len, device=device)
            mask = range_tensor.unsqueeze(0) >= seq_lengths.unsqueeze(1)

        pooled = self.attention_pool(z, mask)
        shared = self.shared(pooled)
        mu = self.mean_head(shared)
        log_var = self.var_head(shared)

        if self.predict_drawdown:
            drawdown = self.drawdown_head(shared)
            return mu, log_var, drawdown

        return mu, log_var
