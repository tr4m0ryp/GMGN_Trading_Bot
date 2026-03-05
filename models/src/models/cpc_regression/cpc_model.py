"""
CPCModel: Contrastive Predictive Coding for self-supervised pretraining.

This module implements CPC with InfoNCE loss for learning meaningful
representations from price sequences without labels.

Key concept:
    - Encode sequences to embeddings z_t
    - Use autoregressive model to summarize context c_t
    - Predict future embeddings z_{t+k} from c_t
    - Use contrastive loss to distinguish true future from negatives

Dependencies:
    torch, numpy

Date: 2025-12-25
"""

from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import CPCEncoder
from .cpc_layers import compute_infonce_full, compute_infonce_efficient


class CPCModel(nn.Module):
    """
    Contrastive Predictive Coding model for self-supervised learning.

    Uses InfoNCE loss to learn representations by predicting future
    embeddings from context.

    Architecture:
        Encoder: sequence -> per-timestep embeddings z_t
        AR Model: z_1...z_t -> context c_t (GRU)
        Predictors: c_t -> predicted z_{t+k} for each k

    Args:
        encoder: CPCEncoder instance
        ar_hidden: Autoregressive GRU hidden size. Default is 256.
        prediction_steps: List of future steps to predict. Default [1,2,3,5,10].
        temperature: Softmax temperature for InfoNCE. Default is 0.07.
    """

    def __init__(
        self,
        encoder: CPCEncoder,
        ar_hidden: int = 256,
        prediction_steps: List[int] = None,
        temperature: float = 0.07,
    ):
        super().__init__()

        if prediction_steps is None:
            prediction_steps = [1, 2, 3, 5, 10]

        self.encoder = encoder
        self.embed_dim = encoder.embed_dim
        self.ar_hidden = ar_hidden
        self.prediction_steps = prediction_steps
        self.temperature = temperature

        self.ar_gru = nn.GRU(
            input_size=self.embed_dim, hidden_size=ar_hidden,
            num_layers=1, batch_first=True,
        )

        self.predictors = nn.ModuleDict({
            str(k): nn.Sequential(
                nn.Linear(ar_hidden, ar_hidden), nn.GELU(),
                nn.Linear(ar_hidden, self.embed_dim),
            )
            for k in prediction_steps
        })

        for predictor in self.predictors.values():
            for layer in predictor:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(
        self, x: torch.Tensor, seq_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode sequences and compute AR context.

        Args:
            x: Input sequences [batch, seq_len, input_dim]
            seq_lengths: Actual sequence lengths [batch]

        Returns:
            z: Per-timestep embeddings [batch, seq_len, embed_dim]
            c: AR context vectors [batch, seq_len, ar_hidden]
        """
        z = self.encoder(x, seq_lengths)
        c, _ = self.ar_gru(z)
        return z, c

    def compute_loss(
        self, x: torch.Tensor, seq_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute InfoNCE contrastive loss.

        Args:
            x: Input sequences [batch, seq_len, input_dim]
            seq_lengths: Actual sequence lengths [batch]

        Returns:
            loss: Scalar InfoNCE loss
            metrics: Dictionary with per-step losses and accuracies
        """
        batch_size, seq_len, _ = x.shape
        z, c = self.forward(x, seq_lengths)

        total_loss = 0.0
        metrics = {}

        for k in self.prediction_steps:
            if seq_len <= k:
                continue

            n_valid = seq_len - k
            c_t = c[:, :n_valid, :]
            z_future = z[:, k:, :]

            predictor = self.predictors[str(k)]
            z_pred = predictor(c_t)

            z_pred_norm = F.normalize(z_pred, dim=-1)
            z_future_norm = F.normalize(z_future, dim=-1)

            z_pred_flat = z_pred_norm.reshape(-1, self.embed_dim)
            z_future_flat = z_future_norm.reshape(-1, self.embed_dim)

            loss_k, acc_k = compute_infonce_full(
                z_pred_flat, z_future_flat, self.temperature
            )

            total_loss += loss_k
            metrics[f'loss_k{k}'] = loss_k.item()
            metrics[f'acc_k{k}'] = acc_k

        n_steps = len([k for k in self.prediction_steps if seq_len > k])
        if n_steps > 0:
            total_loss = total_loss / n_steps

        metrics['loss_total'] = total_loss.item()
        metrics['mean_accuracy'] = sum(
            metrics.get(f'acc_k{k}', 0) for k in self.prediction_steps
        ) / max(n_steps, 1)

        return total_loss, metrics

    def compute_loss_efficient(
        self, x: torch.Tensor, seq_lengths: Optional[torch.Tensor] = None,
        n_negatives: int = 64,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute InfoNCE loss with sampled negatives (memory efficient).

        Args:
            x: Input sequences [batch, seq_len, input_dim]
            seq_lengths: Actual sequence lengths [batch]
            n_negatives: Number of negative samples per positive

        Returns:
            loss: Scalar InfoNCE loss
            metrics: Dictionary with per-step losses and accuracies
        """
        batch_size, seq_len, _ = x.shape
        z, c = self.forward(x, seq_lengths)

        total_loss = 0.0
        metrics = {}

        for k in self.prediction_steps:
            if seq_len <= k:
                continue

            n_valid = seq_len - k
            c_t = c[:, :n_valid, :]
            z_future = z[:, k:, :]

            predictor = self.predictors[str(k)]
            z_pred = predictor(c_t)

            z_pred_norm = F.normalize(z_pred, dim=-1)
            z_future_norm = F.normalize(z_future, dim=-1)

            z_pred_flat = z_pred_norm.reshape(-1, self.embed_dim)
            z_future_flat = z_future_norm.reshape(-1, self.embed_dim)

            loss_k, acc_k = compute_infonce_efficient(
                z_pred_flat, z_future_flat, self.temperature, n_negatives
            )

            total_loss += loss_k
            metrics[f'loss_k{k}'] = loss_k.item()
            metrics[f'acc_k{k}'] = acc_k

        n_steps = len([k for k in self.prediction_steps if seq_len > k])
        if n_steps > 0:
            total_loss = total_loss / n_steps

        metrics['loss_total'] = total_loss.item()
        metrics['mean_accuracy'] = sum(
            metrics.get(f'acc_k{k}', 0) for k in self.prediction_steps
        ) / max(n_steps, 1)

        return total_loss, metrics

    def get_representations(
        self, x: torch.Tensor, seq_lengths: Optional[torch.Tensor] = None,
        pooling: str = 'last',
    ) -> torch.Tensor:
        """
        Get learned representations for downstream tasks.

        Args:
            x: Input sequences [batch, seq_len, input_dim]
            seq_lengths: Actual sequence lengths [batch]
            pooling: How to aggregate: 'last', 'mean', 'max'

        Returns:
            Representations [batch, embed_dim]
        """
        z, _ = self.forward(x, seq_lengths)

        if pooling == 'last':
            return self.encoder.get_last_embedding(x, seq_lengths)
        elif pooling == 'mean':
            if seq_lengths is not None:
                mask = self.encoder._create_padding_mask(seq_lengths, z.size(1))
                mask = ~mask
                mask = mask.unsqueeze(-1).float()
                z_masked = z * mask
                return z_masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            return z.mean(dim=1)
        elif pooling == 'max':
            if seq_lengths is not None:
                mask = self.encoder._create_padding_mask(seq_lengths, z.size(1))
                z = z.masked_fill(mask.unsqueeze(-1), float('-inf'))
            return z.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")


class CPCRegressionModel(nn.Module):
    """
    Combined CPC encoder + regression heads for inference.

    This model wraps the pretrained CPC encoder and adds
    return/drawdown prediction heads.

    Args:
        encoder: Pretrained CPCEncoder
        return_head: ProbabilisticReturnHead for predictions
    """

    def __init__(self, encoder: CPCEncoder, return_head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.return_head = return_head

    def forward(self, x: torch.Tensor, seq_lengths: Optional[torch.Tensor] = None):
        """Forward pass for return prediction."""
        z = self.encoder(x, seq_lengths)
        return self.return_head(z, seq_lengths)

    def predict(
        self, x: torch.Tensor, seq_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Predict returns with uncertainty."""
        with torch.no_grad():
            outputs = self.forward(x, seq_lengths)

        if len(outputs) == 3:
            mu, log_var, drawdown = outputs
            sigma = torch.exp(0.5 * log_var)
            return {'mu': mu, 'log_var': log_var, 'sigma': sigma, 'drawdown': drawdown}
        else:
            mu, log_var = outputs
            sigma = torch.exp(0.5 * log_var)
            return {'mu': mu, 'log_var': log_var, 'sigma': sigma}
