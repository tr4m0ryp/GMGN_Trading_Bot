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

    Example:
        >>> encoder = CPCEncoder(input_dim=14, embed_dim=512)
        >>> cpc = CPCModel(encoder, prediction_steps=[1, 2, 3, 5, 10])
        >>> x = torch.randn(32, 128, 14)
        >>> loss, accuracy = cpc.compute_loss(x)
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

        # Autoregressive model: summarizes z_1...z_t into c_t
        self.ar_gru = nn.GRU(
            input_size=self.embed_dim,
            hidden_size=ar_hidden,
            num_layers=1,
            batch_first=True,
        )

        # Prediction heads: one for each prediction step k
        # Maps context c_t to predicted z_{t+k}
        self.predictors = nn.ModuleDict({
            str(k): nn.Sequential(
                nn.Linear(ar_hidden, ar_hidden),
                nn.GELU(),
                nn.Linear(ar_hidden, self.embed_dim),
            )
            for k in prediction_steps
        })

        # Initialize prediction heads
        for predictor in self.predictors.values():
            for layer in predictor:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(
        self,
        x: torch.Tensor,
        seq_lengths: Optional[torch.Tensor] = None,
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
        # Encode all timesteps
        z = self.encoder(x, seq_lengths)  # [batch, seq_len, embed_dim]

        # Compute autoregressive context
        c, _ = self.ar_gru(z)  # [batch, seq_len, ar_hidden]

        return z, c

    def compute_loss(
        self,
        x: torch.Tensor,
        seq_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute InfoNCE contrastive loss.

        For each prediction step k:
            - Positive: true future embedding z_{t+k}
            - Negatives: embeddings from other positions in batch

        Args:
            x: Input sequences [batch, seq_len, input_dim]
            seq_lengths: Actual sequence lengths [batch]

        Returns:
            loss: Scalar InfoNCE loss
            metrics: Dictionary with per-step losses and accuracies
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Forward pass
        z, c = self.forward(x, seq_lengths)

        total_loss = 0.0
        metrics = {}

        for k in self.prediction_steps:
            # Skip if sequence is too short for this prediction step
            if seq_len <= k:
                continue

            # Number of valid prediction positions
            n_valid = seq_len - k

            # Context at positions 0 to (seq_len - k - 1)
            c_t = c[:, :n_valid, :]  # [batch, n_valid, ar_hidden]

            # True future embeddings at positions k to (seq_len - 1)
            z_future = z[:, k:, :]  # [batch, n_valid, embed_dim]

            # Predict future embeddings
            predictor = self.predictors[str(k)]
            z_pred = predictor(c_t)  # [batch, n_valid, embed_dim]

            # Normalize for cosine similarity
            z_pred_norm = F.normalize(z_pred, dim=-1)
            z_future_norm = F.normalize(z_future, dim=-1)

            # Compute InfoNCE loss
            # Reshape for batch-wise comparison
            # z_pred: [batch * n_valid, embed_dim]
            # z_future: [batch * n_valid, embed_dim]
            z_pred_flat = z_pred_norm.reshape(-1, self.embed_dim)
            z_future_flat = z_future_norm.reshape(-1, self.embed_dim)

            # Similarity matrix: [batch * n_valid, batch * n_valid]
            # Each row i: similarity of prediction i to all futures
            logits = torch.matmul(z_pred_flat, z_future_flat.t()) / self.temperature

            # Labels: diagonal entries are positives
            n_samples = batch_size * n_valid
            labels = torch.arange(n_samples, device=device)

            # Cross-entropy loss (InfoNCE)
            loss_k = F.cross_entropy(logits, labels)

            # Accuracy: how often is the correct future the top prediction
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                acc_k = (preds == labels).float().mean().item()

            total_loss += loss_k
            metrics[f'loss_k{k}'] = loss_k.item()
            metrics[f'acc_k{k}'] = acc_k

        # Average over prediction steps
        n_steps = len([k for k in self.prediction_steps if seq_len > k])
        if n_steps > 0:
            total_loss = total_loss / n_steps

        metrics['loss_total'] = total_loss.item()
        metrics['mean_accuracy'] = sum(
            metrics.get(f'acc_k{k}', 0) for k in self.prediction_steps
        ) / max(n_steps, 1)

        return total_loss, metrics

    def compute_loss_efficient(
        self,
        x: torch.Tensor,
        seq_lengths: Optional[torch.Tensor] = None,
        n_negatives: int = 64,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute InfoNCE loss with sampled negatives (memory efficient).

        For large batches, computing full similarity matrix is expensive.
        This version samples a subset of negatives.

        Args:
            x: Input sequences [batch, seq_len, input_dim]
            seq_lengths: Actual sequence lengths [batch]
            n_negatives: Number of negative samples per positive

        Returns:
            loss: Scalar InfoNCE loss
            metrics: Dictionary with per-step losses and accuracies
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Forward pass
        z, c = self.forward(x, seq_lengths)

        total_loss = 0.0
        metrics = {}

        for k in self.prediction_steps:
            if seq_len <= k:
                continue

            n_valid = seq_len - k

            # Context and future embeddings
            c_t = c[:, :n_valid, :]
            z_future = z[:, k:, :]

            # Predict future
            predictor = self.predictors[str(k)]
            z_pred = predictor(c_t)

            # Normalize
            z_pred_norm = F.normalize(z_pred, dim=-1)
            z_future_norm = F.normalize(z_future, dim=-1)

            # Flatten
            z_pred_flat = z_pred_norm.reshape(-1, self.embed_dim)
            z_future_flat = z_future_norm.reshape(-1, self.embed_dim)
            n_samples = z_pred_flat.size(0)

            # Sample negatives for each positive
            losses = []
            correct = 0

            for i in range(n_samples):
                # Positive: the matching future
                pos_sim = torch.sum(z_pred_flat[i] * z_future_flat[i]) / self.temperature

                # Sample random negatives (exclude positive)
                neg_indices = torch.randperm(n_samples, device=device)[:n_negatives + 1]
                neg_indices = neg_indices[neg_indices != i][:n_negatives]

                neg_z = z_future_flat[neg_indices]
                neg_sim = torch.matmul(z_pred_flat[i:i+1], neg_z.t()) / self.temperature
                neg_sim = neg_sim.squeeze(0)

                # InfoNCE: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
                logits = torch.cat([pos_sim.unsqueeze(0), neg_sim])
                loss_i = F.cross_entropy(logits.unsqueeze(0), torch.zeros(1, dtype=torch.long, device=device))
                losses.append(loss_i)

                # Accuracy
                if logits.argmax() == 0:
                    correct += 1

            loss_k = torch.stack(losses).mean()
            acc_k = correct / n_samples

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
        self,
        x: torch.Tensor,
        seq_lengths: Optional[torch.Tensor] = None,
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
        z, _ = self.forward(x, seq_lengths)  # [batch, seq_len, embed_dim]

        if pooling == 'last':
            return self.encoder.get_last_embedding(x, seq_lengths)
        elif pooling == 'mean':
            if seq_lengths is not None:
                # Masked mean
                mask = self.encoder._create_padding_mask(seq_lengths, z.size(1))
                mask = ~mask  # Invert: True = valid
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

    Example:
        >>> model = CPCRegressionModel(encoder, return_head)
        >>> mu, log_var, dd = model(x, seq_lengths)
    """

    def __init__(
        self,
        encoder: CPCEncoder,
        return_head: nn.Module,
    ):
        super().__init__()
        self.encoder = encoder
        self.return_head = return_head

    def forward(
        self,
        x: torch.Tensor,
        seq_lengths: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for return prediction.

        Args:
            x: Input sequences [batch, seq_len, input_dim]
            seq_lengths: Actual sequence lengths [batch]

        Returns:
            Output from return_head (mu, log_var, and optionally drawdown)
        """
        z = self.encoder(x, seq_lengths)
        return self.return_head(z, seq_lengths)

    def predict(
        self,
        x: torch.Tensor,
        seq_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict returns with uncertainty.

        Args:
            x: Input sequences [batch, seq_len, input_dim]
            seq_lengths: Actual sequence lengths [batch]

        Returns:
            Dictionary with 'mu', 'sigma', 'log_var', and optionally 'drawdown'
        """
        with torch.no_grad():
            outputs = self.forward(x, seq_lengths)

        if len(outputs) == 3:
            mu, log_var, drawdown = outputs
            sigma = torch.exp(0.5 * log_var)
            return {
                'mu': mu,
                'log_var': log_var,
                'sigma': sigma,
                'drawdown': drawdown,
            }
        else:
            mu, log_var = outputs
            sigma = torch.exp(0.5 * log_var)
            return {
                'mu': mu,
                'log_var': log_var,
                'sigma': sigma,
            }
