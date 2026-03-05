"""
CPC Layers: Efficient InfoNCE loss computation.

Provides the memory-efficient variant of InfoNCE loss
using sampled negatives for large batch sizes.

Dependencies:
    torch

Date: 2025-12-25
"""

from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_infonce_efficient(
    z_pred_flat: torch.Tensor,
    z_future_flat: torch.Tensor,
    temperature: float,
    n_negatives: int = 64,
) -> Tuple[torch.Tensor, float]:
    """
    Compute InfoNCE loss with sampled negatives (memory efficient).

    For large batches, computing full similarity matrix is expensive.
    This version samples a subset of negatives.

    Args:
        z_pred_flat: Predicted embeddings [N, embed_dim] (normalized)
        z_future_flat: Future embeddings [N, embed_dim] (normalized)
        temperature: Softmax temperature
        n_negatives: Number of negative samples per positive

    Returns:
        loss: Scalar loss
        accuracy: Fraction of correct predictions
    """
    n_samples = z_pred_flat.size(0)
    device = z_pred_flat.device

    losses = []
    correct = 0

    for i in range(n_samples):
        # Positive: the matching future
        pos_sim = torch.sum(z_pred_flat[i] * z_future_flat[i]) / temperature

        # Sample random negatives (exclude positive)
        neg_indices = torch.randperm(n_samples, device=device)[:n_negatives + 1]
        neg_indices = neg_indices[neg_indices != i][:n_negatives]

        neg_z = z_future_flat[neg_indices]
        neg_sim = torch.matmul(z_pred_flat[i:i+1], neg_z.t()) / temperature
        neg_sim = neg_sim.squeeze(0)

        # InfoNCE: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
        logits = torch.cat([pos_sim.unsqueeze(0), neg_sim])
        loss_i = F.cross_entropy(
            logits.unsqueeze(0),
            torch.zeros(1, dtype=torch.long, device=device)
        )
        losses.append(loss_i)

        if logits.argmax() == 0:
            correct += 1

    loss = torch.stack(losses).mean()
    accuracy = correct / n_samples
    return loss, accuracy


def compute_infonce_full(
    z_pred_flat: torch.Tensor,
    z_future_flat: torch.Tensor,
    temperature: float,
) -> Tuple[torch.Tensor, float]:
    """
    Compute full InfoNCE contrastive loss.

    Uses full similarity matrix (more accurate but more memory).

    Args:
        z_pred_flat: Predicted embeddings [N, embed_dim] (normalized)
        z_future_flat: Future embeddings [N, embed_dim] (normalized)
        temperature: Softmax temperature

    Returns:
        loss: Scalar loss
        accuracy: Fraction of correct predictions
    """
    n_samples = z_pred_flat.size(0)
    device = z_pred_flat.device

    # Full similarity matrix: [N, N]
    logits = torch.matmul(z_pred_flat, z_future_flat.t()) / temperature

    # Labels: diagonal entries are positives
    labels = torch.arange(n_samples, device=device)
    loss = F.cross_entropy(logits, labels)

    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        accuracy = (preds == labels).float().mean().item()

    return loss, accuracy
