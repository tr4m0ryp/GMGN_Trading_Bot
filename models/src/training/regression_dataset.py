"""
Regression Dataset: Data loading utilities for Phase 2 training.

Provides dataset class, collate function, and balanced sampler
for return regression training.

Dependencies:
    torch, numpy

Date: 2025-12-25
"""

from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


class RegressionDataset(Dataset):
    """
    Dataset for return regression training.

    Uses both features and return/drawdown targets.

    Args:
        samples: List of sample dictionaries
        max_seq_len: Maximum sequence length. Default 128.
    """

    def __init__(
        self,
        samples: List[Dict],
        max_seq_len: int = 128,
    ):
        self.samples = samples
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        features = sample['features']  # [seq_len, 14]

        # Truncate if needed
        if features.shape[0] > self.max_seq_len:
            features = features[-self.max_seq_len:]
            features = features.copy()
            features[:, 0] = features[:, 0] - features[0, 0]

        seq_len = features.shape[0]

        return {
            'features': torch.FloatTensor(features),
            'seq_length': seq_len,
            'return_target': torch.FloatTensor([sample['potential_profit_pct']]),
            'drawdown_target': torch.FloatTensor([sample['drawdown_pct']]),
        }


def collate_regression(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for regression training.

    Args:
        batch: List of sample dictionaries

    Returns:
        Batched tensors with padding
    """
    max_len = max(s['seq_length'] for s in batch)
    batch_size = len(batch)
    feature_dim = batch[0]['features'].shape[-1]

    features = torch.zeros(batch_size, max_len, feature_dim)
    seq_lengths = torch.zeros(batch_size, dtype=torch.long)
    return_targets = torch.zeros(batch_size)
    drawdown_targets = torch.zeros(batch_size)

    for i, sample in enumerate(batch):
        seq_len = sample['seq_length']
        features[i, :seq_len] = sample['features']
        seq_lengths[i] = seq_len
        return_targets[i] = sample['return_target']
        drawdown_targets[i] = sample['drawdown_target']

    return {
        'features': features,
        'seq_lengths': seq_lengths,
        'return_target': return_targets,
        'drawdown_target': drawdown_targets,
    }


class BalancedReturnSampler(torch.utils.data.Sampler):
    """
    Sampler that balances samples by return distribution.

    Addresses the class imbalance in return values
    by oversampling rare return ranges.

    Args:
        samples: List of sample dictionaries
        n_bins: Number of bins for return distribution
    """

    def __init__(self, samples: List[Dict], n_bins: int = 10):
        self.samples = samples
        returns = [s['potential_profit_pct'] for s in samples]

        # Create bins
        self.bins = np.histogram_bin_edges(returns, bins=n_bins)
        self.bin_indices = [[] for _ in range(n_bins)]

        for idx, r in enumerate(returns):
            bin_idx = np.digitize(r, self.bins) - 1
            bin_idx = min(bin_idx, n_bins - 1)
            self.bin_indices[bin_idx].append(idx)

        # Calculate sampling weights
        non_empty_bins = [b for b in self.bin_indices if len(b) > 0]
        self.n_samples = len(samples)

    def __iter__(self):
        # Sample equally from each non-empty bin
        indices = []
        samples_per_bin = self.n_samples // len(self.bin_indices)

        for bin_idx in self.bin_indices:
            if len(bin_idx) > 0:
                sampled = np.random.choice(
                    bin_idx,
                    size=min(samples_per_bin, len(bin_idx) * 2),
                    replace=True,
                )
                indices.extend(sampled.tolist())

        np.random.shuffle(indices)
        return iter(indices[:self.n_samples])

    def __len__(self) -> int:
        return self.n_samples
