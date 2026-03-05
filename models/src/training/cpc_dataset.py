"""
CPC Dataset: Data loading utilities for Phase 1 pretraining.

Provides dataset class and collate function for CPC
self-supervised pretraining on sequences.

Dependencies:
    torch

Date: 2025-12-25
"""

from typing import Dict, List

import torch
from torch.utils.data import Dataset


class CPCDataset(Dataset):
    """
    Dataset for CPC pretraining.

    Only uses features, ignores labels (self-supervised).

    Args:
        samples: List of sample dictionaries with 'features' key
        max_seq_len: Maximum sequence length. Default 128.
        min_seq_len: Minimum sequence length. Default 20.
    """

    def __init__(
        self,
        samples: List[Dict],
        max_seq_len: int = 128,
        min_seq_len: int = 20,
    ):
        # Filter samples by sequence length
        self.samples = [
            s for s in samples
            if s['features'].shape[0] >= min_seq_len
        ]
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len

        print(f"CPCDataset: {len(self.samples)} samples "
              f"(filtered from {len(samples)}, min_len={min_seq_len})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        features = sample['features']  # [seq_len, 14]

        # Truncate if needed (keep last max_seq_len)
        if features.shape[0] > self.max_seq_len:
            features = features[-self.max_seq_len:]
            # Re-normalize log_close (index 0) after truncation
            features = features.copy()
            features[:, 0] = features[:, 0] - features[0, 0]

        seq_len = features.shape[0]

        return {
            'features': torch.FloatTensor(features),
            'seq_length': seq_len,
        }


def collate_cpc(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for CPC training.

    Pads sequences to batch maximum length.

    Args:
        batch: List of sample dictionaries

    Returns:
        Batched tensors with padding
    """
    # Get max sequence length in batch
    max_len = max(s['seq_length'] for s in batch)
    batch_size = len(batch)
    feature_dim = batch[0]['features'].shape[-1]

    # Pad features
    features = torch.zeros(batch_size, max_len, feature_dim)
    seq_lengths = torch.zeros(batch_size, dtype=torch.long)

    for i, sample in enumerate(batch):
        seq_len = sample['seq_length']
        features[i, :seq_len] = sample['features']
        seq_lengths[i] = seq_len

    return {
        'features': features,
        'seq_lengths': seq_lengths,
    }
