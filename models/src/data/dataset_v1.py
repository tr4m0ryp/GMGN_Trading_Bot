"""
PyTorch Dataset and collate functions for v1 trading model.

This module provides the TradingDataset class for variable-length trading
sequences and collate functions for batching with padding/truncation.

Dependencies:
    torch: Tensor operations

Author: Trading Team
Date: 2025-12-21
"""

from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


# Maximum sequence length for truncation (prevents OOM and LSTM padding issues)
MAX_SEQ_LEN: int = 128

# Feature indices that need re-normalization after truncation
LOG_CLOSE_INDEX: int = 0  # log_close is relative to first candle


class TradingDataset(Dataset):
    """
    PyTorch Dataset for variable-length trading sequences.

    Loads preprocessed training samples and provides them in a format
    suitable for PyTorch DataLoader with variable-length sequences.

    Attributes:
        samples: List of training samples from prepare_realistic_training_data.

    Args:
        samples: List of sample dictionaries.

    Example:
        >>> dataset = TradingDataset(all_samples)
        >>> print(f"Dataset size: {len(dataset)}")
        >>> sample = dataset[0]
        >>> print(f"Features shape: {sample['features'].shape}")
    """

    def __init__(self, samples: List[Dict[str, Any]]):
        """Initialize dataset with preprocessed samples."""
        self.samples = samples

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Sample index.

        Returns:
            Dictionary containing features, label, and metadata.
        """
        return self.samples[idx]


def collate_variable_length(
    batch: List[Dict[str, Any]], max_seq_len: int = MAX_SEQ_LEN
) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader with variable-length sequences.

    Pads sequences to the same length within a batch for efficient processing.
    Truncates long sequences to max_seq_len, keeping the most recent candles.
    Re-normalizes log_close feature after truncation to maintain relative scaling.

    Args:
        batch: List of samples from TradingDataset.
        max_seq_len: Maximum sequence length. Longer sequences are truncated.

    Returns:
        Dictionary with padded tensors:
            - features: (batch, max_seq_len, 14)
            - labels: (batch,)
            - seq_lengths: (batch,)

    Example:
        >>> loader = DataLoader(dataset, batch_size=32, collate_fn=collate_variable_length)
        >>> for batch in loader:
        ...     print(batch['features'].shape)
    """
    features = []
    actual_lengths = []

    for item in batch:
        feat = torch.FloatTensor(item['features'])
        seq_len = feat.size(0)

        # Truncate to max_seq_len, keeping the most recent (end) candles
        if seq_len > max_seq_len:
            feat = feat[-max_seq_len:]

            # CRITICAL: Re-normalize log_close feature
            # log_close is relative to the original first candle, but that candle
            # is no longer in our truncated sequence. Shift so first value is 0.
            log_close_offset = feat[0, LOG_CLOSE_INDEX].clone()
            feat[:, LOG_CLOSE_INDEX] = feat[:, LOG_CLOSE_INDEX] - log_close_offset

            seq_len = max_seq_len

        features.append(feat)
        actual_lengths.append(seq_len)

    labels = torch.LongTensor([item['label'] for item in batch])
    seq_lengths = torch.LongTensor(actual_lengths)

    padded_features = pad_sequence(features, batch_first=True, padding_value=0.0)

    return {
        'features': padded_features,
        'labels': labels,
        'seq_lengths': seq_lengths,
    }


def collate_for_regression(
    batch: List[Dict[str, Any]], max_seq_len: int = MAX_SEQ_LEN
) -> Dict[str, torch.Tensor]:
    """
    Collate function for regression training with return/drawdown targets.

    Similar to collate_variable_length but includes continuous regression
    targets instead of classification labels.

    Args:
        batch: List of samples from TradingDataset.
        max_seq_len: Maximum sequence length.

    Returns:
        Dictionary with padded tensors:
            - features: (batch, max_seq_len, 14)
            - seq_lengths: (batch,)
            - return_target: (batch,) - potential_profit_pct
            - drawdown_target: (batch,) - drawdown_pct

    Example:
        >>> loader = DataLoader(dataset, batch_size=32, collate_fn=collate_for_regression)
    """
    features = []
    actual_lengths = []

    for item in batch:
        feat = torch.FloatTensor(item['features'])
        seq_len = feat.size(0)

        # Truncate to max_seq_len, keeping the most recent candles
        if seq_len > max_seq_len:
            feat = feat[-max_seq_len:]
            # Re-normalize log_close feature
            log_close_offset = feat[0, LOG_CLOSE_INDEX].clone()
            feat[:, LOG_CLOSE_INDEX] = feat[:, LOG_CLOSE_INDEX] - log_close_offset
            seq_len = max_seq_len

        features.append(feat)
        actual_lengths.append(seq_len)

    seq_lengths = torch.LongTensor(actual_lengths)

    # Regression targets (continuous values)
    return_targets = torch.FloatTensor([item['potential_profit_pct'] for item in batch])
    drawdown_targets = torch.FloatTensor([item['drawdown_pct'] for item in batch])

    padded_features = pad_sequence(features, batch_first=True, padding_value=0.0)

    return {
        'features': padded_features,
        'seq_lengths': seq_lengths,
        'return_target': return_targets,
        'drawdown_target': drawdown_targets,
    }
