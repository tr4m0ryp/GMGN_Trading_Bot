"""
Entry Dataset for Model 2 (LSTM).

Provides variable-length time-series sequences for entry timing.

Author: Trading Team
Date: 2025-12-29
"""

from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from torch.utils.data import Dataset

from .loader import TokenData, load_raw_data, split_tokens_chronological
from .labels import generate_entry_samples


class EntryDataset(Dataset):
    """
    Dataset for Model 2 (Entry) - LSTM classifier.

    Provides variable-length time-series sequences.
    """

    def __init__(
        self,
        samples: Optional[List[Dict]] = None,
        tokens: Optional[List[TokenData]] = None,
        start_time: int = 30,
        sample_interval: int = 5,
    ):
        """
        Initialize entry dataset.

        Args:
            samples: Pre-computed list of sample dicts with features/labels.
            tokens: List of TokenData objects (will generate samples).
            start_time: When to start sampling (after screener).
            sample_interval: Interval between samples.
        """
        if samples is not None:
            self.samples = samples
        elif tokens is not None:
            self.samples = []
            for token in tokens:
                token_samples = generate_entry_samples(token, start_time, sample_interval)
                self.samples.extend(token_samples)
        else:
            self.samples = []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]

    def get_class_distribution(self) -> Dict[int, int]:
        """Get distribution of labels."""
        labels = [s["label"] for s in self.samples]
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))


def prepare_entry_data(
    csv_path: str,
    start_time: int = 30,
    sample_interval: int = 5,
    train_ratio: float = 0.70,
) -> Tuple["EntryDataset", "EntryDataset", "EntryDataset"]:
    """
    Prepare complete entry datasets from raw CSV.

    Args:
        csv_path: Path to raw CSV file.
        start_time: When to start sampling.
        sample_interval: Interval between samples.
        train_ratio: Training split ratio.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    tokens = load_raw_data(csv_path)
    train_tokens, val_tokens, test_tokens = split_tokens_chronological(
        tokens, train_ratio, (1 - train_ratio) / 2, (1 - train_ratio) / 2
    )

    train_ds = EntryDataset(tokens=train_tokens, start_time=start_time, sample_interval=sample_interval)
    val_ds = EntryDataset(tokens=val_tokens, start_time=start_time, sample_interval=sample_interval)
    test_ds = EntryDataset(tokens=test_tokens, start_time=start_time, sample_interval=sample_interval)

    print(f"Entry data: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test")

    return train_ds, val_ds, test_ds
