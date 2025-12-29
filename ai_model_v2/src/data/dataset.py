"""
PyTorch Dataset classes for Multi-Model Trading Architecture.

This module implements dataset classes and data loaders for:
    - Model 1 (Screener): Tabular data for XGBoost
    - Model 2 (Entry): Variable-length sequences for LSTM
    - Model 3 (Exit): Position-aware sequences

Author: Trading Team
Date: 2025-12-29
"""

import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from .loader import TokenData, load_raw_data, split_tokens_chronological
from .features import FeatureExtractor, extract_screener_features, extract_timeseries_features
from .labels import (
    LabelGenerator,
    generate_screener_dataset,
    generate_entry_samples,
    generate_exit_samples,
)


# =============================================================================
# Model 1: Screener Dataset
# =============================================================================

class ScreenerDataset(Dataset):
    """
    Dataset for Model 1 (Screener) - XGBoost classifier.

    Provides tabular features for each token at the decision time.
    """

    def __init__(
        self,
        tokens: Optional[List[TokenData]] = None,
        features: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        metadata: Optional[List[Dict]] = None,
        decision_time: int = 30,
    ):
        """
        Initialize screener dataset.

        Can be initialized with either:
        - tokens: List of TokenData objects (will extract features)
        - features/labels: Pre-computed arrays

        Args:
            tokens: List of TokenData objects.
            features: Pre-computed feature matrix (n_samples, n_features).
            labels: Pre-computed label array (n_samples,).
            metadata: List of metadata dictionaries.
            decision_time: Time at which to make decision.
        """
        if tokens is not None:
            self.features_list, self.labels_list, self.metadata = generate_screener_dataset(
                tokens, decision_time
            )
            self.features = np.array(self.features_list, dtype=np.float32)
            self.labels = np.array(self.labels_list, dtype=np.int64)
        else:
            self.features = features
            self.labels = labels
            self.metadata = metadata

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        return self.features[idx], self.labels[idx]

    def get_class_weights(self) -> np.ndarray:
        """Compute class weights for imbalanced data."""
        unique, counts = np.unique(self.labels, return_counts=True)
        total = len(self.labels)
        weights = total / (len(unique) * counts)
        return weights

    def get_xgb_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get data in format suitable for XGBoost."""
        return self.features, self.labels


# =============================================================================
# Model 2: Entry Dataset
# =============================================================================

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


# =============================================================================
# Model 3: Exit Dataset
# =============================================================================

class ExitDataset(Dataset):
    """
    Dataset for Model 3 (Exit) - Position-aware classifier.

    Generates samples from simulated trades.
    """

    def __init__(
        self,
        samples: Optional[List[Dict]] = None,
        tokens: Optional[List[TokenData]] = None,
        sample_interval: int = 5,
    ):
        """
        Initialize exit dataset.

        For each token, simulates an entry at the optimal point
        and generates exit samples from there.

        Args:
            samples: Pre-computed list of sample dicts.
            tokens: List of TokenData objects (will simulate trades).
            sample_interval: Interval between samples.
        """
        if samples is not None:
            self.samples = samples
        elif tokens is not None:
            self.samples = []
            for token in tokens:
                # Simulate entry at t=30 (after screener decision)
                if len(token.candles) < 40:
                    continue

                entry_time = 30
                entry_price = token.candles[entry_time].close

                token_samples = generate_exit_samples(
                    token, entry_time, entry_price, sample_interval
                )
                self.samples.extend(token_samples)
        else:
            self.samples = []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


# =============================================================================
# Collate Functions
# =============================================================================

def collate_entry_batch(
    batch: List[Dict], max_seq_len: int = 120
) -> Dict[str, torch.Tensor]:
    """
    Collate function for Entry model batches.

    Pads variable-length sequences and handles truncation.

    Args:
        batch: List of sample dictionaries.
        max_seq_len: Maximum sequence length.

    Returns:
        Dictionary with padded tensors.
    """
    features = []
    actual_lengths = []

    for item in batch:
        feat = torch.FloatTensor(item["features"])
        seq_len = feat.size(0)

        # Truncate if needed (keep most recent)
        if seq_len > max_seq_len:
            feat = feat[-max_seq_len:]
            # Re-normalize log_close (index 0)
            feat[:, 0] = feat[:, 0] - feat[0, 0]
            seq_len = max_seq_len

        features.append(feat)
        actual_lengths.append(seq_len)

    labels = torch.LongTensor([item["label"] for item in batch])
    seq_lengths = torch.LongTensor(actual_lengths)

    # Pad sequences
    padded_features = pad_sequence(features, batch_first=True, padding_value=0.0)

    return {
        "features": padded_features,
        "labels": labels,
        "seq_lengths": seq_lengths,
    }


def collate_exit_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for Exit model batches.

    Exit features are fixed-length (14 base + 6 position = 20).

    Args:
        batch: List of sample dictionaries.

    Returns:
        Dictionary with feature and label tensors.
    """
    features = torch.stack([torch.FloatTensor(item["features"]) for item in batch])
    labels = torch.LongTensor([item["label"] for item in batch])

    return {
        "features": features,
        "labels": labels,
    }


# =============================================================================
# Data Loader Factory
# =============================================================================

def create_data_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = 128,
    collate_fn=None,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create train/val/test data loaders.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        test_dataset: Test dataset (optional).
        batch_size: Batch size.
        collate_fn: Custom collate function.
        num_workers: Number of worker processes.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader, test_loader


# =============================================================================
# Full Pipeline Functions
# =============================================================================

def prepare_screener_data(
    csv_path: str,
    decision_time: int = 30,
    train_ratio: float = 0.70,
) -> Tuple[ScreenerDataset, ScreenerDataset, ScreenerDataset]:
    """
    Prepare complete screener datasets from raw CSV.

    Args:
        csv_path: Path to raw CSV file.
        decision_time: Time for screener decision.
        train_ratio: Training split ratio.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    tokens = load_raw_data(csv_path)
    train_tokens, val_tokens, test_tokens = split_tokens_chronological(
        tokens, train_ratio, (1 - train_ratio) / 2, (1 - train_ratio) / 2
    )

    train_ds = ScreenerDataset(tokens=train_tokens, decision_time=decision_time)
    val_ds = ScreenerDataset(tokens=val_tokens, decision_time=decision_time)
    test_ds = ScreenerDataset(tokens=test_tokens, decision_time=decision_time)

    print(f"Screener data: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test")

    return train_ds, val_ds, test_ds


def prepare_entry_data(
    csv_path: str,
    start_time: int = 30,
    sample_interval: int = 5,
    train_ratio: float = 0.70,
) -> Tuple[EntryDataset, EntryDataset, EntryDataset]:
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


def prepare_exit_data(
    csv_path: str,
    sample_interval: int = 5,
    train_ratio: float = 0.70,
) -> Tuple[ExitDataset, ExitDataset, ExitDataset]:
    """
    Prepare complete exit datasets from raw CSV.

    Args:
        csv_path: Path to raw CSV file.
        sample_interval: Interval between samples.
        train_ratio: Training split ratio.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    tokens = load_raw_data(csv_path)
    train_tokens, val_tokens, test_tokens = split_tokens_chronological(
        tokens, train_ratio, (1 - train_ratio) / 2, (1 - train_ratio) / 2
    )

    train_ds = ExitDataset(tokens=train_tokens, sample_interval=sample_interval)
    val_ds = ExitDataset(tokens=val_tokens, sample_interval=sample_interval)
    test_ds = ExitDataset(tokens=test_tokens, sample_interval=sample_interval)

    print(f"Exit data: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test")

    return train_ds, val_ds, test_ds


def save_datasets(
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    output_dir: str,
    prefix: str = "data",
) -> None:
    """
    Save datasets to pickle files.

    Args:
        train_ds: Training dataset.
        val_ds: Validation dataset.
        test_ds: Test dataset.
        output_dir: Output directory.
        prefix: Filename prefix.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / f"{prefix}_train.pkl", "wb") as f:
        pickle.dump(train_ds, f)

    with open(output_path / f"{prefix}_val.pkl", "wb") as f:
        pickle.dump(val_ds, f)

    with open(output_path / f"{prefix}_test.pkl", "wb") as f:
        pickle.dump(test_ds, f)

    print(f"Saved datasets to {output_dir}")


def load_datasets(
    data_dir: str, prefix: str = "data"
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load datasets from pickle files.

    Args:
        data_dir: Directory containing pickle files.
        prefix: Filename prefix.

    Returns:
        Tuple of (train_ds, val_ds, test_ds).
    """
    data_path = Path(data_dir)

    with open(data_path / f"{prefix}_train.pkl", "rb") as f:
        train_ds = pickle.load(f)

    with open(data_path / f"{prefix}_val.pkl", "rb") as f:
        val_ds = pickle.load(f)

    with open(data_path / f"{prefix}_test.pkl", "rb") as f:
        test_ds = pickle.load(f)

    return train_ds, val_ds, test_ds
