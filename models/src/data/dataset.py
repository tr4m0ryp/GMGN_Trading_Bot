"""
PyTorch Dataset classes for Multi-Model Trading Architecture.

This module re-exports dataset classes and provides shared utilities:
    - Collate functions for batching
    - Data loader factory
    - Serialization helpers

Author: Trading Team
Date: 2025-12-29
"""

import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Re-export dataset classes and prepare functions
from .dataset_screener import ScreenerDataset, prepare_screener_data
from .dataset_entry import EntryDataset, prepare_entry_data
from .dataset_exit import ExitDataset, prepare_exit_data


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
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
        )

    return train_loader, val_loader, test_loader


# =============================================================================
# Serialization Helpers
# =============================================================================

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
