"""
Dataset preparation and loading pipeline for v1 trading model.

This module provides functions to prepare train/val/test datasets from
raw CSV data, and to load preprocessed datasets from disk.

Dependencies:
    numpy: Numerical computations
    pickle: Data serialization

Author: Trading Team
Date: 2025-12-21
"""

import pickle
from typing import Dict, Tuple, Any
from pathlib import Path

import numpy as np

from .preparation import (
    load_raw_data,
    parse_candles,
    prepare_realistic_training_data,
)
from .dataset_v1 import TradingDataset


def prepare_datasets(csv_path: str,
                    train_split: float = 0.8,
                    val_split: float = 0.1,
                    test_split: float = 0.1,
                    random_seed: int = 42) -> Tuple[TradingDataset,
                                                     TradingDataset,
                                                     TradingDataset]:
    """
    Load raw data and create train/val/test datasets.

    Loads token data from CSV, generates training samples for each token,
    and splits into train/validation/test sets at the token level.

    Args:
        csv_path: Path to raw CSV file.
        train_split: Fraction of tokens for training. Default is 0.8.
        val_split: Fraction of tokens for validation. Default is 0.1.
        test_split: Fraction of tokens for testing. Default is 0.1.
        random_seed: Random seed for reproducibility. Default is 42.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).

    Example:
        >>> train_ds, val_ds, test_ds = prepare_datasets('data/raw/rawdata.csv')
        >>> print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    """
    np.random.seed(random_seed)

    df = load_raw_data(csv_path)

    indices = np.arange(len(df))
    np.random.shuffle(indices)

    train_end = int(len(df) * train_split)
    val_end = train_end + int(len(df) * val_split)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    train_samples = []
    val_samples = []
    test_samples = []

    for idx in train_indices:
        candles = parse_candles(df.iloc[idx]['candles'])
        samples = prepare_realistic_training_data(candles)
        train_samples.extend(samples)

    for idx in val_indices:
        candles = parse_candles(df.iloc[idx]['candles'])
        samples = prepare_realistic_training_data(candles)
        val_samples.extend(samples)

    for idx in test_indices:
        candles = parse_candles(df.iloc[idx]['candles'])
        samples = prepare_realistic_training_data(candles)
        test_samples.extend(samples)

    train_dataset = TradingDataset(train_samples)
    val_dataset = TradingDataset(val_samples)
    test_dataset = TradingDataset(test_samples)

    return train_dataset, val_dataset, test_dataset


def load_preprocessed_datasets(processed_dir: str) -> Tuple[TradingDataset,
                                                             TradingDataset,
                                                             TradingDataset,
                                                             Dict[str, Any]]:
    """
    Load preprocessed datasets from disk.

    Loads train/val/test datasets that were previously processed and saved
    by the data/preprocess.py script. This is much faster than processing
    from raw CSV each time.

    Args:
        processed_dir: Directory containing preprocessed pickle files.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, metadata).

    Raises:
        FileNotFoundError: If required pickle files are not found.

    Example:
        >>> train_ds, val_ds, test_ds, meta = load_preprocessed_datasets('../data/processed')
        >>> print(f"Loaded {len(train_ds)} train samples")
        >>> print(f"Random seed used: {meta['random_seed']}")
    """
    processed_path = Path(processed_dir)

    if not processed_path.exists():
        raise FileNotFoundError(f"Processed data directory not found: {processed_dir}")

    train_path = processed_path / 'train_samples.pkl'
    val_path = processed_path / 'val_samples.pkl'
    test_path = processed_path / 'test_samples.pkl'
    metadata_path = processed_path / 'metadata.pkl'

    for path in [train_path, val_path, test_path, metadata_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    print(f"Loading preprocessed data from {processed_dir}...")

    with open(train_path, 'rb') as f:
        train_samples = pickle.load(f)
    print(f"  Loaded {len(train_samples):,} train samples")

    with open(val_path, 'rb') as f:
        val_samples = pickle.load(f)
    print(f"  Loaded {len(val_samples):,} validation samples")

    with open(test_path, 'rb') as f:
        test_samples = pickle.load(f)
    print(f"  Loaded {len(test_samples):,} test samples")

    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    print(f"  Loaded metadata")

    train_dataset = TradingDataset(train_samples)
    val_dataset = TradingDataset(val_samples)
    test_dataset = TradingDataset(test_samples)

    return train_dataset, val_dataset, test_dataset, metadata
