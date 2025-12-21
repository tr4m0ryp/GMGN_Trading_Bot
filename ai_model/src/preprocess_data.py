"""
Data preprocessing script for trading model.

This script loads raw token data from CSV, generates training samples with
full historical context, and saves processed data to disk for faster training.

Dependencies:
    numpy: Numerical computations
    pickle: Data serialization
    pathlib: Path handling

Author: Trading Team
Date: 2025-12-21
"""

import pickle
import argparse
from pathlib import Path
from typing import Dict, Any, List
import sys

import numpy as np

from data_preparation import (
    load_raw_data,
    parse_candles,
    prepare_realistic_training_data,
)
from utils import set_seed


def process_and_save_data(csv_path: str,
                         output_dir: str,
                         train_split: float = 0.8,
                         val_split: float = 0.1,
                         test_split: float = 0.1,
                         random_seed: int = 42) -> None:
    """
    Process raw data and save to disk.

    Loads token data from CSV, generates training samples for each token,
    splits into train/val/test sets, and saves processed data.

    Args:
        csv_path: Path to raw CSV file.
        output_dir: Directory to save processed data.
        train_split: Fraction of tokens for training. Default is 0.8.
        val_split: Fraction of tokens for validation. Default is 0.1.
        test_split: Fraction of tokens for testing. Default is 0.1.
        random_seed: Random seed for reproducibility. Default is 42.

    Example:
        >>> process_and_save_data('data/raw/rawdata.csv', 'data/processed')
    """
    set_seed(random_seed)
    np.random.seed(random_seed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading raw data...")
    df = load_raw_data(csv_path)
    print(f"Loaded {len(df)} tokens")

    indices = np.arange(len(df))
    np.random.shuffle(indices)

    train_end = int(len(df) * train_split)
    val_end = train_end + int(len(df) * val_split)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    print(f"\nSplit: {len(train_indices)} train, {len(val_indices)} val, "
          f"{len(test_indices)} test tokens")

    print("\nProcessing training tokens...")
    train_samples = []
    train_token_info = []
    for i, idx in enumerate(train_indices):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(train_indices)} tokens")

        row = df.iloc[idx]
        try:
            candles = parse_candles(row['candles'])
            samples = prepare_realistic_training_data(candles)

            if samples:
                train_samples.extend(samples)
                train_token_info.append({
                    'token_address': row['token_address'],
                    'symbol': row.get('symbol', 'UNKNOWN'),
                    'num_samples': len(samples),
                    'num_candles': len(candles),
                })
        except Exception as e:
            print(f"  Warning: Failed to process token {idx}: {e}")
            continue

    print(f"\nProcessing validation tokens...")
    val_samples = []
    val_token_info = []
    for i, idx in enumerate(val_indices):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(val_indices)} tokens")

        row = df.iloc[idx]
        try:
            candles = parse_candles(row['candles'])
            samples = prepare_realistic_training_data(candles)

            if samples:
                val_samples.extend(samples)
                val_token_info.append({
                    'token_address': row['token_address'],
                    'symbol': row.get('symbol', 'UNKNOWN'),
                    'num_samples': len(samples),
                    'num_candles': len(candles),
                })
        except Exception as e:
            print(f"  Warning: Failed to process token {idx}: {e}")
            continue

    print(f"\nProcessing test tokens...")
    test_samples = []
    test_token_info = []
    for i, idx in enumerate(test_indices):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(test_indices)} tokens")

        row = df.iloc[idx]
        try:
            candles = parse_candles(row['candles'])
            samples = prepare_realistic_training_data(candles)

            if samples:
                test_samples.extend(samples)
                test_token_info.append({
                    'token_address': row['token_address'],
                    'symbol': row.get('symbol', 'UNKNOWN'),
                    'num_samples': len(samples),
                    'num_candles': len(candles),
                })
        except Exception as e:
            print(f"  Warning: Failed to process token {idx}: {e}")
            continue

    print(f"\n{'='*60}")
    print("Processing Summary:")
    print(f"{'='*60}")
    print(f"Train samples: {len(train_samples):,} from {len(train_token_info)} tokens")
    print(f"Val samples:   {len(val_samples):,} from {len(val_token_info)} tokens")
    print(f"Test samples:  {len(test_samples):,} from {len(test_token_info)} tokens")
    print(f"Total samples: {len(train_samples) + len(val_samples) + len(test_samples):,}")

    print(f"\nSaving processed data to {output_dir}...")

    train_path = output_path / 'train_samples.pkl'
    with open(train_path, 'wb') as f:
        pickle.dump(train_samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved {train_path}")

    val_path = output_path / 'val_samples.pkl'
    with open(val_path, 'wb') as f:
        pickle.dump(val_samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved {val_path}")

    test_path = output_path / 'test_samples.pkl'
    with open(test_path, 'wb') as f:
        pickle.dump(test_samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved {test_path}")

    metadata = {
        'train_tokens': len(train_token_info),
        'val_tokens': len(val_token_info),
        'test_tokens': len(test_token_info),
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'test_samples': len(test_samples),
        'train_token_info': train_token_info,
        'val_token_info': val_token_info,
        'test_token_info': test_token_info,
        'random_seed': random_seed,
        'train_split': train_split,
        'val_split': val_split,
        'test_split': test_split,
    }

    metadata_path = output_path / 'metadata.pkl'
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved {metadata_path}")

    print(f"\n{'='*60}")
    print("Data Statistics:")
    print(f"{'='*60}")

    if train_samples:
        seq_lengths = [s['seq_length'] for s in train_samples]
        print(f"\nSequence lengths (train):")
        print(f"  Min:    {min(seq_lengths)}")
        print(f"  Max:    {max(seq_lengths)}")
        print(f"  Mean:   {np.mean(seq_lengths):.1f}")
        print(f"  Median: {np.median(seq_lengths):.1f}")

        labels = [s['label'] for s in train_samples]
        label_counts = np.bincount(labels)
        print(f"\nLabel distribution (train):")
        print(f"  HOLD (0): {label_counts[0]:,} ({label_counts[0]/len(labels)*100:.1f}%)")
        print(f"  BUY  (1): {label_counts[1]:,} ({label_counts[1]/len(labels)*100:.1f}%)")
        print(f"  SELL (2): {label_counts[2]:,} ({label_counts[2]/len(labels)*100:.1f}%)")

        profits = [s['potential_profit_pct'] for s in train_samples if s['label'] == 1]
        if profits:
            print(f"\nPotential profit for BUY signals (train):")
            print(f"  Min:  {min(profits)*100:.2f}%")
            print(f"  Max:  {max(profits)*100:.2f}%")
            print(f"  Mean: {np.mean(profits)*100:.2f}%")

    print(f"\n{'='*60}")
    print("Processing completed successfully!")
    print(f"{'='*60}\n")


def main():
    """Main entry point for preprocessing script."""
    parser = argparse.ArgumentParser(
        description='Preprocess raw token data for training'
    )
    parser.add_argument(
        '--csv-path',
        type=str,
        default='../data/raw/rawdata.csv',
        help='Path to raw CSV file (default: ../data/raw/rawdata.csv)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../data/processed',
        help='Output directory for processed data (default: ../data/processed)'
    )
    parser.add_argument(
        '--train-split',
        type=float,
        default=0.8,
        help='Training split ratio (default: 0.8)'
    )
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.1,
        help='Validation split ratio (default: 0.1)'
    )
    parser.add_argument(
        '--test-split',
        type=float,
        default=0.1,
        help='Test split ratio (default: 0.1)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    args = parser.parse_args()

    if abs(args.train_split + args.val_split + args.test_split - 1.0) > 1e-6:
        print("Error: train_split + val_split + test_split must equal 1.0")
        sys.exit(1)

    process_and_save_data(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        random_seed=args.seed
    )


if __name__ == '__main__':
    main()
