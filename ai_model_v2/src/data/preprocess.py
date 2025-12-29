"""
Data preprocessing script for Multi-Model Trading Architecture.

This script processes raw token data and generates training datasets
for all three models:
    - Model 1 (Screener): Tabular features + binary labels
    - Model 2 (Entry): Time-series features + 3-class labels
    - Model 3 (Exit): Position-aware features + 3-class labels

Usage:
    python -m data.preprocess --csv-path ../ai_data/data/combined_tokens.csv

Author: Trading Team
Date: 2025-12-29
"""

import argparse
import pickle
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import sys

import numpy as np
import pandas as pd


# Configuration constants (inline to avoid import issues)
SCREENER_DECISION_TIME = 30
SCREENER_WORTHY_THRESHOLD = 2.0
SCREENER_LOOKAHEAD_SEC = 300
ENTRY_PROFIT_THRESHOLD = 0.20
ENTRY_MAX_DRAWDOWN = 0.15
ENTRY_LOOKAHEAD_SEC = 60
EXIT_TRAILING_STOP_PCT = 0.15
EXIT_STOP_LOSS_PCT = 0.25
EXIT_PROFIT_TARGET_PCT = 2.00
DELAY_SECONDS = 1
TOTAL_FEE_PER_TX = 0.024
FIXED_POSITION_SIZE_SOL = 0.1


def process_screener_data(
    tokens: List[TokenData],
    decision_time: int = 30,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Process tokens for Model 1 (Screener) training.

    Args:
        tokens: List of TokenData objects.
        decision_time: Time at which to make screening decision.

    Returns:
        Tuple of (features, labels, metadata).
    """
    features_list = []
    labels_list = []
    metadata_list = []

    for token in tokens:
        # Extract features at decision time
        features = extract_screener_features(token, decision_time)
        if features is None:
            continue

        # Generate label
        label = generate_screener_labels(token, decision_time)
        if label is None:
            continue

        features_list.append(features)
        labels_list.append(label)
        metadata_list.append({
            "token_address": token.token_address,
            "symbol": token.symbol,
            "peak_ratio": token.peak_ratio,
            "lifespan_sec": token.lifespan_sec,
            "death_reason": token.death_reason,
        })

    features = np.array(features_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int64)

    return features, labels, metadata_list


def process_entry_data(
    tokens: List[TokenData],
    start_time: int = 30,
    sample_interval: int = 5,
    max_samples_per_token: int = 50,
) -> List[Dict]:
    """
    Process tokens for Model 2 (Entry) training.

    Args:
        tokens: List of TokenData objects.
        start_time: When to start sampling (after screener).
        sample_interval: Interval between samples.
        max_samples_per_token: Maximum samples per token.

    Returns:
        List of sample dictionaries.
    """
    all_samples = []

    for token in tokens:
        token_samples = generate_entry_samples(token, start_time, sample_interval)

        # Limit samples per token to prevent imbalance
        if len(token_samples) > max_samples_per_token:
            indices = np.linspace(0, len(token_samples) - 1, max_samples_per_token, dtype=int)
            token_samples = [token_samples[i] for i in indices]

        all_samples.extend(token_samples)

    return all_samples


def process_exit_data(
    tokens: List[TokenData],
    sample_interval: int = 5,
    max_samples_per_token: int = 30,
) -> List[Dict]:
    """
    Process tokens for Model 3 (Exit) training.

    Args:
        tokens: List of TokenData objects.
        sample_interval: Interval between samples.
        max_samples_per_token: Maximum samples per token.

    Returns:
        List of sample dictionaries.
    """
    all_samples = []

    for token in tokens:
        # Simulate entry at t=30 (after screener decision)
        if len(token.candles) < 40:
            continue

        entry_time = 30
        entry_price = token.candles[entry_time].close

        token_samples = generate_exit_samples(token, entry_time, entry_price, sample_interval)

        # Limit samples
        if len(token_samples) > max_samples_per_token:
            indices = np.linspace(0, len(token_samples) - 1, max_samples_per_token, dtype=int)
            token_samples = [token_samples[i] for i in indices]

        all_samples.extend(token_samples)

    return all_samples


def save_screener_datasets(
    train_data: Tuple,
    val_data: Tuple,
    test_data: Tuple,
    output_dir: Path,
) -> None:
    """Save screener datasets to disk."""
    # Unpack
    train_features, train_labels, train_meta = train_data
    val_features, val_labels, val_meta = val_data
    test_features, test_labels, test_meta = test_data

    # Save as pickle
    with open(output_dir / "screener_train.pkl", "wb") as f:
        pickle.dump({
            "features": train_features,
            "labels": train_labels,
            "metadata": train_meta,
        }, f)

    with open(output_dir / "screener_val.pkl", "wb") as f:
        pickle.dump({
            "features": val_features,
            "labels": val_labels,
            "metadata": val_meta,
        }, f)

    with open(output_dir / "screener_test.pkl", "wb") as f:
        pickle.dump({
            "features": test_features,
            "labels": test_labels,
            "metadata": test_meta,
        }, f)

    print(f"  Saved screener datasets:")
    print(f"    Train: {len(train_labels)} samples")
    print(f"    Val: {len(val_labels)} samples")
    print(f"    Test: {len(test_labels)} samples")


def save_entry_datasets(
    train_samples: List[Dict],
    val_samples: List[Dict],
    test_samples: List[Dict],
    output_dir: Path,
) -> None:
    """Save entry datasets to disk."""
    with open(output_dir / "entry_train.pkl", "wb") as f:
        pickle.dump(train_samples, f)

    with open(output_dir / "entry_val.pkl", "wb") as f:
        pickle.dump(val_samples, f)

    with open(output_dir / "entry_test.pkl", "wb") as f:
        pickle.dump(test_samples, f)

    print(f"  Saved entry datasets:")
    print(f"    Train: {len(train_samples)} samples")
    print(f"    Val: {len(val_samples)} samples")
    print(f"    Test: {len(test_samples)} samples")


def save_exit_datasets(
    train_samples: List[Dict],
    val_samples: List[Dict],
    test_samples: List[Dict],
    output_dir: Path,
) -> None:
    """Save exit datasets to disk."""
    with open(output_dir / "exit_train.pkl", "wb") as f:
        pickle.dump(train_samples, f)

    with open(output_dir / "exit_val.pkl", "wb") as f:
        pickle.dump(val_samples, f)

    with open(output_dir / "exit_test.pkl", "wb") as f:
        pickle.dump(test_samples, f)

    print(f"  Saved exit datasets:")
    print(f"    Train: {len(train_samples)} samples")
    print(f"    Val: {len(val_samples)} samples")
    print(f"    Test: {len(test_samples)} samples")


def compute_label_statistics(labels: np.ndarray, name: str) -> Dict:
    """Compute and print label distribution statistics."""
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)

    print(f"  {name} label distribution:")
    stats = {}
    for u, c in zip(unique, counts):
        pct = c / total * 100
        print(f"    Class {u}: {c} ({pct:.1f}%)")
        stats[int(u)] = {"count": int(c), "pct": float(pct)}

    return stats


def preprocess_all(
    csv_path: str,
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Complete preprocessing pipeline for all three models.

    Args:
        csv_path: Path to raw CSV file.
        output_dir: Output directory for processed data.
        train_ratio: Training split ratio.
        val_ratio: Validation split ratio.
        test_ratio: Test split ratio.
        random_seed: Random seed for reproducibility.

    Returns:
        Dictionary with preprocessing statistics.
    """
    start_time = time.time()
    np.random.seed(random_seed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DATA PREPROCESSING FOR MULTI-MODEL ARCHITECTURE")
    print("=" * 70)

    # Load raw data
    print("\n[1/6] Loading raw data...")
    tokens = load_raw_data(csv_path)

    # Compute statistics
    stats = compute_dataset_statistics(tokens)
    print(f"  Total tokens: {stats['num_tokens']}")
    print(f"  Median lifespan: {stats['lifespan']['median']:.1f}s")
    print(f"  Success rate (2x): {stats['success_rates']['2x']:.1%}")

    # Split tokens chronologically
    print("\n[2/6] Splitting data chronologically...")
    train_tokens, val_tokens, test_tokens = split_tokens_chronological(
        tokens, train_ratio, val_ratio, test_ratio
    )

    # Process Model 1 (Screener) data
    print("\n[3/6] Processing screener data (Model 1)...")
    train_screener = process_screener_data(train_tokens)
    val_screener = process_screener_data(val_tokens)
    test_screener = process_screener_data(test_tokens)

    save_screener_datasets(train_screener, val_screener, test_screener, output_path)

    screener_stats = compute_label_statistics(train_screener[1], "Screener (train)")

    # Process Model 2 (Entry) data
    print("\n[4/6] Processing entry data (Model 2)...")
    train_entry = process_entry_data(train_tokens)
    val_entry = process_entry_data(val_tokens)
    test_entry = process_entry_data(test_tokens)

    save_entry_datasets(train_entry, val_entry, test_entry, output_path)

    entry_labels = np.array([s["label"] for s in train_entry])
    entry_stats = compute_label_statistics(entry_labels, "Entry (train)")

    # Process Model 3 (Exit) data
    print("\n[5/6] Processing exit data (Model 3)...")
    train_exit = process_exit_data(train_tokens)
    val_exit = process_exit_data(val_tokens)
    test_exit = process_exit_data(test_tokens)

    save_exit_datasets(train_exit, val_exit, test_exit, output_path)

    exit_labels = np.array([s["label"] for s in train_exit])
    exit_stats = compute_label_statistics(exit_labels, "Exit (train)")

    # Save metadata
    print("\n[6/6] Saving metadata...")
    metadata = {
        "csv_path": str(csv_path),
        "random_seed": random_seed,
        "splits": {
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
        },
        "token_counts": {
            "total": len(tokens),
            "train": len(train_tokens),
            "val": len(val_tokens),
            "test": len(test_tokens),
        },
        "dataset_stats": stats,
        "screener": {
            "train_samples": len(train_screener[1]),
            "val_samples": len(val_screener[1]),
            "test_samples": len(test_screener[1]),
            "label_distribution": screener_stats,
        },
        "entry": {
            "train_samples": len(train_entry),
            "val_samples": len(val_entry),
            "test_samples": len(test_entry),
            "label_distribution": entry_stats,
        },
        "exit": {
            "train_samples": len(train_exit),
            "val_samples": len(val_exit),
            "test_samples": len(test_exit),
            "label_distribution": exit_stats,
        },
        "processing_time_seconds": time.time() - start_time,
    }

    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    with open(output_path / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    # Print summary
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"Output directory: {output_path}")
    print(f"Processing time: {metadata['processing_time_seconds']:.1f}s")
    print()
    print("Generated files:")
    print("  - screener_train.pkl, screener_val.pkl, screener_test.pkl")
    print("  - entry_train.pkl, entry_val.pkl, entry_test.pkl")
    print("  - exit_train.pkl, exit_val.pkl, exit_test.pkl")
    print("  - metadata.json, metadata.pkl")
    print()
    print("Sample counts:")
    print(f"  Screener: {metadata['screener']['train_samples']} train, "
          f"{metadata['screener']['val_samples']} val, {metadata['screener']['test_samples']} test")
    print(f"  Entry: {metadata['entry']['train_samples']} train, "
          f"{metadata['entry']['val_samples']} val, {metadata['entry']['test_samples']} test")
    print(f"  Exit: {metadata['exit']['train_samples']} train, "
          f"{metadata['exit']['val_samples']} val, {metadata['exit']['test_samples']} test")
    print("=" * 70)

    return metadata


def main():
    """Main entry point for preprocessing script."""
    parser = argparse.ArgumentParser(
        description="Preprocess data for Multi-Model Trading Architecture"
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="../../ai_data/data/combined_tokens.csv",
        help="Path to raw CSV file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../data/processed",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Training split ratio (default: 0.70)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio (default: 0.15)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test split ratio (default: 0.15)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    # Validate splits
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        print("Error: train_ratio + val_ratio + test_ratio must equal 1.0")
        sys.exit(1)

    preprocess_all(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
