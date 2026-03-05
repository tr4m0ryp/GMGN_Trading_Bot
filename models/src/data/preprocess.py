"""
Data preprocessing pipeline for Multi-Model Trading Architecture.

Orchestrates the full preprocessing workflow: loads raw token data,
generates features and labels for all three models (screener, entry, exit),
and saves train/val/test splits as pickle files.

Usage:
    python -m models.src.data.preprocess --csv-path ../../data/raw/combined_tokens.csv

Dependencies: numpy, pandas, pickle, json
Date: 2025-12-29
"""

import argparse
import pickle
import json
import time
from pathlib import Path
from typing import List

import numpy as np

from .data_loading import TokenData, load_raw_data, split_tokens_chronological
from .feature_extraction import (
    extract_screener_features,
    extract_timeseries_features,
    SCREENER_FEATURE_NAMES,
)
from .label_generation import (
    ScreenerLabel,
    EntryLabel,
    ExitLabel,
    generate_screener_label,
    generate_entry_label,
    generate_exit_label,
    get_execution_price,
    ENTRY_LOOKAHEAD_SEC,
    ENTRY_SAMPLES_PER_TOKEN,
    DELAY_SECONDS,
)


# =============================================================================
# Dataset Generation
# =============================================================================

def process_screener_data(tokens: List[TokenData], decision_time: int = 30):
    """Process tokens for Model 1 (Screener).

    Returns (features_array, labels_array, metadata_list).
    """
    features_list = []
    labels_list = []
    metadata_list = []

    for token in tokens:
        features = extract_screener_features(token, decision_time)
        if features is None:
            continue

        label = generate_screener_label(token, decision_time)
        if label is None:
            continue

        features_list.append(features)
        labels_list.append(label)
        metadata_list.append({
            "token_address": token.token_address,
            "symbol": token.symbol,
            "peak_ratio": token.peak_ratio,
        })

    return np.array(features_list), np.array(labels_list), metadata_list


def process_entry_data(tokens: List[TokenData], start_time: int = 30, sample_interval: int = 5):
    """Process tokens for Model 2 (Entry) with stratified sampling."""
    all_samples = []

    for token in tokens:
        candles = token.candles
        token_samples = []

        for t in range(start_time, len(candles) - ENTRY_LOOKAHEAD_SEC, sample_interval):
            current_candles = candles[:t + 1]
            features = extract_timeseries_features(current_candles)

            if len(features) < 10:
                continue

            label = generate_entry_label(candles, t)
            if label is None:
                continue

            token_samples.append({
                "features": features,
                "label": label,
                "timestamp": t,
                "token_address": token.token_address,
            })

            # Limit samples per token to reduce correlation
            if len(token_samples) >= ENTRY_SAMPLES_PER_TOKEN:
                break

        all_samples.extend(token_samples)

    return all_samples


def process_exit_data(tokens: List[TokenData], sample_interval: int = 5):
    """Process tokens for Model 3 (Exit)."""
    all_samples = []

    for token in tokens:
        candles = token.candles
        if len(candles) < 40:
            continue

        entry_time = 30
        entry_price = candles[entry_time].close

        token_samples = []
        for t in range(entry_time + 1, len(candles), sample_interval):
            # Extract exit features (base + position-aware)
            current_candles = candles[:t + 1]
            base_features = extract_timeseries_features(current_candles)

            if len(base_features) == 0:
                continue

            current_price = candles[t].close
            position_candles = candles[entry_time:t + 1]
            position_high = max(c.high for c in position_candles) if position_candles else entry_price

            unrealized_pnl = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0
            time_since_entry = (t - entry_time) / 300.0
            drawdown_from_position_high = (
                (current_price - position_high) / position_high if position_high > 0 else 0.0
            )

            exit_features = np.concatenate([
                base_features[-1],
                np.array([unrealized_pnl, time_since_entry, drawdown_from_position_high],
                         dtype=np.float32),
            ])

            label = generate_exit_label(candles, entry_time, entry_price, t)

            token_samples.append({
                "features": exit_features,
                "label": label,
                "timestamp": t,
                "token_address": token.token_address,
            })

            if label == ExitLabel.EXIT_NOW:
                break

            if len(token_samples) >= 30:
                break

        all_samples.extend(token_samples)

    return all_samples


# =============================================================================
# Conversion Helpers
# =============================================================================

def _convert_entry_to_arrays(samples):
    """Convert entry sample dicts to arrays for pickling."""
    if not samples:
        return {"sequences": [], "labels": np.array([], dtype=np.int64),
                "timestamps": [], "token_addresses": []}
    return {
        "sequences": [s["features"] for s in samples],
        "labels": np.array([int(s["label"]) for s in samples], dtype=np.int64),
        "timestamps": [s["timestamp"] for s in samples],
        "token_addresses": [s["token_address"] for s in samples],
    }


def _convert_exit_to_arrays(samples):
    """Convert exit sample dicts to arrays for pickling."""
    if not samples:
        return {"X": np.array([]), "y": np.array([], dtype=np.int64),
                "timestamps": [], "token_addresses": []}
    return {
        "X": np.array([s["features"] for s in samples], dtype=np.float32),
        "y": np.array([int(s["label"]) for s in samples], dtype=np.int64),
        "timestamps": [s["timestamp"] for s in samples],
        "token_addresses": [s["token_address"] for s in samples],
    }


# =============================================================================
# Main Preprocessing Pipeline
# =============================================================================

def preprocess_all(
    csv_path: str,
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
):
    """Complete preprocessing pipeline.

    Loads CSV, splits chronologically, generates features/labels for
    all three models, and saves pickle files to output_dir.
    """
    start_time_ts = time.time()
    np.random.seed(random_seed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DATA PREPROCESSING FOR MULTI-MODEL ARCHITECTURE")
    print("=" * 70)

    # Load data
    print("\n[1/6] Loading raw data...")
    tokens = load_raw_data(csv_path)
    print(f"  Loaded {len(tokens)} tokens")

    # Split
    print("\n[2/6] Splitting data chronologically...")
    train_tokens, val_tokens, test_tokens = split_tokens_chronological(
        tokens, train_ratio, val_ratio, test_ratio
    )
    print(f"  Train: {len(train_tokens)}, Val: {len(val_tokens)}, Test: {len(test_tokens)}")

    # Screener data
    print("\n[3/6] Processing screener data (Model 1)...")
    screener_splits = [process_screener_data(t) for t in (train_tokens, val_tokens, test_tokens)]
    for (feat, lab, meta), name in zip(screener_splits, ("train", "val", "test")):
        with open(output_path / f"screener_{name}.pkl", "wb") as f:
            pickle.dump({"X": feat, "y": lab, "feature_names": SCREENER_FEATURE_NAMES, "metadata": meta}, f)

    train_s_lab = screener_splits[0][1]
    print(f"  Train: {len(screener_splits[0][1])}, Val: {len(screener_splits[1][1])}, Test: {len(screener_splits[2][1])}")
    if len(train_s_lab) > 0:
        worthy_pct = np.sum(train_s_lab == 1) / len(train_s_lab) * 100
        print(f"  Class dist (train): AVOID={100-worthy_pct:.1f}%, WORTHY={worthy_pct:.1f}%")

    # Entry data
    print("\n[4/6] Processing entry data (Model 2)...")
    entry_splits = [process_entry_data(t) for t in (train_tokens, val_tokens, test_tokens)]
    for samples, name in zip(entry_splits, ("train", "val", "test")):
        with open(output_path / f"entry_{name}.pkl", "wb") as f:
            pickle.dump(_convert_entry_to_arrays(samples), f)
    print(f"  Train: {len(entry_splits[0])}, Val: {len(entry_splits[1])}, Test: {len(entry_splits[2])}")

    # Exit data
    print("\n[5/6] Processing exit data (Model 3)...")
    exit_splits = [process_exit_data(t) for t in (train_tokens, val_tokens, test_tokens)]
    for samples, name in zip(exit_splits, ("train", "val", "test")):
        with open(output_path / f"exit_{name}.pkl", "wb") as f:
            pickle.dump(_convert_exit_to_arrays(samples), f)
    print(f"  Train: {len(exit_splits[0])}, Val: {len(exit_splits[1])}, Test: {len(exit_splits[2])}")

    # Metadata
    print("\n[6/6] Saving metadata...")
    processing_time = time.time() - start_time_ts
    split_names = ("train", "val", "test")
    metadata = {
        "csv_path": str(csv_path), "random_seed": random_seed,
        "token_counts": dict(zip(("total",) + split_names, (len(tokens), len(train_tokens), len(val_tokens), len(test_tokens)))),
        "screener": dict(zip(split_names, (len(s[1]) for s in screener_splits))),
        "entry": dict(zip(split_names, (len(s) for s in entry_splits))),
        "exit": dict(zip(split_names, (len(s) for s in exit_splits))),
        "processing_time_seconds": processing_time,
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'=' * 70}\nPREPROCESSING COMPLETE\n{'=' * 70}")
    print(f"Output: {output_path}\nTime: {processing_time:.1f}s\n{'=' * 70}")

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=str, default="../../data/raw/combined_tokens.csv")
    parser.add_argument("--output-dir", type=str, default="../data/processed")
    args = parser.parse_args()

    preprocess_all(args.csv_path, args.output_dir)
