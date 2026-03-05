"""
Data Loader Utilities: Splitting and statistics for token data.

Contains utilities for chronological data splitting and
computing dataset statistics.

Dependencies:
    numpy

Author: Trading Team
Date: 2025-12-29
"""

from typing import List, Dict, Any

import numpy as np


def split_tokens_chronological(
    tokens,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple:
    """
    Split tokens chronologically for time-series cross-validation.

    Uses token discovery time to ensure train/val/test are in temporal order.
    This prevents look-ahead bias.

    Args:
        tokens: List of TokenData objects.
        train_ratio: Fraction for training (default 0.70).
        val_ratio: Fraction for validation (default 0.15).
        test_ratio: Fraction for testing (default 0.15).

    Returns:
        Tuple of (train_tokens, val_tokens, test_tokens).
    """
    sorted_tokens = sorted(tokens, key=lambda t: t.discovered_at_unix)

    n = len(sorted_tokens)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_tokens = sorted_tokens[:train_end]
    val_tokens = sorted_tokens[train_end:val_end]
    test_tokens = sorted_tokens[val_end:]

    print(f"Split: {len(train_tokens)} train, {len(val_tokens)} val, {len(test_tokens)} test")

    return train_tokens, val_tokens, test_tokens


def compute_dataset_statistics(tokens) -> Dict[str, Any]:
    """
    Compute statistics for a list of tokens.

    Args:
        tokens: List of TokenData objects.

    Returns:
        Dictionary with dataset statistics.
    """
    lifespans = [t.lifespan_sec for t in tokens]
    peak_times = [t.peak_time_sec for t in tokens]
    peak_ratios = [t.peak_ratio for t in tokens]
    entry_mcs = [t.entry_mc for t in tokens]

    stats = {
        "num_tokens": len(tokens),
        "lifespan": {
            "mean": np.mean(lifespans),
            "median": np.median(lifespans),
            "min": np.min(lifespans),
            "max": np.max(lifespans),
        },
        "peak_time": {
            "mean": np.mean(peak_times),
            "median": np.median(peak_times),
            "percentile_90": np.percentile(peak_times, 90),
        },
        "peak_ratio": {
            "mean": np.mean(peak_ratios),
            "median": np.median(peak_ratios),
            "percentile_75": np.percentile(peak_ratios, 75),
            "percentile_90": np.percentile(peak_ratios, 90),
            "max": np.max(peak_ratios),
        },
        "entry_mc": {
            "mean": np.mean(entry_mcs),
            "median": np.median(entry_mcs),
        },
        "success_rates": {
            "2x": sum(1 for t in tokens if t.peak_ratio >= 2.0) / len(tokens),
            "4x": sum(1 for t in tokens if t.peak_ratio >= 4.0) / len(tokens),
            "10x": sum(1 for t in tokens if t.peak_ratio >= 10.0) / len(tokens),
        },
        "death_reasons": {},
    }

    for t in tokens:
        reason = t.death_reason
        if reason not in stats["death_reasons"]:
            stats["death_reasons"][reason] = 0
        stats["death_reasons"][reason] += 1

    return stats
