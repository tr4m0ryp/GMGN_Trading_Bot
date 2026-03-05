"""
Training script for Model 3: Exit Point Optimizer.

Trains XGBoost model for exit timing predictions.
Works in conjunction with hard-coded risk rules.

Author: Trading Team
Date: 2025-12-29
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import time
import json

import numpy as np

from ..config import ExitConfig, DEFAULT_EXIT_CONFIG
from ..models.exit_model import ExitModel
from ..data.dataset import ExitDataset, prepare_exit_data


def train_exit_model(
    data_path: str,
    output_dir: str,
    config: Optional[ExitConfig] = None,
    verbose: int = 1,
) -> Tuple[ExitModel, Dict[str, Any]]:
    """
    Full training pipeline for Model 3 (Exit).

    Args:
        data_path: Path to raw CSV data.
        output_dir: Directory to save model and results.
        config: Model configuration.
        verbose: Verbosity level.

    Returns:
        Tuple of (trained_model, results_dict).
    """
    start_time = time.time()
    config = config or DEFAULT_EXIT_CONFIG
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MODEL 3: EXIT POINT OPTIMIZER (XGBoost + Rules)")
    print("=" * 70)

    # Prepare data
    print("\n[1/4] Preparing data...")
    train_ds, val_ds, test_ds = prepare_exit_data(data_path)

    # Convert to arrays
    print("  Converting to arrays...")
    X_train = np.array([s["features"] for s in train_ds.samples])
    y_train = np.array([s["label"] for s in train_ds.samples])
    X_val = np.array([s["features"] for s in val_ds.samples])
    y_val = np.array([s["label"] for s in val_ds.samples])
    X_test = np.array([s["features"] for s in test_ds.samples])
    y_test = np.array([s["label"] for s in test_ds.samples])

    print(f"  Train: {len(y_train)} samples")
    print(f"  Val:   {len(y_val)} samples")
    print(f"  Test:  {len(y_test)} samples")

    # Class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    class_dist = dict(zip(unique.tolist(), counts.tolist()))
    print(f"  Class dist: {class_dist}")

    # Train model
    print("\n[2/4] Training ML component...")
    model = ExitModel(config)
    train_results = model.fit(X_train, y_train, X_val, y_val, verbose)

    # Evaluate
    print("\n[3/4] Evaluating...")
    y_pred = model.predict_ml(X_test)
    accuracy = np.mean(y_pred == y_test)

    # Per-class metrics
    class_acc = {}
    for c in range(3):
        mask = y_test == c
        if mask.sum() > 0:
            class_acc[c] = np.mean(y_pred[mask] == c)

    print(f"  Test accuracy: {accuracy:.4f}")
    print(f"  Class accuracies: {class_acc}")

    # Save model
    print("\n[4/4] Saving model...")
    model_path = output_path / "exit_model.pkl"
    model.save(model_path)
    print(f"  Model saved to: {model_path}")

    # Results
    results = {
        "train_results": train_results,
        "test_metrics": {
            "accuracy": float(accuracy),
            "class_accuracy": {str(k): float(v) for k, v in class_acc.items()},
        },
        "data_stats": {
            "n_train": len(y_train),
            "n_val": len(y_val),
            "n_test": len(y_test),
            "class_distribution": {str(k): int(v) for k, v in class_dist.items()},
        },
        "risk_rules": {
            "stop_loss_pct": config.stop_loss_pct,
            "trailing_stop_pct": config.trailing_stop_pct,
            "profit_target_pct": config.profit_target_pct,
            "time_stop_sec": config.time_stop_sec,
        },
        "total_time_seconds": time.time() - start_time,
    }

    with open(output_path / "exit_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total time: {results['total_time_seconds']:.1f}s")
    print(f"Test accuracy: {accuracy:.4f}")
    print("\nNote: Risk rules will override ML predictions at inference time.")
    print("=" * 70)

    return model, results
