"""
Training script for Model 1: Entry Worthiness Screener.

Trains XGBoost classifier on tabular features to predict
if a token is worth monitoring for potential entry.

Author: Trading Team
Date: 2025-12-29
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import time
import json

import numpy as np

from ..config import ScreenerConfig, DEFAULT_SCREENER_CONFIG, SCREENER_STATIC_FEATURES, SCREENER_DYNAMIC_FEATURES
from ..models.screener import ScreenerModel, train_screener, evaluate_screener
from ..data.dataset import ScreenerDataset, prepare_screener_data


def train_screener_model(
    data_path: str,
    output_dir: str,
    config: Optional[ScreenerConfig] = None,
    decision_time: int = 30,
    verbose: int = 1,
) -> Tuple[ScreenerModel, Dict[str, Any]]:
    """
    Full training pipeline for Model 1 (Screener).

    Args:
        data_path: Path to raw CSV data.
        output_dir: Directory to save model and results.
        config: Model configuration.
        decision_time: Time at which to make screening decision.
        verbose: Verbosity level.

    Returns:
        Tuple of (trained_model, results_dict).
    """
    start_time = time.time()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MODEL 1: ENTRY WORTHINESS SCREENER (XGBoost)")
    print("=" * 70)

    # Prepare data
    print("\n[1/4] Preparing data...")
    train_ds, val_ds, test_ds = prepare_screener_data(
        data_path, decision_time=decision_time
    )

    X_train, y_train = train_ds.get_xgb_data()
    X_val, y_val = val_ds.get_xgb_data()
    X_test, y_test = test_ds.get_xgb_data()

    print(f"  Train: {len(y_train)} samples")
    print(f"  Val:   {len(y_val)} samples")
    print(f"  Test:  {len(y_test)} samples")

    # Class distribution
    train_worthy_pct = np.sum(y_train == 1) / len(y_train) * 100
    print(f"  Train class dist: {100-train_worthy_pct:.1f}% AVOID, {train_worthy_pct:.1f}% WORTHY")

    # Train model
    print("\n[2/4] Training model...")
    model, train_results = train_screener(
        X_train, y_train, X_val, y_val,
        config=config, verbose=verbose
    )

    # Evaluate
    print("\n[3/4] Evaluating on test set...")
    feature_names = SCREENER_STATIC_FEATURES + SCREENER_DYNAMIC_FEATURES
    eval_results = evaluate_screener(model, X_test, y_test, feature_names)

    # Save model
    print("\n[4/4] Saving model...")
    model_path = output_path / "screener_model.pkl"
    model.save(model_path)
    print(f"  Model saved to: {model_path}")

    # Save results
    results = {
        "train_results": train_results,
        "eval_results": {
            "metrics": eval_results["metrics"],
            "feature_importance": {k: float(v) for k, v in list(eval_results["feature_importance"].items())[:20]},
        },
        "data_stats": {
            "n_train": len(y_train),
            "n_val": len(y_val),
            "n_test": len(y_test),
            "train_worthy_pct": float(train_worthy_pct),
        },
        "total_time_seconds": time.time() - start_time,
    }

    results_path = output_path / "screener_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"  Results saved to: {results_path}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total time: {results['total_time_seconds']:.1f}s")
    print(f"Test ROC-AUC: {eval_results['metrics']['roc_auc']:.4f}")
    print(f"Test F1: {eval_results['metrics']['f1']:.4f}")
    print("=" * 70)

    return model, results
