"""
Model 1: Entry Worthiness Screener - Training and evaluation functions.

Provides train_screener() and evaluate_screener() convenience functions
for the XGBoost-based token screener.

Author: Trading Team
Date: 2025-12-29
"""

from typing import Dict, Any, Tuple, Optional, List

import numpy as np
from sklearn.metrics import precision_score, recall_score

from ..config import ScreenerConfig
from .screener_model import ScreenerModel


def train_screener(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Optional[ScreenerConfig] = None,
    verbose: int = 1,
) -> Tuple[ScreenerModel, Dict[str, Any]]:
    """
    Train screener model with given data.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        config: Model configuration.
        verbose: Verbosity level.

    Returns:
        Tuple of (trained_model, training_results).
    """
    model = ScreenerModel(config)

    print("=" * 60)
    print("Training Model 1: Entry Worthiness Screener (XGBoost)")
    print("=" * 60)
    print(f"Training samples: {len(y_train)}")
    print(f"Validation samples: {len(y_val)}")
    print(f"Class distribution (train): AVOID={np.sum(y_train==0)}, WORTHY={np.sum(y_train==1)}")
    print(f"Features: {X_train.shape[1]}")
    print()

    results = model.fit(X_train, y_train, X_val, y_val, verbose)

    print()
    print("Training complete!")
    if "val_metrics" in results:
        print(f"Validation ROC-AUC: {results['val_metrics']['roc_auc']:.4f}")
        print(f"Validation Precision: {results['val_metrics']['precision']:.4f}")
        print(f"Validation Recall: {results['val_metrics']['recall']:.4f}")

    return model, results


def evaluate_screener(
    model: ScreenerModel,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of screener model.

    Args:
        model: Trained ScreenerModel.
        X_test: Test features.
        y_test: Test labels.
        feature_names: Optional feature names for importance.

    Returns:
        Dictionary with evaluation results.
    """
    print("=" * 60)
    print("Evaluating Model 1: Entry Worthiness Screener")
    print("=" * 60)

    metrics = model.evaluate(X_test, y_test)

    print(f"\nTest Set Results ({len(y_test)} samples):")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

    if "true_positives" in metrics:
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives (AVOID correct):   {metrics['true_negatives']}")
        print(f"  False Positives (AVOID as WORTHY): {metrics['false_positives']}")
        print(f"  False Negatives (WORTHY as AVOID): {metrics['false_negatives']}")
        print(f"  True Positives (WORTHY correct):   {metrics['true_positives']}")

    importance = model.get_feature_importance(feature_names)
    print(f"\nTop 10 Features:")
    for i, (name, score) in enumerate(list(importance.items())[:10]):
        print(f"  {i+1}. {name}: {score:.4f}")

    print(f"\nThreshold Analysis:")
    y_proba = model.predict_worthy_proba(X_test)
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        y_pred_thresh = (y_proba >= threshold).astype(int)
        prec = precision_score(y_test, y_pred_thresh, zero_division=0)
        rec = recall_score(y_test, y_pred_thresh, zero_division=0)
        n_passed = np.sum(y_pred_thresh)
        print(f"  Threshold {threshold:.1f}: Precision={prec:.3f}, Recall={rec:.3f}, Passed={n_passed}")

    return {"metrics": metrics, "feature_importance": importance}
