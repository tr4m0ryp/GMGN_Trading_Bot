"""
Model 1: Entry Worthiness Screener (XGBoost).

Binary classification model to determine if a token is worth monitoring
for potential entry. Uses static and early dynamic features.

Architecture: XGBoost classifier with GPU acceleration.

Author: Trading Team
Date: 2025-12-29
"""

import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from ..config import ScreenerConfig, DEFAULT_SCREENER_CONFIG


class ScreenerModel:
    """
    Model 1: Entry Worthiness Screener.

    XGBoost binary classifier that predicts whether a token is:
    - WORTHY (1): Worth monitoring for potential entry (>2x potential)
    - AVOID (0): Likely rug, scam, or dead-on-arrival

    Attributes:
        config: ScreenerConfig with hyperparameters.
        model: Trained XGBoost classifier.
        feature_importance: Dictionary of feature importances.
    """

    def __init__(self, config: Optional[ScreenerConfig] = None):
        """
        Initialize screener model.

        Args:
            config: ScreenerConfig with hyperparameters. Uses default if None.
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost is required for ScreenerModel. "
                "Install with: pip install xgboost"
            )

        self.config = config or DEFAULT_SCREENER_CONFIG
        self.model = None
        self.feature_importance = {}
        self.training_history = {}

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: int = 1,
    ) -> Dict[str, Any]:
        """
        Train the screener model.

        Args:
            X_train: Training features (n_samples, n_features).
            y_train: Training labels (n_samples,).
            X_val: Validation features (optional).
            y_val: Validation labels (optional).
            verbose: Verbosity level (0=silent, 1=progress).

        Returns:
            Dictionary with training results.
        """
        # Get XGBoost parameters
        params = self.config.to_xgb_params()

        # Create classifier
        self.model = xgb.XGBClassifier(**params)

        # Setup evaluation sets
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        # Train with early stopping
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=verbose > 0,
        )

        # Get feature importance
        self.feature_importance = dict(
            zip(
                [f"feature_{i}" for i in range(X_train.shape[1])],
                self.model.feature_importances_.tolist(),
            )
        )

        # Store training history
        self.training_history = {
            "best_iteration": self.model.best_iteration if hasattr(self.model, 'best_iteration') else self.config.n_estimators,
            "n_features": X_train.shape[1],
            "n_train_samples": len(y_train),
            "n_val_samples": len(y_val) if y_val is not None else 0,
            "class_distribution": {
                "train_avoid": int(np.sum(y_train == 0)),
                "train_worthy": int(np.sum(y_train == 1)),
            },
        }

        # Evaluate on validation set
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            self.training_history["val_metrics"] = val_metrics

        return self.training_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Features (n_samples, n_features).

        Returns:
            Predicted labels (n_samples,).
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features (n_samples, n_features).

        Returns:
            Predicted probabilities (n_samples, 2).
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict_proba(X)

    def predict_worthy_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability of WORTHY class.

        Args:
            X: Features (n_samples, n_features).

        Returns:
            Probability of WORTHY (n_samples,).
        """
        proba = self.predict_proba(X)
        return proba[:, 1]  # WORTHY is class 1

    def is_worthy(
        self, X: np.ndarray, threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Determine if tokens are worthy based on probability threshold.

        Args:
            X: Features (n_samples, n_features).
            threshold: Probability threshold. Uses config default if None.

        Returns:
            Boolean array (n_samples,).
        """
        threshold = threshold or self.config.confidence_threshold
        proba = self.predict_worthy_proba(X)
        return proba >= threshold

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on given data.

        Args:
            X: Features (n_samples, n_features).
            y: True labels (n_samples,).

        Returns:
            Dictionary with evaluation metrics.
        """
        y_pred = self.predict(X)
        y_proba = self.predict_worthy_proba(X)

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.5,
        }

        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["true_negatives"] = int(tn)
            metrics["false_positives"] = int(fp)
            metrics["false_negatives"] = int(fn)
            metrics["true_positives"] = int(tp)

        return metrics

    def get_feature_importance(
        self, feature_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Get feature importance with optional custom names.

        Args:
            feature_names: List of feature names.

        Returns:
            Dictionary mapping feature names to importance scores.
        """
        if self.model is None:
            raise ValueError("Model not trained.")

        importance = self.model.feature_importances_
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]

        return dict(sorted(
            zip(feature_names, importance),
            key=lambda x: x[1],
            reverse=True
        ))

    def save(self, path: str) -> None:
        """
        Save model to file.

        Args:
            path: Path to save model.
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "config": self.config,
            "model": self.model,
            "feature_importance": self.feature_importance,
            "training_history": self.training_history,
        }

        with open(save_path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> "ScreenerModel":
        """
        Load model from file.

        Args:
            path: Path to saved model.

        Returns:
            Loaded ScreenerModel instance.
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        instance = cls(config=state["config"])
        instance.model = state["model"]
        instance.feature_importance = state["feature_importance"]
        instance.training_history = state["training_history"]

        return instance


# =============================================================================
# Training Functions
# =============================================================================

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

    # Basic metrics
    metrics = model.evaluate(X_test, y_test)

    print(f"\nTest Set Results ({len(y_test)} samples):")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

    # Confusion matrix
    if "true_positives" in metrics:
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives (AVOID correct):   {metrics['true_negatives']}")
        print(f"  False Positives (AVOID as WORTHY): {metrics['false_positives']}")
        print(f"  False Negatives (WORTHY as AVOID): {metrics['false_negatives']}")
        print(f"  True Positives (WORTHY correct):   {metrics['true_positives']}")

    # Feature importance
    importance = model.get_feature_importance(feature_names)
    print(f"\nTop 10 Features:")
    for i, (name, score) in enumerate(list(importance.items())[:10]):
        print(f"  {i+1}. {name}: {score:.4f}")

    # Threshold analysis
    print(f"\nThreshold Analysis:")
    y_proba = model.predict_worthy_proba(X_test)
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        y_pred_thresh = (y_proba >= threshold).astype(int)
        prec = precision_score(y_test, y_pred_thresh, zero_division=0)
        rec = recall_score(y_test, y_pred_thresh, zero_division=0)
        n_passed = np.sum(y_pred_thresh)
        print(f"  Threshold {threshold:.1f}: Precision={prec:.3f}, Recall={rec:.3f}, Passed={n_passed}")

    results = {
        "metrics": metrics,
        "feature_importance": importance,
    }

    return results
