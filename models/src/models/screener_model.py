"""
Screener Model: XGBoost binary classifier for token screening.

Predicts whether a token is worth monitoring for potential entry.

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
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

from ..config import ScreenerConfig, DEFAULT_SCREENER_CONFIG


class ScreenerModel:
    """
    Model 1: Entry Worthiness Screener.

    XGBoost binary classifier that predicts whether a token is:
    - WORTHY (1): Worth monitoring for potential entry (>2x potential)
    - AVOID (0): Likely rug, scam, or dead-on-arrival
    """

    def __init__(self, config: Optional[ScreenerConfig] = None):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is required. Install with: pip install xgboost")
        self.config = config or DEFAULT_SCREENER_CONFIG
        self.model = None
        self.feature_importance = {}
        self.training_history = {}

    def fit(
        self, X_train: np.ndarray, y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
        verbose: int = 1,
    ) -> Dict[str, Any]:
        """Train the screener model."""
        params = self.config.to_xgb_params()
        self.model = xgb.XGBClassifier(**params)

        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=verbose > 0)

        self.feature_importance = dict(
            zip([f"feature_{i}" for i in range(X_train.shape[1])],
                self.model.feature_importances_.tolist())
        )

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

        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            self.training_history["val_metrics"] = val_metrics

        return self.training_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict_proba(X)

    def predict_worthy_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability of WORTHY class."""
        return self.predict_proba(X)[:, 1]

    def is_worthy(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Determine if tokens are worthy based on probability threshold."""
        threshold = threshold or self.config.confidence_threshold
        return self.predict_worthy_proba(X) >= threshold

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model on given data."""
        y_pred = self.predict(X)
        y_proba = self.predict_worthy_proba(X)

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.5,
        }

        cm = confusion_matrix(y, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["true_negatives"] = int(tn)
            metrics["false_positives"] = int(fp)
            metrics["false_negatives"] = int(fn)
            metrics["true_positives"] = int(tp)

        return metrics

    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Get feature importance with optional custom names."""
        if self.model is None:
            raise ValueError("Model not trained.")
        importance = self.model.feature_importances_
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        return dict(sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True))

    def save(self, path: str) -> None:
        """Save model to file."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "config": self.config, "model": self.model,
            "feature_importance": self.feature_importance,
            "training_history": self.training_history,
        }
        with open(save_path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> "ScreenerModel":
        """Load model from file."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        instance = cls(config=state["config"])
        instance.model = state["model"]
        instance.feature_importance = state["feature_importance"]
        instance.training_history = state["training_history"]
        return instance
