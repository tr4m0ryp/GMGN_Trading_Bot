"""
Exit Model: XGBoost classifier with hard-coded risk management.

Contains the ExitModel class that combines ML predictions
with risk rules to determine optimal exit points.

Author: Trading Team
Date: 2025-12-29
"""

from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import pickle

import numpy as np

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

from ..config import ExitConfig, DEFAULT_EXIT_CONFIG
from ..data.labels import ExitLabel
from .exit import ExitReason, Position, RiskManager


class ExitModel:
    """
    Model 3: Exit Point Optimizer.

    Combines XGBoost classifier with hard-coded risk rules.
    Risk rules always override ML predictions.
    """

    def __init__(self, config: Optional[ExitConfig] = None):
        """
        Initialize exit model.

        Args:
            config: ExitConfig with hyperparameters.
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost required. Install with: pip install xgboost")

        self.config = config or DEFAULT_EXIT_CONFIG
        self.ml_model = None
        self.risk_manager = RiskManager(self.config)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: int = 1,
    ) -> Dict[str, Any]:
        """
        Train the ML component of exit model.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.
            verbose: Verbosity level.

        Returns:
            Training results dictionary.
        """
        params = {
            "n_estimators": self.config.n_estimators,
            "max_depth": self.config.max_depth,
            "learning_rate": self.config.learning_rate,
            "objective": "multi:softprob",
            "num_class": self.config.num_classes,
            "random_state": 42,
            "tree_method": "hist",
        }

        self.ml_model = xgb.XGBClassifier(**params)

        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self.ml_model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=verbose > 0,
        )

        results = {
            "n_train": len(y_train),
            "n_val": len(y_val) if y_val is not None else 0,
        }

        return results

    def predict_ml(self, X: np.ndarray) -> np.ndarray:
        """Get ML model predictions."""
        if self.ml_model is None:
            raise ValueError("Model not trained")
        return self.ml_model.predict(X)

    def predict_ml_proba(self, X: np.ndarray) -> np.ndarray:
        """Get ML model probability predictions."""
        if self.ml_model is None:
            raise ValueError("Model not trained")
        return self.ml_model.predict_proba(X)

    def get_exit_signal(
        self,
        features: np.ndarray,
        position: Position,
        current_price: float,
        current_time: int,
    ) -> Tuple[int, ExitReason, float]:
        """
        Get combined exit signal from ML + rules.

        Risk rules take priority over ML predictions.

        Args:
            features: Exit features for ML model.
            position: Current position.
            current_price: Current price.
            current_time: Current candle index.

        Returns:
            Tuple of (exit_label, reason, exit_fraction).
        """
        # Update position high
        position.update(current_price)

        # 1. Check hard rules first (always takes priority)
        should_exit, reason, fraction = self.risk_manager.check_exit_rules(
            position, current_price, current_time
        )

        if should_exit:
            label = ExitLabel.EXIT_NOW if fraction == 1.0 else ExitLabel.PARTIAL_EXIT
            return label, reason, fraction

        # 2. Get ML prediction
        if self.ml_model is not None:
            features = features.reshape(1, -1) if features.ndim == 1 else features
            ml_proba = self.predict_ml_proba(features)[0]

            exit_proba = ml_proba[ExitLabel.EXIT_NOW]
            if exit_proba >= self.config.confidence_threshold:
                return ExitLabel.EXIT_NOW, ExitReason.ML_SIGNAL, 1.0

            partial_proba = ml_proba[ExitLabel.PARTIAL_EXIT]
            if partial_proba >= self.config.confidence_threshold:
                return ExitLabel.PARTIAL_EXIT, ExitReason.ML_SIGNAL, 0.5

        # 3. Default: hold
        return ExitLabel.HOLD, ExitReason.NONE, 0.0

    def save(self, path: str) -> None:
        """Save model to file."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "config": self.config,
            "ml_model": self.ml_model,
        }

        with open(save_path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> "ExitModel":
        """Load model from file."""
        with open(path, "rb") as f:
            state = pickle.load(f)

        instance = cls(config=state["config"])
        instance.ml_model = state["ml_model"]
        return instance
