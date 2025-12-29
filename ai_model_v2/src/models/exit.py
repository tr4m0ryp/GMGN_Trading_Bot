"""
Model 3: Exit Point Optimizer with Risk Management.

Ensemble model combining ML predictions with hard-coded risk rules
to determine optimal exit points for open positions.

Author: Trading Team
Date: 2025-12-29
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Tuple, Optional, List
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


class ExitReason(Enum):
    """Reason for exit signal."""
    NONE = "none"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    TIME_STOP = "time_stop"
    PROFIT_TARGET = "profit_target"
    ML_SIGNAL = "ml_signal"
    PARTIAL_PROFIT = "partial_profit"


@dataclass
class Position:
    """Represents an open trading position."""

    entry_price: float
    entry_time: int  # Candle index
    position_size: float = 1.0
    position_high: float = 0.0  # Highest price since entry
    partial_exit_taken: bool = False

    def update(self, current_price: float) -> None:
        """Update position high."""
        if current_price > self.position_high:
            self.position_high = current_price

    @property
    def has_position_high(self) -> bool:
        """Check if position high is set."""
        return self.position_high > 0


class RiskManager:
    """
    Hard-coded risk management rules that override ML predictions.

    These rules enforce strict risk limits:
    - Stop loss: Exit if unrealized loss exceeds threshold
    - Trailing stop: Exit if price drops from position high
    - Time stop: Exit after maximum time in position
    - Profit target: Exit at profit target
    """

    def __init__(self, config: Optional[ExitConfig] = None):
        """
        Initialize risk manager.

        Args:
            config: ExitConfig with risk parameters.
        """
        self.config = config or DEFAULT_EXIT_CONFIG

    def check_exit_rules(
        self,
        position: Position,
        current_price: float,
        current_time: int,
    ) -> Tuple[bool, ExitReason, float]:
        """
        Check all hard-coded exit rules.

        Args:
            position: Current position.
            current_price: Current market price.
            current_time: Current candle index.

        Returns:
            Tuple of (should_exit, reason, exit_fraction).
            exit_fraction is 1.0 for full exit, 0.5 for partial.
        """
        entry_price = position.entry_price
        position_high = position.position_high if position.has_position_high else entry_price

        # Calculate metrics
        unrealized_pnl = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0
        drawdown_from_high = (current_price - position_high) / position_high if position_high > 0 else 0.0
        time_in_position = current_time - position.entry_time

        # 1. Stop Loss
        if unrealized_pnl <= -self.config.stop_loss_pct:
            return True, ExitReason.STOP_LOSS, 1.0

        # 2. Trailing Stop
        if drawdown_from_high <= -self.config.trailing_stop_pct:
            return True, ExitReason.TRAILING_STOP, 1.0

        # 3. Profit Target
        if unrealized_pnl >= self.config.profit_target_pct:
            return True, ExitReason.PROFIT_TARGET, 1.0

        # 4. Time Stop (unless exceptional gains)
        if time_in_position >= self.config.time_stop_sec:
            if unrealized_pnl < self.config.time_exception_gain_pct:
                return True, ExitReason.TIME_STOP, 1.0

        # 5. Partial Exit (take some profit)
        if (
            unrealized_pnl >= self.config.partial_exit_threshold_pct
            and not position.partial_exit_taken
        ):
            return True, ExitReason.PARTIAL_PROFIT, self.config.partial_exit_fraction

        return False, ExitReason.NONE, 0.0


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

            # Check if ML suggests exit with high confidence
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


class TradingPipeline:
    """
    Complete trading pipeline combining all three models.

    Flow:
    1. Model 1 (Screener) filters tokens
    2. Model 2 (Entry) times entry
    3. Model 3 (Exit) manages position
    """

    def __init__(
        self,
        screener_model=None,
        entry_model=None,
        exit_model=None,
    ):
        """
        Initialize trading pipeline.

        Args:
            screener_model: Trained ScreenerModel.
            entry_model: Trained EntryModel.
            exit_model: Trained ExitModel.
        """
        self.screener = screener_model
        self.entry = entry_model
        self.exit = exit_model

    def screen_token(
        self,
        features: np.ndarray,
        threshold: float = 0.5,
    ) -> Tuple[bool, float]:
        """
        Screen token for worthiness.

        Args:
            features: Screener features.
            threshold: Confidence threshold.

        Returns:
            Tuple of (is_worthy, confidence).
        """
        if self.screener is None:
            return True, 1.0  # Pass through if no screener

        proba = self.screener.predict_worthy_proba(features.reshape(1, -1))[0]
        return proba >= threshold, proba

    def check_entry(
        self,
        features: np.ndarray,
        seq_lengths: Optional[np.ndarray] = None,
        threshold: float = 0.7,
    ) -> Tuple[int, float]:
        """
        Check for entry signal.

        Args:
            features: Time-series features.
            seq_lengths: Sequence lengths.
            threshold: Confidence threshold.

        Returns:
            Tuple of (label, confidence).
            label: 0=WAIT, 1=ENTER_NOW, 2=ABORT
        """
        if self.entry is None:
            return 1, 1.0  # Default to enter if no model

        import torch
        features_t = torch.FloatTensor(features).unsqueeze(0)
        if seq_lengths is not None:
            seq_lengths_t = torch.LongTensor([seq_lengths])
        else:
            seq_lengths_t = None

        proba = self.entry.predict_proba(features_t, seq_lengths_t)[0]

        label = proba.argmax().item()
        confidence = proba[label].item()

        return label, confidence

    def check_exit(
        self,
        features: np.ndarray,
        position: Position,
        current_price: float,
        current_time: int,
    ) -> Tuple[int, ExitReason, float]:
        """
        Check for exit signal.

        Args:
            features: Exit features.
            position: Current position.
            current_price: Current price.
            current_time: Current candle index.

        Returns:
            Tuple of (label, reason, exit_fraction).
        """
        if self.exit is None:
            # Use only risk rules
            rm = RiskManager()
            should_exit, reason, fraction = rm.check_exit_rules(
                position, current_price, current_time
            )
            if should_exit:
                label = ExitLabel.EXIT_NOW if fraction == 1.0 else ExitLabel.PARTIAL_EXIT
                return label, reason, fraction
            return ExitLabel.HOLD, ExitReason.NONE, 0.0

        return self.exit.get_exit_signal(
            features, position, current_price, current_time
        )

    def save(self, directory: str) -> None:
        """Save all models to directory."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        if self.screener is not None:
            self.screener.save(dir_path / "screener.pkl")

        if self.entry is not None:
            from .entry import save_entry_model
            save_entry_model(self.entry, dir_path / "entry.pt")

        if self.exit is not None:
            self.exit.save(dir_path / "exit.pkl")

    @classmethod
    def load(cls, directory: str, device: str = "cpu") -> "TradingPipeline":
        """Load all models from directory."""
        dir_path = Path(directory)

        screener = None
        entry = None
        exit_model = None

        if (dir_path / "screener.pkl").exists():
            from .screener import ScreenerModel
            screener = ScreenerModel.load(dir_path / "screener.pkl")

        if (dir_path / "entry.pt").exists():
            from .entry import load_entry_model
            entry, _ = load_entry_model(dir_path / "entry.pt", device)

        if (dir_path / "exit.pkl").exists():
            exit_model = ExitModel.load(dir_path / "exit.pkl")

        return cls(screener, entry, exit_model)
