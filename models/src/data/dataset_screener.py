"""
Screener Dataset for Model 1 (XGBoost).

Provides tabular features for each token at the decision time.

Author: Trading Team
Date: 2025-12-29
"""

from typing import List, Dict, Tuple, Optional

import numpy as np
from torch.utils.data import Dataset

from .loader import TokenData, load_raw_data, split_tokens_chronological
from .labels import generate_screener_dataset


class ScreenerDataset(Dataset):
    """
    Dataset for Model 1 (Screener) - XGBoost classifier.

    Provides tabular features for each token at the decision time.
    """

    def __init__(
        self,
        tokens: Optional[List[TokenData]] = None,
        features: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        metadata: Optional[List[Dict]] = None,
        decision_time: int = 30,
    ):
        """
        Initialize screener dataset.

        Can be initialized with either:
        - tokens: List of TokenData objects (will extract features)
        - features/labels: Pre-computed arrays

        Args:
            tokens: List of TokenData objects.
            features: Pre-computed feature matrix (n_samples, n_features).
            labels: Pre-computed label array (n_samples,).
            metadata: List of metadata dictionaries.
            decision_time: Time at which to make decision.
        """
        if tokens is not None:
            self.features_list, self.labels_list, self.metadata = generate_screener_dataset(
                tokens, decision_time
            )
            self.features = np.array(self.features_list, dtype=np.float32)
            self.labels = np.array(self.labels_list, dtype=np.int64)
        else:
            self.features = features
            self.labels = labels
            self.metadata = metadata

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        return self.features[idx], self.labels[idx]

    def get_class_weights(self) -> np.ndarray:
        """Compute class weights for imbalanced data."""
        unique, counts = np.unique(self.labels, return_counts=True)
        total = len(self.labels)
        weights = total / (len(unique) * counts)
        return weights

    def get_xgb_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get data in format suitable for XGBoost."""
        return self.features, self.labels


def prepare_screener_data(
    csv_path: str,
    decision_time: int = 30,
    train_ratio: float = 0.70,
) -> Tuple["ScreenerDataset", "ScreenerDataset", "ScreenerDataset"]:
    """
    Prepare complete screener datasets from raw CSV.

    Args:
        csv_path: Path to raw CSV file.
        decision_time: Time for screener decision.
        train_ratio: Training split ratio.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    tokens = load_raw_data(csv_path)
    train_tokens, val_tokens, test_tokens = split_tokens_chronological(
        tokens, train_ratio, (1 - train_ratio) / 2, (1 - train_ratio) / 2
    )

    train_ds = ScreenerDataset(tokens=train_tokens, decision_time=decision_time)
    val_ds = ScreenerDataset(tokens=val_tokens, decision_time=decision_time)
    test_ds = ScreenerDataset(tokens=test_tokens, decision_time=decision_time)

    print(f"Screener data: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test")

    return train_ds, val_ds, test_ds
