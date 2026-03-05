"""
Exit Dataset for Model 3 (Position-aware classifier).

Generates samples from simulated trades for exit timing.

Author: Trading Team
Date: 2025-12-29
"""

from typing import List, Dict, Any, Tuple, Optional

from torch.utils.data import Dataset

from .loader import TokenData, load_raw_data, split_tokens_chronological
from .labels import generate_exit_samples


class ExitDataset(Dataset):
    """
    Dataset for Model 3 (Exit) - Position-aware classifier.

    Generates samples from simulated trades.
    """

    def __init__(
        self,
        samples: Optional[List[Dict]] = None,
        tokens: Optional[List[TokenData]] = None,
        sample_interval: int = 5,
    ):
        """
        Initialize exit dataset.

        For each token, simulates an entry at the optimal point
        and generates exit samples from there.

        Args:
            samples: Pre-computed list of sample dicts.
            tokens: List of TokenData objects (will simulate trades).
            sample_interval: Interval between samples.
        """
        if samples is not None:
            self.samples = samples
        elif tokens is not None:
            self.samples = []
            for token in tokens:
                # Simulate entry at t=30 (after screener decision)
                if len(token.candles) < 40:
                    continue

                entry_time = 30
                entry_price = token.candles[entry_time].close

                token_samples = generate_exit_samples(
                    token, entry_time, entry_price, sample_interval
                )
                self.samples.extend(token_samples)
        else:
            self.samples = []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def prepare_exit_data(
    csv_path: str,
    sample_interval: int = 5,
    train_ratio: float = 0.70,
) -> Tuple["ExitDataset", "ExitDataset", "ExitDataset"]:
    """
    Prepare complete exit datasets from raw CSV.

    Args:
        csv_path: Path to raw CSV file.
        sample_interval: Interval between samples.
        train_ratio: Training split ratio.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    tokens = load_raw_data(csv_path)
    train_tokens, val_tokens, test_tokens = split_tokens_chronological(
        tokens, train_ratio, (1 - train_ratio) / 2, (1 - train_ratio) / 2
    )

    train_ds = ExitDataset(tokens=train_tokens, sample_interval=sample_interval)
    val_ds = ExitDataset(tokens=val_tokens, sample_interval=sample_interval)
    test_ds = ExitDataset(tokens=test_tokens, sample_interval=sample_interval)

    print(f"Exit data: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test")

    return train_ds, val_ds, test_ds
