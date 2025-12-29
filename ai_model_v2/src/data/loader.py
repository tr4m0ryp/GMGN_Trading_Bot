"""
Data loading and parsing for Multi-Model Trading Architecture.

This module handles loading raw token data from CSV files and parsing
the JSON candle data into structured format.

Dependencies:
    pandas: DataFrame operations
    numpy: Numerical computations

Author: Trading Team
Date: 2025-12-29
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd


@dataclass
class Candle:
    """Single OHLCV candle with timestamp."""

    time: int  # Unix timestamp in milliseconds
    open: float
    high: float
    low: float
    close: float
    volume: float

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Candle":
        """Create Candle from dictionary (handles both long and short keys)."""
        return cls(
            time=int(d.get("time", d.get("t", 0))),
            open=float(d.get("open", d.get("o", 0))),
            high=float(d.get("high", d.get("h", 0))),
            low=float(d.get("low", d.get("l", 0))),
            close=float(d.get("close", d.get("c", 0))),
            volume=float(d.get("volume", d.get("v", 0))),
        )

    def is_valid(self) -> bool:
        """Check if candle has valid price data."""
        return self.close > 0 and self.open > 0 and self.high > 0 and self.low > 0


@dataclass
class TokenData:
    """Complete token data including metadata and price history."""

    token_address: str
    symbol: str
    discovered_at_unix: int
    discovered_age_sec: int
    death_reason: str
    candles: List[Candle]

    # Computed properties
    lifespan_sec: int = field(init=False)
    peak_price: float = field(init=False)
    peak_time_sec: int = field(init=False)
    peak_ratio: float = field(init=False)
    entry_mc: float = field(init=False)

    def __post_init__(self):
        """Compute derived metrics after initialization."""
        if self.candles:
            self.lifespan_sec = len(self.candles)
            prices = [c.close for c in self.candles]
            self.peak_price = max(prices)
            self.peak_time_sec = prices.index(self.peak_price)
            self.entry_mc = self.candles[0].close if self.candles else 0
            self.peak_ratio = self.peak_price / self.entry_mc if self.entry_mc > 0 else 1.0
        else:
            self.lifespan_sec = 0
            self.peak_price = 0
            self.peak_time_sec = 0
            self.entry_mc = 0
            self.peak_ratio = 1.0

    def get_prices(self) -> np.ndarray:
        """Get array of closing prices."""
        return np.array([c.close for c in self.candles], dtype=np.float32)

    def get_volumes(self) -> np.ndarray:
        """Get array of volumes."""
        return np.array([c.volume for c in self.candles], dtype=np.float32)

    def get_ohlcv_matrix(self) -> np.ndarray:
        """Get OHLCV matrix (N x 5)."""
        return np.array(
            [[c.open, c.high, c.low, c.close, c.volume] for c in self.candles],
            dtype=np.float32
        )


def parse_candles(candles_json: str) -> List[Candle]:
    """
    Parse candles JSON string into list of Candle objects.

    Handles the GMGN API response format where candles are nested in data.list,
    and converts from long key names to Candle objects.

    Args:
        candles_json: JSON string containing candle data.

    Returns:
        List of Candle objects sorted by time.

    Raises:
        json.JSONDecodeError: If JSON string is invalid.
        ValueError: If candle data cannot be extracted.

    Example:
        >>> candles = parse_candles('{"data":{"list":[{"time":1234,"open":"100",...}]}}')
        >>> print(len(candles))
        50
    """
    data = json.loads(candles_json)

    # Handle nested API response format from GMGN
    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], dict) and "list" in data["data"]:
            raw_candles = data["data"]["list"]
        elif "list" in data:
            raw_candles = data["list"]
        else:
            raise ValueError(f"Cannot extract candles from dict with keys: {data.keys()}")
    elif isinstance(data, list):
        raw_candles = data
    else:
        raise ValueError(f"Unknown candle format: {type(data)}")

    if not raw_candles:
        return []

    # Convert to Candle objects and filter invalid
    candles = []
    for c in raw_candles:
        try:
            candle = Candle.from_dict(c)
            if candle.is_valid():
                candles.append(candle)
        except (ValueError, TypeError):
            continue

    # Sort by time
    candles.sort(key=lambda x: x.time)

    return candles


def load_raw_data(csv_path: str) -> List[TokenData]:
    """
    Load raw token data from CSV file.

    Args:
        csv_path: Path to CSV file containing token data.

    Returns:
        List of TokenData objects with parsed candles.

    Raises:
        FileNotFoundError: If CSV file does not exist.
        ValueError: If CSV file is empty or has invalid format.

    Example:
        >>> tokens = load_raw_data('data/combined_tokens.csv')
        >>> print(f"Loaded {len(tokens)} tokens")
        Loaded 956 tokens
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Read CSV with proper handling of JSON column
    df = pd.read_csv(
        csv_path,
        quotechar='"',
        escapechar="\\",
        on_bad_lines="warn",
        engine="python"
    )

    if df.empty:
        raise ValueError("CSV file is empty")

    # Check for required columns
    required = ["token_address"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Handle column name variations
    candle_col = None
    for name in ["chart_data_json", "candles"]:
        if name in df.columns:
            candle_col = name
            break

    if candle_col is None:
        raise ValueError("Missing candle data column (chart_data_json or candles)")

    tokens = []
    parse_errors = 0

    for idx, row in df.iterrows():
        try:
            candles = parse_candles(row[candle_col])
            if len(candles) < 5:  # Skip tokens with too few candles
                continue

            token = TokenData(
                token_address=row["token_address"],
                symbol=row.get("symbol", "UNKNOWN"),
                discovered_at_unix=int(row.get("discovered_at_unix", 0)),
                discovered_age_sec=int(row.get("discovered_age_sec", 0)),
                death_reason=str(row.get("death_reason", "unknown")),
                candles=candles,
            )
            tokens.append(token)
        except Exception as e:
            parse_errors += 1
            continue

    if parse_errors > 0:
        print(f"Warning: Failed to parse {parse_errors} tokens")

    print(f"Loaded {len(tokens)} tokens successfully")
    return tokens


def split_tokens_chronological(
    tokens: List[TokenData],
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
    # Sort by discovery time
    sorted_tokens = sorted(tokens, key=lambda t: t.discovered_at_unix)

    n = len(sorted_tokens)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_tokens = sorted_tokens[:train_end]
    val_tokens = sorted_tokens[train_end:val_end]
    test_tokens = sorted_tokens[val_end:]

    print(f"Split: {len(train_tokens)} train, {len(val_tokens)} val, {len(test_tokens)} test")

    return train_tokens, val_tokens, test_tokens


def compute_dataset_statistics(tokens: List[TokenData]) -> Dict[str, Any]:
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

    # Count death reasons
    for t in tokens:
        reason = t.death_reason
        if reason not in stats["death_reasons"]:
            stats["death_reasons"][reason] = 0
        stats["death_reasons"][reason] += 1

    return stats
