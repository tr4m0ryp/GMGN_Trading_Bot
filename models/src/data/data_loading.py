"""
Data loading utilities for Multi-Model Trading Architecture.

Provides dataclasses for candle and token data, CSV parsing,
and chronological train/val/test splitting.

Dependencies: json, pandas, numpy
Date: 2025-12-29
"""

import json
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

import pandas as pd


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Candle:
    """Single OHLCV candle."""
    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    @classmethod
    def from_dict(cls, d: Dict) -> "Candle":
        return cls(
            time=int(d.get("time", d.get("t", 0))),
            open=float(d.get("open", d.get("o", 0))),
            high=float(d.get("high", d.get("h", 0))),
            low=float(d.get("low", d.get("l", 0))),
            close=float(d.get("close", d.get("c", 0))),
            volume=float(d.get("volume", d.get("v", 0))),
        )

    def is_valid(self) -> bool:
        return self.close > 0 and self.open > 0


@dataclass
class TokenData:
    """Aggregated data for a single token."""
    token_address: str
    symbol: str
    discovered_at_unix: int
    discovered_age_sec: int
    death_reason: str
    candles: List[Candle]
    lifespan_sec: int = field(init=False)
    peak_price: float = field(init=False)
    peak_time_sec: int = field(init=False)
    peak_ratio: float = field(init=False)
    entry_mc: float = field(init=False)

    def __post_init__(self):
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


# =============================================================================
# Candle Parsing
# =============================================================================

def parse_candles(candles_json: str) -> List[Candle]:
    """Parse JSON candles string into a sorted list of Candle objects."""
    data = json.loads(candles_json)

    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], dict) and "list" in data["data"]:
            raw_candles = data["data"]["list"]
        elif "list" in data:
            raw_candles = data["list"]
        else:
            return []
    elif isinstance(data, list):
        raw_candles = data
    else:
        return []

    candles = []
    for c in raw_candles:
        try:
            candle = Candle.from_dict(c)
            if candle.is_valid():
                candles.append(candle)
        except (ValueError, TypeError):
            continue

    candles.sort(key=lambda x: x.time)
    return candles


# =============================================================================
# CSV Loading
# =============================================================================

def load_raw_data(csv_path: str) -> List[TokenData]:
    """Load tokens from CSV file.

    Expects columns: token_address, symbol, discovered_at_unix,
    discovered_age_sec, death_reason, and a candle data column
    (chart_data_json or candles).

    Tokens with fewer than 10 valid candles are skipped.
    """
    df = pd.read_csv(
        csv_path, quotechar='"', escapechar="\\",
        on_bad_lines="warn", engine="python",
    )

    candle_col = None
    for name in ["chart_data_json", "candles"]:
        if name in df.columns:
            candle_col = name
            break

    if candle_col is None:
        raise ValueError("Missing candle data column")

    tokens = []
    for _, row in df.iterrows():
        try:
            candles = parse_candles(row[candle_col])
            if len(candles) < 10:
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
        except Exception:
            continue

    return tokens


# =============================================================================
# Chronological Splitting
# =============================================================================

def split_tokens_chronological(
    tokens: List[TokenData],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[List[TokenData], List[TokenData], List[TokenData]]:
    """Split tokens chronologically by discovered_at_unix.

    Returns (train, val, test) token lists.
    """
    sorted_tokens = sorted(tokens, key=lambda t: t.discovered_at_unix)
    n = len(sorted_tokens)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return sorted_tokens[:train_end], sorted_tokens[train_end:val_end], sorted_tokens[val_end:]
