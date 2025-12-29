"""
Data processing module for Multi-Model Trading Architecture.

This module handles:
    - Raw data loading and parsing
    - Feature extraction for each model type
    - Label generation for training
    - Dataset creation and data loaders

Author: Trading Team
Date: 2025-12-29
"""

from .loader import (
    load_raw_data,
    parse_candles,
    TokenData,
)
from .features import (
    extract_screener_features,
    extract_timeseries_features,
    extract_exit_features,
    FeatureExtractor,
)
from .labels import (
    generate_screener_labels,
    generate_entry_labels,
    generate_exit_labels,
    LabelGenerator,
)
from .dataset import (
    ScreenerDataset,
    EntryDataset,
    ExitDataset,
    create_data_loaders,
)

__all__ = [
    "load_raw_data",
    "parse_candles",
    "TokenData",
    "extract_screener_features",
    "extract_timeseries_features",
    "extract_exit_features",
    "FeatureExtractor",
    "generate_screener_labels",
    "generate_entry_labels",
    "generate_exit_labels",
    "LabelGenerator",
    "ScreenerDataset",
    "EntryDataset",
    "ExitDataset",
    "create_data_loaders",
]
