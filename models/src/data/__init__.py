"""Data loading, feature extraction, and dataset utilities."""

from .preparation import (
    load_raw_data,
    parse_candles,
    calculate_bollinger_bands,
    calculate_vwap,
    calculate_momentum,
    extract_features,
    get_execution_price,
    calculate_net_profit,
    prepare_realistic_training_data,
)

# Import shared technical indicators (rsi, macd) from canonical source
from .technical_indicators import (
    calculate_rsi,
    calculate_macd,
)

from .dataset_v1 import (
    TradingDataset,
    collate_variable_length,
    collate_for_regression,
)

from .pipeline import (
    prepare_datasets,
    load_preprocessed_datasets,
)

__all__ = [
    'load_raw_data',
    'parse_candles',
    'calculate_rsi',
    'calculate_macd',
    'calculate_bollinger_bands',
    'calculate_vwap',
    'calculate_momentum',
    'extract_features',
    'get_execution_price',
    'calculate_net_profit',
    'prepare_realistic_training_data',
    'TradingDataset',
    'collate_variable_length',
    'collate_for_regression',
    'prepare_datasets',
    'load_preprocessed_datasets',
]
