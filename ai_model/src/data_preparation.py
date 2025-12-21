"""
Data preprocessing and feature extraction for trading model.

This module implements data loading, feature extraction, and training sample
generation with full historical context. Each sample contains all price history
from token discovery to current timestamp, simulating real market observation.

Dependencies:
    numpy: Numerical computations
    pandas: Data manipulation
    torch: Tensor operations

Author: Trading Team
Date: 2025-12-21
"""

import json
import pickle
from typing import List, Dict, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from config import (
    FIXED_POSITION_SIZE,
    DELAY_SECONDS,
    TOTAL_FEE_PER_TX,
    MIN_HISTORY_LENGTH,
    BUY_THRESHOLD,
    HOLD_THRESHOLD,
)


def load_raw_data(csv_path: str) -> pd.DataFrame:
    """
    Load raw token data from CSV.

    Args:
        csv_path: Path to CSV file containing token data.

    Returns:
        DataFrame with columns: token_address, symbol, discovered_at_unix,
        discovered_age_sec, death_reason, candles.

    Raises:
        FileNotFoundError: If CSV file does not exist.
        ValueError: If CSV file is empty or has invalid format.

    Example:
        >>> df = load_raw_data('data/raw/rawdata.csv')
        >>> print(f"Loaded {len(df)} tokens")
        Loaded 329 tokens
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(
        csv_path,
        quotechar='"',
        escapechar='\\',
        on_bad_lines='warn',
        engine='python'
    )

    if df.empty:
        raise ValueError("CSV file is empty")

    required_columns = ['token_address', 'candles']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    return df


def parse_candles(candles_json: str) -> List[Dict[str, float]]:
    """
    Parse candles JSON string into list of dictionaries.

    Args:
        candles_json: JSON string containing candle data.

    Returns:
        List of candle dictionaries with keys: t, o, h, l, c, v.

    Raises:
        json.JSONDecodeError: If JSON string is invalid.

    Example:
        >>> candles = parse_candles('[{"t":1234,"o":100,"h":110,"l":95,"c":105,"v":1000}]')
        >>> print(len(candles))
        1
    """
    return json.loads(candles_json)


def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index (RSI).

    Computes RSI using exponential moving average of gains and losses.
    For sequences shorter than period, uses available data.

    Args:
        prices: Array of closing prices.
        period: RSI period. Default is 14.

    Returns:
        Array of RSI values (0-100 range).

    Note:
        First RSI value will be 50 (neutral) when insufficient history.
    """
    if len(prices) < 2:
        return np.full(len(prices), 50.0)

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    rsi_values = np.zeros(len(prices))
    rsi_values[0] = 50.0

    for i in range(1, len(prices)):
        window_start = max(0, i - period)
        avg_gain = np.mean(gains[window_start:i]) if i > window_start else 0
        avg_loss = np.mean(losses[window_start:i]) if i > window_start else 0

        if avg_loss == 0:
            rsi_values[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_values[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi_values


def calculate_macd(prices: np.ndarray,
                  fast: int = 12,
                  slow: int = 26,
                  signal: int = 9) -> np.ndarray:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Args:
        prices: Array of closing prices.
        fast: Fast EMA period. Default is 12.
        slow: Slow EMA period. Default is 26.
        signal: Signal line EMA period. Default is 9.

    Returns:
        Array of MACD values.
    """
    if len(prices) < 2:
        return np.zeros(len(prices))

    ema_fast = pd.Series(prices).ewm(span=fast, adjust=False).mean().values
    ema_slow = pd.Series(prices).ewm(span=slow, adjust=False).mean().values
    macd = ema_fast - ema_slow

    return macd


def calculate_bollinger_bands(prices: np.ndarray,
                              period: int = 20,
                              num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Bollinger Bands.

    Args:
        prices: Array of closing prices.
        period: Moving average period. Default is 20.
        num_std: Number of standard deviations. Default is 2.0.

    Returns:
        Tuple of (upper_band, lower_band) arrays.
    """
    if len(prices) < 2:
        return np.full(len(prices), prices[0]), np.full(len(prices), prices[0])

    sma = pd.Series(prices).rolling(window=period, min_periods=1).mean().values
    std = pd.Series(prices).rolling(window=period, min_periods=1).std().values

    upper = sma + (num_std * std)
    lower = sma - (num_std * std)

    return upper, lower


def calculate_vwap(candles: List[Dict[str, float]]) -> np.ndarray:
    """
    Calculate Volume-Weighted Average Price (VWAP).

    Computes cumulative VWAP from token discovery to each timestamp.

    Args:
        candles: List of candle dictionaries with o, h, l, c, v keys.

    Returns:
        Array of VWAP values.

    Example:
        >>> candles = [{"o": 100, "h": 110, "l": 95, "c": 105, "v": 1000}]
        >>> vwap = calculate_vwap(candles)
    """
    typical_prices = np.array([(c['h'] + c['l'] + c['c']) / 3 for c in candles])
    volumes = np.array([c['v'] for c in candles])

    cumulative_tp_vol = np.cumsum(typical_prices * volumes)
    cumulative_vol = np.cumsum(volumes)

    vwap = np.divide(cumulative_tp_vol, cumulative_vol,
                     where=cumulative_vol != 0,
                     out=typical_prices.copy())

    return vwap


def calculate_momentum(prices: np.ndarray, period: int = 10) -> np.ndarray:
    """
    Calculate price momentum (rate of change).

    Args:
        prices: Array of closing prices.
        period: Lookback period. Default is 10.

    Returns:
        Array of momentum values (percentage change).
    """
    momentum = np.zeros(len(prices))

    for i in range(len(prices)):
        lookback_idx = max(0, i - period)
        if prices[lookback_idx] != 0:
            momentum[i] = (prices[i] - prices[lookback_idx]) / prices[lookback_idx]

    return momentum


def extract_features(candles: List[Dict[str, float]]) -> np.ndarray:
    """
    Extract OHLCV and technical indicators from candles.

    Creates feature matrix with 11 features per timestep:
    - OHLCV (5 features): open, high, low, close, volume
    - Technical indicators (6 features): RSI, MACD, BB_upper, BB_lower, VWAP, Momentum

    Args:
        candles: List of candle dictionaries.

    Returns:
        Feature array of shape (len(candles), 11).

    Example:
        >>> candles = load_token_candles('token_ABC.csv')
        >>> features = extract_features(candles)
        >>> print(features.shape)
        (285, 11)
    """
    opens = np.array([c['o'] for c in candles])
    highs = np.array([c['h'] for c in candles])
    lows = np.array([c['l'] for c in candles])
    closes = np.array([c['c'] for c in candles])
    volumes = np.array([c['v'] for c in candles])

    rsi = calculate_rsi(closes)
    macd = calculate_macd(closes)
    bb_upper, bb_lower = calculate_bollinger_bands(closes)
    vwap = calculate_vwap(candles)
    momentum = calculate_momentum(closes)

    features = np.column_stack([
        opens, highs, lows, closes, volumes,
        rsi, macd, bb_upper, bb_lower, vwap, momentum
    ])

    return features.astype(np.float32)


def get_execution_price(candles: List[Dict[str, float]],
                       start_idx: int,
                       is_buy: bool = True) -> float:
    """
    Simulate execution price with realistic delay.

    Simulates 1-second transaction confirmation delay with worst-case slippage.

    Args:
        candles: List of candle dictionaries.
        start_idx: Index where order is placed.
        is_buy: True for buy order, False for sell order.

    Returns:
        Execution price (highest in window for buy, lowest for sell).

    Note:
        Uses worst-case pricing to simulate realistic market conditions.
    """
    end_idx = min(start_idx + DELAY_SECONDS + 1, len(candles))
    delay_window = candles[start_idx:end_idx]

    if not delay_window:
        return candles[start_idx]['c']

    if is_buy:
        return max(c['h'] for c in delay_window)
    else:
        return min(c['l'] for c in delay_window)


def calculate_net_profit(buy_price: float, sell_price: float) -> float:
    """
    Calculate net profit after transaction fees.

    Args:
        buy_price: Buy execution price.
        sell_price: Sell execution price.

    Returns:
        Net profit in SOL after fees.

    Note:
        Includes Jito tips, gas fees, and priority fees (~7% total).
    """
    tokens = FIXED_POSITION_SIZE / buy_price
    sell_value = tokens * sell_price
    net_value = sell_value - (2 * TOTAL_FEE_PER_TX)
    return net_value - FIXED_POSITION_SIZE


def prepare_realistic_training_data(token_candles: List[Dict[str, float]],
                                   min_history: int = MIN_HISTORY_LENGTH
                                   ) -> List[Dict[str, Any]]:
    """
    Prepare training data with full historical context.

    Creates training samples where each sample contains ALL price history
    from token discovery to current timestamp, simulating real market
    conditions. Includes realistic Jito fee simulation and 1-second
    execution delay.

    Args:
        token_candles: List of OHLCV candles for a single token.
            Each candle dict must contain: 'o', 'h', 'l', 'c', 'v' keys.
        min_history: Minimum number of candles required before generating
            samples. Default is 30 (30 seconds of history).

    Returns:
        List of training samples, each containing:
            - features: np.ndarray of shape (seq_len, 11)
            - label: int (0=HOLD, 1=BUY, 2=SELL)
            - seq_length: int (actual sequence length)
            - timestamp: int (seconds since discovery)
            - buy_price: float (simulated execution price)
            - potential_profit_pct: float (NET profit potential)

    Raises:
        ValueError: If token_candles is empty or contains invalid data.

    Example:
        >>> candles = parse_candles(raw_data['candles'].iloc[0])
        >>> samples = prepare_realistic_training_data(candles, min_history=30)
        >>> print(f"Generated {len(samples)} training samples")
        Generated 285 training samples

    Note:
        This function simulates worst-case execution with 1-second delay.
        Buy orders use highest price in delay window, sell orders use lowest.
        All profit calculations include Jito fees (~7% per round trip).
    """
    if not token_candles:
        raise ValueError("token_candles cannot be empty")

    if len(token_candles) < min_history:
        return []

    samples = []

    for current_time in range(min_history, len(token_candles)):
        historical_data = token_candles[0:current_time]

        features = extract_features(historical_data)

        buy_price = get_execution_price(token_candles, current_time, is_buy=True)

        future_end = min(current_time + DELAY_SECONDS + 20, len(token_candles))
        future_candles = token_candles[current_time + DELAY_SECONDS:future_end]

        if not future_candles:
            continue

        best_sell_price = max(c['h'] for c in future_candles)

        net_profit = calculate_net_profit(buy_price, best_sell_price)
        profit_pct = net_profit / FIXED_POSITION_SIZE

        if profit_pct > BUY_THRESHOLD:
            label = 1
        elif profit_pct < HOLD_THRESHOLD:
            label = 0
        else:
            label = 2

        samples.append({
            'features': features,
            'label': label,
            'seq_length': current_time,
            'timestamp': current_time,
            'buy_price': buy_price,
            'potential_profit_pct': profit_pct,
        })

    return samples


class TradingDataset(Dataset):
    """
    PyTorch Dataset for variable-length trading sequences.

    Loads preprocessed training samples and provides them in a format
    suitable for PyTorch DataLoader with variable-length sequences.

    Attributes:
        samples: List of training samples from prepare_realistic_training_data.

    Args:
        samples: List of sample dictionaries.

    Example:
        >>> dataset = TradingDataset(all_samples)
        >>> print(f"Dataset size: {len(dataset)}")
        >>> sample = dataset[0]
        >>> print(f"Features shape: {sample['features'].shape}")
    """

    def __init__(self, samples: List[Dict[str, Any]]):
        """Initialize dataset with preprocessed samples."""
        self.samples = samples

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Sample index.

        Returns:
            Dictionary containing features, label, and metadata.
        """
        return self.samples[idx]


def collate_variable_length(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader with variable-length sequences.

    Pads sequences to the same length within a batch for efficient processing.

    Args:
        batch: List of samples from TradingDataset.

    Returns:
        Dictionary with padded tensors:
            - features: (batch, max_seq_len, 11)
            - labels: (batch,)
            - seq_lengths: (batch,)

    Example:
        >>> loader = DataLoader(dataset, batch_size=32, collate_fn=collate_variable_length)
        >>> for batch in loader:
        ...     print(batch['features'].shape)
    """
    features = [torch.FloatTensor(item['features']) for item in batch]
    labels = torch.LongTensor([item['label'] for item in batch])
    seq_lengths = torch.LongTensor([item['seq_length'] for item in batch])

    padded_features = pad_sequence(features, batch_first=True, padding_value=0.0)

    return {
        'features': padded_features,
        'labels': labels,
        'seq_lengths': seq_lengths,
    }


def prepare_datasets(csv_path: str,
                    train_split: float = 0.8,
                    val_split: float = 0.1,
                    test_split: float = 0.1,
                    random_seed: int = 42) -> Tuple[TradingDataset,
                                                     TradingDataset,
                                                     TradingDataset]:
    """
    Load raw data and create train/val/test datasets.

    Loads token data from CSV, generates training samples for each token,
    and splits into train/validation/test sets at the token level.

    Args:
        csv_path: Path to raw CSV file.
        train_split: Fraction of tokens for training. Default is 0.8.
        val_split: Fraction of tokens for validation. Default is 0.1.
        test_split: Fraction of tokens for testing. Default is 0.1.
        random_seed: Random seed for reproducibility. Default is 42.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).

    Example:
        >>> train_ds, val_ds, test_ds = prepare_datasets('data/raw/rawdata.csv')
        >>> print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    """
    np.random.seed(random_seed)

    df = load_raw_data(csv_path)

    indices = np.arange(len(df))
    np.random.shuffle(indices)

    train_end = int(len(df) * train_split)
    val_end = train_end + int(len(df) * val_split)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    train_samples = []
    val_samples = []
    test_samples = []

    for idx in train_indices:
        candles = parse_candles(df.iloc[idx]['candles'])
        samples = prepare_realistic_training_data(candles)
        train_samples.extend(samples)

    for idx in val_indices:
        candles = parse_candles(df.iloc[idx]['candles'])
        samples = prepare_realistic_training_data(candles)
        val_samples.extend(samples)

    for idx in test_indices:
        candles = parse_candles(df.iloc[idx]['candles'])
        samples = prepare_realistic_training_data(candles)
        test_samples.extend(samples)

    train_dataset = TradingDataset(train_samples)
    val_dataset = TradingDataset(val_samples)
    test_dataset = TradingDataset(test_samples)

    return train_dataset, val_dataset, test_dataset


def load_preprocessed_datasets(processed_dir: str) -> Tuple[TradingDataset,
                                                             TradingDataset,
                                                             TradingDataset,
                                                             Dict[str, Any]]:
    """
    Load preprocessed datasets from disk.

    Loads train/val/test datasets that were previously processed and saved
    by the preprocess_data.py script. This is much faster than processing
    from raw CSV each time.

    Args:
        processed_dir: Directory containing preprocessed pickle files.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, metadata).

    Raises:
        FileNotFoundError: If required pickle files are not found.

    Example:
        >>> train_ds, val_ds, test_ds, meta = load_preprocessed_datasets('../data/processed')
        >>> print(f"Loaded {len(train_ds)} train samples")
        >>> print(f"Random seed used: {meta['random_seed']}")
    """
    processed_path = Path(processed_dir)

    if not processed_path.exists():
        raise FileNotFoundError(f"Processed data directory not found: {processed_dir}")

    train_path = processed_path / 'train_samples.pkl'
    val_path = processed_path / 'val_samples.pkl'
    test_path = processed_path / 'test_samples.pkl'
    metadata_path = processed_path / 'metadata.pkl'

    for path in [train_path, val_path, test_path, metadata_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    print(f"Loading preprocessed data from {processed_dir}...")

    with open(train_path, 'rb') as f:
        train_samples = pickle.load(f)
    print(f"  Loaded {len(train_samples):,} train samples")

    with open(val_path, 'rb') as f:
        val_samples = pickle.load(f)
    print(f"  Loaded {len(val_samples):,} validation samples")

    with open(test_path, 'rb') as f:
        test_samples = pickle.load(f)
    print(f"  Loaded {len(test_samples):,} test samples")

    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    print(f"  Loaded metadata")

    train_dataset = TradingDataset(train_samples)
    val_dataset = TradingDataset(val_samples)
    test_dataset = TradingDataset(test_samples)

    return train_dataset, val_dataset, test_dataset, metadata
