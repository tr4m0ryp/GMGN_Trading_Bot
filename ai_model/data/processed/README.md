# Processed Training Data

This directory contains preprocessed training data generated from the raw token data.

## Files

### train_samples.pkl (124 MB)
Training samples containing 25,256 samples from 246 tokens.
- Each sample includes:
  - `features`: np.ndarray of shape (seq_length, 11) with OHLCV + technical indicators
  - `label`: int (0=HOLD, 1=BUY, 2=SELL)
  - `seq_length`: int (actual sequence length)
  - `timestamp`: int (seconds since discovery)
  - `buy_price`: float (simulated execution price)
  - `potential_profit_pct`: float (NET profit potential)

### val_samples.pkl (16 MB)
Validation samples containing 2,983 samples from 31 tokens.
Same structure as training samples.

### test_samples.pkl (15 MB)
Test samples containing 3,061 samples from 33 tokens.
Same structure as training samples.

### metadata.pkl (22 KB)
Metadata about the preprocessing:
- Token counts per split
- Sample counts per split
- Token information (address, symbol, num_samples, num_candles)
- Random seed used (42)
- Split ratios (0.8 train, 0.1 val, 0.1 test)

## Data Statistics

### Sequence Lengths (Training Set)
- Min: 30 candles
- Max: 499 candles
- Mean: 115.1 candles
- Median: 90.0 candles

### Label Distribution (Training Set)
- HOLD (0): 15,655 samples (62.0%)
- BUY (1): 6,082 samples (24.1%)
- SELL (2): 3,519 samples (13.9%)

### Profit Potential for BUY Signals (Training Set)
- Min: 10.00%
- Max: 322.09%
- Mean: 26.75%

## How to Load

Use the `load_preprocessed_datasets()` function from `src/data_preparation.py`:

```python
from data_preparation import load_preprocessed_datasets

train_ds, val_ds, test_ds, metadata = load_preprocessed_datasets('../data/processed')
```

## Regenerating Processed Data

If you need to regenerate the processed data (e.g., with different splits or parameters):

```bash
cd src
python preprocess_data.py --help
```

Example with custom parameters:
```bash
python preprocess_data.py \
    --csv-path ../data/raw/rawdata.csv \
    --output-dir ../data/processed \
    --train-split 0.8 \
    --val-split 0.1 \
    --test-split 0.1 \
    --seed 42
```

## Notes

- Data was preprocessed on 2025-12-21
- Source: ../data/raw/rawdata.csv (329 tokens, 3 skipped due to parsing errors)
- All samples include full historical context from token discovery
- Realistic execution simulation with 1-second delay and worst-case slippage
- Fee-aware labels accounting for ~7% round-trip transaction costs
