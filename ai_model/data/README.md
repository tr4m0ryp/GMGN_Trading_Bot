# Data Directory

## Structure

- `raw/` - Raw CSV files from GMGN token logger
- `processed/` - Preprocessed training/validation data (features, labels)
- `test/` - Test dataset for final evaluation

## Data Flow

1. Raw CSV files → `raw/`
2. Run `src/data_preparation.py` → generates `processed/` files
3. Train/validation data in `processed/`
4. Test data in `test/` (never used during training)

## File Format

Processed files are saved as:
- `features_train.npy` - Training features (variable-length sequences)
- `labels_train.npy` - Training labels (BUY/HOLD/SELL)
- `features_val.npy` - Validation features
- `labels_val.npy` - Validation labels
- `features_test.npy` - Test features
- `labels_test.npy` - Test labels
