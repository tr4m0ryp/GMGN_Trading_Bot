"""
Debug script to analyze training data quality.

Checks for issues in features, labels, and data distribution.
"""

import sys
from pathlib import Path
import numpy as np
import pickle

sys.path.insert(0, str(Path(__file__).parent))

from data.preparation import load_preprocessed_datasets


def main():
    """Analyze data quality."""
    print("Loading preprocessed data...")
    train_dataset, val_dataset, test_dataset, metadata = load_preprocessed_datasets(
        '../data/processed'
    )

    print(f"\n{'='*60}")
    print("DETAILED DATA ANALYSIS")
    print(f"{'='*60}")

    # Analyze one sample
    sample = train_dataset[0]
    print(f"\nSample 0:")
    print(f"  Features shape: {sample['features'].shape}")
    print(f"  Label: {sample['label']}")
    print(f"  Seq length: {sample['seq_length']}")

    # Feature statistics across all training samples
    print(f"\n{'='*60}")
    print("Feature Statistics (first 1000 samples)")
    print(f"{'='*60}")

    all_features = []
    all_labels = []
    for i in range(min(1000, len(train_dataset))):
        sample = train_dataset[i]
        all_features.append(sample['features'])
        all_labels.append(sample['label'])

    # Concatenate all features
    concat_features = np.concatenate(all_features, axis=0)
    print(f"\nConcatenated features shape: {concat_features.shape}")
    print(f"Features (15): {['log_close', 'ret_1', 'ret_3', 'ret_5', 'range_ratio', 'volume_log', 'rsi_norm', 'macd_norm', 'bb_upper', 'bb_lower', 'vwap_dev', 'momentum', 'ready_short', 'ready_long', 'in_position']}")

    for feat_idx in range(15):
        feat_col = concat_features[:, feat_idx]
        print(f"\nFeature {feat_idx}:")
        print(f"  Mean: {feat_col.mean():.4f}")
        print(f"  Std:  {feat_col.std():.4f}")
        print(f"  Min:  {feat_col.min():.4f}")
        print(f"  Max:  {feat_col.max():.4f}")
        print(f"  NaN:  {np.isnan(feat_col).sum()}")
        print(f"  Inf:  {np.isinf(feat_col).sum()}")

    # Label distribution
    print(f"\n{'='*60}")
    print("Label Distribution (all training)")
    print(f"{'='*60}")

    all_train_labels = [train_dataset[i]['label'] for i in range(len(train_dataset))]
    label_counts = np.bincount(all_train_labels)

    print(f"HOLD (0): {label_counts[0]:,} ({label_counts[0]/len(all_train_labels)*100:.2f}%)")
    print(f"BUY  (1): {label_counts[1]:,} ({label_counts[1]/len(all_train_labels)*100:.2f}%)")
    print(f"SELL (2): {label_counts[2]:,} ({label_counts[2]/len(all_train_labels)*100:.2f}%)")

    # Check for data leakage or issues
    print(f"\n{'='*60}")
    print("Checking for Issues")
    print(f"{'='*60}")

    # Check if in_position flag correlates perfectly with labels
    in_position_flags = []
    labels = []
    for i in range(min(1000, len(train_dataset))):
        sample = train_dataset[i]
        # in_position is feature 14
        in_pos = sample['features'][:, 14][0]  # First timestep
        in_position_flags.append(in_pos)
        labels.append(sample['label'])

    in_position_flags = np.array(in_position_flags)
    labels = np.array(labels)

    print(f"\nIn-position flag distribution:")
    print(f"  Flat (0.0): {(in_position_flags == 0).sum()}")
    print(f"  In-pos (1.0): {(in_position_flags == 1).sum()}")

    print(f"\nLabel distribution when FLAT:")
    flat_labels = labels[in_position_flags == 0]
    if len(flat_labels) > 0:
        flat_counts = np.bincount(flat_labels, minlength=3)
        print(f"  HOLD: {flat_counts[0]} ({flat_counts[0]/len(flat_labels)*100:.1f}%)")
        print(f"  BUY:  {flat_counts[1]} ({flat_counts[1]/len(flat_labels)*100:.1f}%)")
        print(f"  SELL: {flat_counts[2]} ({flat_counts[2]/len(flat_labels)*100:.1f}%)")

    print(f"\nLabel distribution when IN-POSITION:")
    inpos_labels = labels[in_position_flags == 1]
    if len(inpos_labels) > 0:
        inpos_counts = np.bincount(inpos_labels, minlength=3)
        print(f"  HOLD: {inpos_counts[0]} ({inpos_counts[0]/len(inpos_labels)*100:.1f}%)")
        print(f"  BUY:  {inpos_counts[1]} ({inpos_counts[1]/len(inpos_labels)*100:.1f}%)")
        print(f"  SELL: {inpos_counts[2]} ({inpos_counts[2]/len(inpos_labels)*100:.1f}%)")


if __name__ == '__main__':
    main()
