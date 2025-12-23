"""
Diagnostic script to test model before full training.

This script loads preprocessed data, creates the model, and runs comprehensive
diagnostics including sanity checks and prediction distribution analysis.

Usage:
    python diagnose.py
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from config import get_config
from models.lstm import VariableLengthLSTMTrader
from data.preparation import load_preprocessed_datasets, collate_variable_length
from training.train import diagnose_training_setup, create_weighted_sampler


def main():
    """Run diagnostics on model and data."""
    print("Loading configuration...")
    config = get_config()

    print("Loading preprocessed data...")
    train_dataset, val_dataset, test_dataset, metadata = load_preprocessed_datasets(
        '../data/processed'
    )

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset):,}")
    print(f"  Val:   {len(val_dataset):,}")
    print(f"  Test:  {len(test_dataset):,}")

    dataloader_config = config['dataloader']

    # Create weighted sampler if enabled
    use_weighted_sampler = config['training'].get('use_weighted_sampler', False)
    if use_weighted_sampler:
        print("\nCreating weighted sampler for balanced training...")
        sampler = create_weighted_sampler(train_dataset, num_classes=3)
        shuffle = False  # Can't use shuffle with sampler
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_variable_length,
        num_workers=dataloader_config['num_workers'],
        pin_memory=dataloader_config['pin_memory'],
        persistent_workers=dataloader_config['persistent_workers'],
        prefetch_factor=dataloader_config['prefetch_factor'],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_variable_length,
        num_workers=dataloader_config['num_workers'],
        pin_memory=dataloader_config['pin_memory'],
        persistent_workers=dataloader_config['persistent_workers'],
        prefetch_factor=dataloader_config['prefetch_factor'],
    )

    print("\nCreating model...")
    model_config = config['model']
    model = VariableLengthLSTMTrader(
        input_size=model_config['input_size'],
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        num_classes=model_config['num_classes'],
        dropout=model_config['dropout']
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = model.to(device)

    print("\nRunning diagnostics...\n")
    passed = diagnose_training_setup(model, train_loader, val_loader, device)

    if passed:
        print("\nAll diagnostics PASSED - ready for training!")
        return 0
    else:
        print("\nDiagnostics FAILED - fix issues before training")
        return 1


if __name__ == '__main__':
    sys.exit(main())
