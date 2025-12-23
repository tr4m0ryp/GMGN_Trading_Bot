"""
Final comprehensive sanity check with extended training.

Tests if the model can learn the trading data with more optimization steps and higher learning rate.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from config import get_config
from models.lstm import VariableLengthLSTMTrader
from data.preparation import load_preprocessed_datasets, collate_variable_length
from training.train import create_weighted_sampler


def extended_sanity_check():
    """Run extended sanity check with more steps."""
    print("Loading data...")
    config = get_config()
    train_dataset, _, _, _ = load_preprocessed_datasets('../data/processed')

    print(f"Creating weighted sampler...")
    sampler = create_weighted_sampler(train_dataset, num_classes=3)

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        sampler=sampler,
        collate_fn=collate_variable_length,
        num_workers=0,  # No workers for sanity check
        pin_memory=False,
    )

    print("\nCreating model...")
    model = VariableLengthLSTMTrader(
        input_size=14,
        hidden_size=128,
        num_layers=2,
        dropout=0.3
    )

    device = 'cpu'
    model = model.to(device)

    batch = next(iter(train_loader))
    features = batch['features'].to(device)
    labels = batch['labels'].to(device)
    seq_lengths = batch['seq_lengths']

    print(f"\nBatch info:")
    print(f"  Features: {features.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Label dist: HOLD={(labels==0).sum().item()}, BUY={(labels==1).sum().item()}, SELL={(labels==2).sum().item()}")

    criterion = nn.CrossEntropyLoss()

    print("\n" + "="*60)
    print("EXTENDED SANITY CHECK - 500 steps, LR=0.01")
    print("="*60)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for step in range(500):
        optimizer.zero_grad()
        logits, _ = model(features, seq_lengths)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == 499:
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                acc = (preds == labels).float().mean().item()
                class_acc = {}
                for cls in [0, 1, 2]:
                    mask = labels == cls
                    if mask.sum() > 0:
                        class_acc[cls] = (preds[mask] == labels[mask]).float().mean().item()
                    else:
                        class_acc[cls] = 0.0

                print(f"Step {step:3d}: loss={loss.item():.4f}, acc={acc:.4f} | "
                      f"HOLD:{class_acc[0]:.2f} BUY:{class_acc[1]:.2f} SELL:{class_acc[2]:.2f}")

    final_acc = (preds == labels).float().mean().item()

    print("\n" + "="*60)
    if final_acc > 0.8:
        print(f"✓ SUCCESS: Model can learn (acc={final_acc:.2f})")
        print("  Ready for full training!")
        return True
    elif final_acc > 0.6:
        print(f"⚠ PARTIAL: Model learning slowly (acc={final_acc:.2f})")
        print("  May work with more epochs/different architecture")
        return True
    else:
        print(f"✗ FAILURE: Model cannot learn (acc={final_acc:.2f})")
        print("  Data may have fundamental issues")
        return False


if __name__ == '__main__':
    success = extended_sanity_check()
    sys.exit(0 if success else 1)
