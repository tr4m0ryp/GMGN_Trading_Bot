"""
Training script for Model 2: Entry Timing Optimizer.

Trains LSTM model on time-series features to predict
optimal entry timing for screener-passed tokens.

Author: Trading Team
Date: 2025-12-29
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import time
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from ..config import EntryConfig, DEFAULT_ENTRY_CONFIG
from ..models.entry import EntryModel, create_entry_model, save_entry_model
from ..data.dataset import EntryDataset, prepare_entry_data, create_data_loaders, collate_entry_batch
from ..utils import set_seed, get_device, format_time


def train_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    class_weights: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: Model to train.
        loader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Compute device.
        class_weights: Optional class weights for loss.

    Returns:
        Dictionary with epoch metrics.
    """
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)
        seq_lengths = batch["seq_lengths"]

        optimizer.zero_grad()

        logits = model(features, seq_lengths)
        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_samples += len(labels)

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: str,
) -> Dict[str, float]:
    """
    Validate model.

    Args:
        model: Model to validate.
        loader: Validation data loader.
        criterion: Loss function.
        device: Compute device.

    Returns:
        Dictionary with validation metrics.
    """
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)
            seq_lengths = batch["seq_lengths"]

            logits = model(features, seq_lengths)
            loss = criterion(logits, labels)

            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += len(labels)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Per-class accuracy
    class_acc = {}
    for c in range(3):
        mask = all_labels == c
        if mask.sum() > 0:
            class_acc[c] = (all_preds[mask] == c).mean()

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "class_accuracy": class_acc,
    }


def train_entry_model(
    data_path: str,
    output_dir: str,
    config: Optional[EntryConfig] = None,
    device: Optional[str] = None,
    verbose: int = 1,
) -> Tuple[EntryModel, Dict[str, Any]]:
    """
    Full training pipeline for Model 2 (Entry).

    Args:
        data_path: Path to raw CSV data.
        output_dir: Directory to save model and results.
        config: Model configuration.
        device: Compute device.
        verbose: Verbosity level.

    Returns:
        Tuple of (trained_model, results_dict).
    """
    start_time = time.time()
    set_seed(42)
    device = get_device(device)
    config = config or DEFAULT_ENTRY_CONFIG
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MODEL 2: ENTRY TIMING OPTIMIZER (LSTM)")
    print("=" * 70)
    print(f"Device: {device}")

    # Prepare data
    print("\n[1/5] Preparing data...")
    train_ds, val_ds, test_ds = prepare_entry_data(data_path)

    # Get class distribution for weighting
    train_dist = train_ds.get_class_distribution()
    total = sum(train_dist.values())
    class_weights = torch.tensor([
        total / (3 * train_dist.get(i, 1)) for i in range(3)
    ], dtype=torch.float32).to(device)

    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val:   {len(val_ds)} samples")
    print(f"  Test:  {len(test_ds)} samples")
    print(f"  Class dist: {train_dist}")

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_ds, val_ds, test_ds,
        batch_size=config.batch_size,
        collate_fn=collate_entry_batch,
    )

    # Create model
    print("\n[2/5] Creating model...")
    model = create_entry_model(config, device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup training
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # Training loop
    print("\n[3/5] Training...")
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, config.total_epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, class_weights
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Record history
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["lr"].append(current_lr)

        # Check for improvement
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            save_entry_model(
                model, output_path / "best_entry_model.pt",
                optimizer, epoch, val_metrics
            )
        else:
            patience_counter += 1

        # Print progress
        if verbose > 0 and (epoch % 5 == 0 or epoch == 1):
            print(
                f"  Epoch {epoch:3d}/{config.total_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.3f} | "
                f"LR: {current_lr:.2e}"
            )

        # Early stopping
        if patience_counter >= config.patience:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    # Load best model
    print("\n[4/5] Evaluating best model...")
    checkpoint = torch.load(output_path / "best_entry_model.pt")
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = validate(model, test_loader, criterion, device)

    print(f"\n  Test Results:")
    print(f"    Loss: {test_metrics['loss']:.4f}")
    print(f"    Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"    Class accuracies: {test_metrics['class_accuracy']}")

    # Save results
    print("\n[5/5] Saving results...")
    results = {
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "test_metrics": {
            "loss": float(test_metrics["loss"]),
            "accuracy": float(test_metrics["accuracy"]),
            "class_accuracy": {str(k): float(v) for k, v in test_metrics["class_accuracy"].items()},
        },
        "history": {k: [float(x) for x in v] for k, v in history.items()},
        "total_time_seconds": time.time() - start_time,
    }

    with open(output_path / "entry_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total time: {format_time(results['total_time_seconds'])}")
    print(f"Best epoch: {best_epoch}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print("=" * 70)

    return model, results
