"""
Training logic and utilities for trading model.

This module implements the training loop with mixed precision training,
validation, early stopping, and checkpoint management. Optimized for
NVIDIA Tesla T4 GPU with automatic mixed precision (AMP).

Dependencies:
    torch: Deep learning framework
    torch.cuda.amp: Automatic mixed precision
    torch.utils.data: DataLoader utilities
    tqdm: Progress bars

Author: Trading Team
Date: 2025-12-21
"""

from typing import Dict, Any, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from utils import save_checkpoint

import numpy as np
from torch.utils.data import WeightedRandomSampler


def sanity_check_overfit_batch(model: nn.Module,
                               train_loader,
                               device: str,
                               num_steps: int = 100) -> bool:
    """
    Verify model can overfit a single batch (proves learning is possible).

    This sanity check trains the model on a single batch for many steps.
    If the model cannot achieve high accuracy on one batch, there is likely
    a fundamental issue with the architecture or data.

    Args:
        model: PyTorch model to test.
        train_loader: DataLoader for training data.
        device: Device to train on.
        num_steps: Number of optimization steps. Default is 100.

    Returns:
        True if model achieves >80% accuracy on the batch, False otherwise.
    """
    import copy
    test_model = copy.deepcopy(model)
    test_model.train()

    batch = next(iter(train_loader))
    features = batch['features'].to(device)
    labels = batch['labels'].to(device)
    seq_lengths = batch['seq_lengths']

    optimizer = torch.optim.Adam(test_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    print("Sanity check: overfitting single batch...")
    for step in range(num_steps):
        optimizer.zero_grad()
        logits, _ = test_model(features, seq_lengths)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            acc = (logits.argmax(dim=1) == labels).float().mean()
            print(f"  Step {step}: loss={loss.item():.4f}, acc={acc.item():.4f}")

    with torch.no_grad():
        logits, _ = test_model(features, seq_lengths)
        final_acc = (logits.argmax(dim=1) == labels).float().mean().item()

    if final_acc < 0.8:
        print(f"WARNING: Model cannot overfit single batch (acc={final_acc:.2f})")
        print("         Check architecture or data quality")
        return False

    print(f"Sanity check PASSED: model can learn (acc={final_acc:.2f})")
    return True


def analyze_predictions(model: nn.Module,
                       val_loader,
                       device: str) -> tuple:
    """
    Analyze prediction distribution to detect class collapse.

    Computes the distribution of predictions vs actual labels to identify
    if the model is collapsing to always predict one class.

    Args:
        model: PyTorch model to analyze.
        val_loader: DataLoader for validation data.
        device: Device to run on.

    Returns:
        Tuple of (predictions_list, labels_list).
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            seq_lengths = batch['seq_lengths']

            logits, _ = model(features, seq_lengths)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    print("\nPrediction Distribution:")
    class_names = ['HOLD', 'BUY', 'SELL']
    for cls in [0, 1, 2]:
        pred_count = sum(1 for p in all_preds if p == cls)
        label_count = sum(1 for l in all_labels if l == cls)
        print(f"  {class_names[cls]} ({cls}): pred={pred_count:5d} ({pred_count/len(all_preds)*100:5.1f}%), "
              f"actual={label_count:5d} ({label_count/len(all_labels)*100:5.1f}%)")

    return all_preds, all_labels


def diagnose_training_setup(model: nn.Module,
                            train_loader,
                            val_loader,
                            device: str) -> bool:
    """
    Run comprehensive diagnostics before full training.

    Checks data quality, feature statistics, label distribution, and
    model learning capability.

    Args:
        model: PyTorch model to diagnose.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        device: Device to run on.

    Returns:
        True if all checks pass, False otherwise.
    """
    print("=" * 60)
    print("TRAINING DIAGNOSTICS")
    print("=" * 60)

    batch = next(iter(train_loader))

    print(f"\n[Data]")
    print(f"  Batch size: {batch['features'].shape[0]}")
    print(f"  Sequence lengths: min={batch['seq_lengths'].min().item()}, "
          f"max={batch['seq_lengths'].max().item()}")
    print(f"  Feature shape: {batch['features'].shape}")

    print(f"\n[Labels in batch]")
    labels = batch['labels'].numpy()
    class_names = ['HOLD', 'BUY', 'SELL']
    for cls in [0, 1, 2]:
        pct = (labels == cls).sum() / len(labels) * 100
        print(f"  {class_names[cls]} ({cls}): {pct:.1f}%")

    feats = batch['features'].numpy()
    print(f"\n[Features]")
    print(f"  Mean: {feats.mean():.4f}, Std: {feats.std():.4f}")
    print(f"  Min: {feats.min():.4f}, Max: {feats.max():.4f}")
    print(f"  NaN count: {np.isnan(feats).sum()}")
    print(f"  Inf count: {np.isinf(feats).sum()}")

    print(f"\n[Sanity Check]")
    passed = sanity_check_overfit_batch(model, train_loader, device, num_steps=50)

    print(f"\n[Initial Predictions]")
    analyze_predictions(model, val_loader, device)

    print("=" * 60)
    return passed


class FocalLoss(nn.Module):
    """Standard focal loss for multi-class classification."""

    def __init__(self,
                 weight: torch.Tensor = None,
                 gamma: float = 2.0,
                 reduction: str = 'mean'):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal.mean()
        if self.reduction == 'sum':
            return focal.sum()
        return focal


def train_epoch(model: nn.Module,
               train_loader: DataLoader,
               criterion: nn.Module,
               optimizer: torch.optim.Optimizer,
               device: str,
               scaler: Optional[GradScaler] = None,
               gradient_clip: float = 1.0,
               accumulation_steps: int = 1) -> float:
    """
    Train model for one epoch.

    Args:
        model: PyTorch model to train.
        train_loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on ('cuda' or 'cpu').
        scaler: Gradient scaler for mixed precision. If None, uses FP32.
        gradient_clip: Maximum gradient norm. Default is 1.0.
        accumulation_steps: Gradient accumulation steps. Default is 1.

    Returns:
        Average training loss for the epoch.

    Example:
        >>> scaler = GradScaler()
        >>> loss = train_epoch(model, train_loader, criterion, optimizer,
        ...                   device='cuda', scaler=scaler)
        >>> print(f"Epoch loss: {loss:.4f}")
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    optimizer.zero_grad()

    for i, batch in enumerate(tqdm(train_loader, desc="Training")):
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)
        seq_lengths = batch['seq_lengths']

        if scaler is not None:
            with autocast(device_type='cuda' if device == 'cuda' else 'cpu'):
                logits, _ = model(features, seq_lengths)
                loss = criterion(logits, labels)
                loss = loss / accumulation_steps
        else:
            logits, _ = model(features, seq_lengths)
            loss = criterion(logits, labels)
            loss = loss / accumulation_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()

            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        num_batches += 1

    return total_loss / num_batches


def validate(model: nn.Module,
            val_loader: DataLoader,
            criterion: nn.Module,
            device: str) -> Tuple[float, float]:
    """
    Validate model on validation set.

    Args:
        model: PyTorch model to validate.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        device: Device to validate on ('cuda' or 'cpu').

    Returns:
        Tuple of (average_loss, accuracy).

    Example:
        >>> val_loss, val_acc = validate(model, val_loader, criterion, device='cuda')
        >>> print(f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            seq_lengths = batch['seq_lengths']

            logits, _ = model(features, seq_lengths)
            loss = criterion(logits, labels)

            total_loss += loss.item()

            predicted = torch.argmax(logits, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total

    return avg_loss, accuracy


def compute_class_weights(train_loader: DataLoader,
                         num_classes: int,
                         device: str,
                         smoothing: float = 0.3) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets.

    Uses inverse class frequency weighting with smoothing to handle class imbalance.
    Weights are normalized so they average to 1.0. Smoothing reduces extreme weights.

    Args:
        train_loader: DataLoader for training data.
        num_classes: Number of classes.
        device: Device to place weights on.
        smoothing: Smoothing factor (0=no smoothing, 1=uniform weights). Default 0.3.

    Returns:
        Tensor of class weights of shape (num_classes,).

    Example:
        >>> weights = compute_class_weights(train_loader, 3, 'cuda', smoothing=0.3)
        >>> print(f"Class weights: {weights}")
    """
    class_counts = torch.zeros(num_classes)

    for batch in train_loader:
        labels = batch['labels']
        for i in range(num_classes):
            class_counts[i] += (labels == i).sum().item()

    total_samples = class_counts.sum()
    class_weights = total_samples / (num_classes * class_counts)

    # Apply smoothing: move weights toward 1.0
    class_weights = class_weights * (1 - smoothing) + smoothing

    # Normalize weights so they average to 1.0
    class_weights = class_weights / class_weights.mean()

    return class_weights.to(device)


def create_weighted_sampler(dataset, num_classes: int = 3) -> WeightedRandomSampler:
    """
    Create weighted sampler to balance class distribution.

    Oversamples minority classes to achieve balanced batches during training.
    This is complementary to class weights in the loss function.

    Args:
        dataset: PyTorch dataset with samples containing 'label' field.
        num_classes: Number of classes. Default is 3.

    Returns:
        WeightedRandomSampler for use in DataLoader.

    Example:
        >>> sampler = create_weighted_sampler(train_dataset)
        >>> train_loader = DataLoader(train_dataset, sampler=sampler, ...)
    """
    labels = [dataset[i]['label'] for i in range(len(dataset))]
    class_counts = np.bincount(labels, minlength=num_classes)

    # Inverse frequency weights
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    print(f"Created weighted sampler:")
    print(f"  Class counts: {class_counts}")
    print(f"  Class weights: {class_weights}")

    return sampler


def train_model(model: nn.Module,
               train_loader: DataLoader,
               val_loader: DataLoader,
               config: Dict[str, Any],
               device: str = 'cuda',
               checkpoint_dir: str = '../models/checkpoints') -> Dict[str, Any]:
    """
    Train the model with early stopping and checkpointing.

    Implements full training loop with:
    - Class weighting for imbalanced datasets
    - Mixed precision training (if enabled)
    - Gradient accumulation
    - Early stopping
    - Checkpoint saving
    - Learning rate scheduling

    Args:
        model: PyTorch model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        config: Configuration dictionary with training parameters.
        device: Device to train on. Default is 'cuda'.
        checkpoint_dir: Directory to save checkpoints. Default is '../models/checkpoints'.

    Returns:
        Dictionary containing training history:
            - train_losses: List of training losses per epoch
            - val_losses: List of validation losses per epoch
            - val_accuracies: List of validation accuracies per epoch
            - best_epoch: Epoch with best validation loss
            - best_val_loss: Best validation loss achieved

    Example:
        >>> from config import get_config
        >>> config = get_config()
        >>> history = train_model(model, train_loader, val_loader, config)
        >>> print(f"Best validation loss: {history['best_val_loss']:.4f}")
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    training_config = config['training']
    epochs = training_config['epochs']
    learning_rate = training_config['learning_rate']
    patience = training_config['early_stopping_patience']
    gradient_clip = training_config['gradient_clip_value']
    weight_decay = training_config['weight_decay']
    use_mixed_precision = training_config['use_mixed_precision']
    accumulation_steps = training_config['accumulation_steps']

    # Compute class weights to handle class imbalance
    num_classes = config['model']['num_classes']
    print("Computing class weights from training data...")
    class_weights = compute_class_weights(train_loader, num_classes, device)
    print(f"Class weights: {class_weights.cpu().numpy()}")

    use_focal = training_config.get('use_focal_loss', False)
    focal_gamma = training_config.get('focal_gamma', 2.0)
    label_smoothing = training_config.get('label_smoothing', 0.0)

    if use_focal:
        criterion = FocalLoss(weight=class_weights, gamma=focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    scaler = GradScaler(device='cuda') if use_mixed_precision and device == 'cuda' else None

    best_val_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0

    train_losses = []
    val_losses = []
    val_accuracies = []

    print(f"Starting training on {device}")
    print(f"Mixed precision: {use_mixed_precision}")
    print(f"Gradient accumulation steps: {accumulation_steps}")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler=scaler,
            gradient_clip=gradient_clip,
            accumulation_steps=accumulation_steps
        )

        val_loss, val_accuracy = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            best_model_path = f"{checkpoint_dir}/../best_model.pth"
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                best_model_path,
                val_accuracy=val_accuracy,
                config=config
            )
            print(f"Saved best model to {best_model_path}")
        else:
            epochs_without_improvement += 1

        checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_{epoch + 1}.pth"
        save_checkpoint(
            model,
            optimizer,
            epoch,
            val_loss,
            checkpoint_path,
            val_accuracy=val_accuracy
        )

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch + 1}")
            break

    print("\nTraining completed")
    print(f"Best epoch: {best_epoch + 1}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {val_accuracies[best_epoch]:.4f}")

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
    }
