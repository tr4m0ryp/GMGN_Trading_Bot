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
                predictions, _ = model(features, seq_lengths)
                loss = criterion(predictions, labels)
                loss = loss / accumulation_steps
        else:
            predictions, _ = model(features, seq_lengths)
            loss = criterion(predictions, labels)
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

            predictions, _ = model(features, seq_lengths)
            loss = criterion(predictions, labels)

            total_loss += loss.item()

            predicted = torch.argmax(predictions, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total

    return avg_loss, accuracy


def train_model(model: nn.Module,
               train_loader: DataLoader,
               val_loader: DataLoader,
               config: Dict[str, Any],
               device: str = 'cuda',
               checkpoint_dir: str = '../models/checkpoints') -> Dict[str, Any]:
    """
    Train the model with early stopping and checkpointing.

    Implements full training loop with:
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

    criterion = nn.CrossEntropyLoss()
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
