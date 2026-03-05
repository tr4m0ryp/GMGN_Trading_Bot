"""
CPC Trainer: Phase 1 self-supervised pretraining.

Trains the CPCEncoder using InfoNCE contrastive loss on
all available sequences without using labels.

The learned representations capture temporal patterns
and market dynamics useful for downstream prediction.

Dependencies:
    torch, numpy, tqdm

Date: 2025-12-25
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

from cpc_regression.encoder import CPCEncoder
from cpc_regression.cpc_model import CPCModel
from cpc_regression.config import CPCConfig, detect_gpu, get_config_for_gpu
from training.cpc_dataset import CPCDataset, collate_cpc
from training.cpc_utils import load_pretrained_encoder  # noqa: F401 (re-export)


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: CPCConfig,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler with warmup.

    Args:
        optimizer: Optimizer
        config: CPC configuration
        steps_per_epoch: Number of training steps per epoch

    Returns:
        LR scheduler
    """
    total_steps = config.total_epochs * steps_per_epoch
    warmup_steps = config.warmup_epochs * steps_per_epoch

    if config.scheduler == 'cosine':
        warmup = LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps,
            eta_min=config.min_lr,
        )
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )
    elif config.scheduler == 'linear':
        scheduler = LinearLR(
            optimizer, start_factor=1.0,
            end_factor=config.min_lr / config.learning_rate,
            total_iters=total_steps,
        )
    else:
        # Constant LR (with warmup)
        scheduler = LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0,
            total_iters=warmup_steps,
        )

    return scheduler


def train_cpc(
    data_dir: str,
    output_dir: str,
    config: Optional[CPCConfig] = None,
    device: str = 'cuda',
    verbose: int = 1,
) -> Dict:
    """
    Train CPC model (Phase 1: Self-supervised pretraining).

    Args:
        data_dir: Directory containing preprocessed data
        output_dir: Directory to save trained model
        config: CPCConfig instance. If None, auto-detect GPU and configure.
        device: Device to train on ('cuda' or 'cpu')
        verbose: Verbosity level (0=silent, 1=progress, 2=detailed)

    Returns:
        Dictionary with training results and metrics
    """
    start_time = time.time()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if config is None:
        gpu_config = get_config_for_gpu()
        config = CPCConfig(**gpu_config['cpc'])
        print(f"Auto-configured for GPU: {gpu_config['gpu_type']}")

    if verbose >= 1:
        print(f"CPC Pretraining Configuration:")
        print(f"  Hidden dim: {config.hidden_dim}")
        print(f"  Embed dim: {config.embed_dim}")
        print(f"  LSTM layers: {config.lstm_layers}")
        print(f"  Attention heads: {config.n_heads}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Epochs: {config.total_epochs}")

    # Load data
    data_path = Path(data_dir)
    import pickle

    with open(data_path / 'train_samples.pkl', 'rb') as f:
        train_samples = pickle.load(f)
    with open(data_path / 'val_samples.pkl', 'rb') as f:
        val_samples = pickle.load(f)

    if verbose >= 1:
        print(f"Loaded {len(train_samples)} training samples")
        print(f"Loaded {len(val_samples)} validation samples")

    # Create datasets and loaders
    train_dataset = CPCDataset(train_samples, max_seq_len=config.max_seq_len, min_seq_len=config.min_seq_len)
    val_dataset = CPCDataset(val_samples, max_seq_len=config.max_seq_len, min_seq_len=config.min_seq_len)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=collate_cpc,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collate_cpc,
    )

    # Create model
    encoder = CPCEncoder(
        input_dim=config.input_dim, hidden_dim=config.hidden_dim,
        embed_dim=config.embed_dim, lstm_layers=config.lstm_layers,
        n_heads=config.n_heads, ff_dim=config.ff_dim, dropout=config.dropout,
    )
    model = CPCModel(
        encoder=encoder, ar_hidden=config.ar_hidden,
        prediction_steps=config.prediction_steps, temperature=config.temperature,
    )
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    if verbose >= 1:
        print(f"Model parameters: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    steps_per_epoch = len(train_loader)
    scheduler = create_scheduler(optimizer, config, steps_per_epoch)

    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}

    for epoch in range(config.total_epochs):
        epoch_start = time.time()

        # Train
        model.train()
        train_loss, train_acc, n_batches = 0.0, 0.0, 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.total_epochs}', disable=verbose < 1)

        for batch in pbar:
            features = batch['features'].to(device)
            seq_lengths = batch['seq_lengths'].to(device)
            optimizer.zero_grad()
            loss, metrics = model.compute_loss(features, seq_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_acc += metrics['mean_accuracy']
            n_batches += 1
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{metrics["mean_accuracy"]:.2%}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}',
            })

        train_loss /= n_batches
        train_acc /= n_batches

        # Validate
        model.eval()
        val_loss, val_acc, n_val_batches = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                seq_lengths = batch['seq_lengths'].to(device)
                loss, metrics = model.compute_loss(features, seq_lengths)
                val_loss += loss.item()
                val_acc += metrics['mean_accuracy']
                n_val_batches += 1

        val_loss /= n_val_batches
        val_acc /= n_val_batches

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(scheduler.get_last_lr()[0])

        epoch_time = time.time() - epoch_start
        if verbose >= 1:
            print(f'Epoch {epoch+1}: '
                  f'train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, '
                  f'train_acc={train_acc:.2%}, val_acc={val_acc:.2%}, '
                  f'time={epoch_time:.1f}s')

        if val_loss < best_val_loss - config.min_delta:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch + 1, 'encoder_state_dict': model.encoder.state_dict(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss, 'val_acc': val_acc, 'config': asdict(config),
            }, output_path / 'best_cpc_model.pt')
            if verbose >= 1:
                print(f'  -> New best model saved!')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                if verbose >= 1:
                    print(f'Early stopping at epoch {epoch+1}')
                break

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1, 'encoder_state_dict': model.encoder.state_dict(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss, 'config': asdict(config),
            }, output_path / f'checkpoint_epoch_{epoch+1}.pt')

    # Save final model
    torch.save({
        'epoch': config.total_epochs, 'encoder_state_dict': model.encoder.state_dict(),
        'model_state_dict': model.state_dict(), 'config': asdict(config),
    }, output_path / 'final_cpc_model.pt')

    total_time = time.time() - start_time
    results = {
        'best_epoch': best_epoch, 'best_val_loss': best_val_loss,
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1],
        'final_train_acc': history['train_acc'][-1],
        'final_val_acc': history['val_acc'][-1],
        'total_time_seconds': total_time, 'n_params': n_params,
        'config': asdict(config), 'history': history,
    }

    with open(output_path / 'cpc_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    if verbose >= 1:
        print(f'\nCPC Pretraining Complete!')
        print(f'  Best epoch: {best_epoch}')
        print(f'  Best val loss: {best_val_loss:.4f}')
        print(f'  Total time: {total_time/60:.1f} minutes')
        print(f'  Model saved to: {output_path}')

    return results
