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
from torch.utils.data import DataLoader, Dataset
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


class CPCDataset(Dataset):
    """
    Dataset for CPC pretraining.

    Only uses features, ignores labels (self-supervised).

    Args:
        samples: List of sample dictionaries with 'features' key
        max_seq_len: Maximum sequence length. Default 128.
        min_seq_len: Minimum sequence length. Default 20.
    """

    def __init__(
        self,
        samples: List[Dict],
        max_seq_len: int = 128,
        min_seq_len: int = 20,
    ):
        # Filter samples by sequence length
        self.samples = [
            s for s in samples
            if s['features'].shape[0] >= min_seq_len
        ]
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len

        print(f"CPCDataset: {len(self.samples)} samples "
              f"(filtered from {len(samples)}, min_len={min_seq_len})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        features = sample['features']  # [seq_len, 14]

        # Truncate if needed (keep last max_seq_len)
        if features.shape[0] > self.max_seq_len:
            features = features[-self.max_seq_len:]
            # Re-normalize log_close (index 0) after truncation
            features = features.copy()
            features[:, 0] = features[:, 0] - features[0, 0]

        seq_len = features.shape[0]

        return {
            'features': torch.FloatTensor(features),
            'seq_length': seq_len,
        }


def collate_cpc(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for CPC training.

    Pads sequences to batch maximum length.

    Args:
        batch: List of sample dictionaries

    Returns:
        Batched tensors with padding
    """
    # Get max sequence length in batch
    max_len = max(s['seq_length'] for s in batch)
    batch_size = len(batch)
    feature_dim = batch[0]['features'].shape[-1]

    # Pad features
    features = torch.zeros(batch_size, max_len, feature_dim)
    seq_lengths = torch.zeros(batch_size, dtype=torch.long)

    for i, sample in enumerate(batch):
        seq_len = sample['seq_length']
        features[i, :seq_len] = sample['features']
        seq_lengths[i] = seq_len

    return {
        'features': features,
        'seq_lengths': seq_lengths,
    }


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
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=config.min_lr,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )
    elif config.scheduler == 'linear':
        scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=config.min_lr / config.learning_rate,
            total_iters=total_steps,
        )
    else:
        # Constant LR (with warmup)
        scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
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

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Auto-configure if needed
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

    # Create datasets
    train_dataset = CPCDataset(
        train_samples,
        max_seq_len=config.max_seq_len,
        min_seq_len=config.min_seq_len,
    )
    val_dataset = CPCDataset(
        val_samples,
        max_seq_len=config.max_seq_len,
        min_seq_len=config.min_seq_len,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_cpc,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_cpc,
    )

    # Create model
    encoder = CPCEncoder(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        embed_dim=config.embed_dim,
        lstm_layers=config.lstm_layers,
        n_heads=config.n_heads,
        ff_dim=config.ff_dim,
        dropout=config.dropout,
    )

    model = CPCModel(
        encoder=encoder,
        ar_hidden=config.ar_hidden,
        prediction_steps=config.prediction_steps,
        temperature=config.temperature,
    )

    model = model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    if verbose >= 1:
        print(f"Model parameters: {n_params:,}")

    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Create scheduler
    steps_per_epoch = len(train_loader)
    scheduler = create_scheduler(optimizer, config, steps_per_epoch)

    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'lr': [],
    }

    for epoch in range(config.total_epochs):
        epoch_start = time.time()

        # Training
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.total_epochs}',
                    disable=verbose < 1)

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

        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        n_val_batches = 0

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

        # Record history
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

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1

            torch.save({
                'epoch': epoch + 1,
                'encoder_state_dict': model.encoder.state_dict(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'config': asdict(config),
            }, output_path / 'best_cpc_model.pt')

            if verbose >= 1:
                print(f'  -> New best model saved!')

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'encoder_state_dict': model.encoder.state_dict(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': asdict(config),
            }, output_path / f'checkpoint_epoch_{epoch+1}.pt')

    # Save final model
    torch.save({
        'epoch': config.total_epochs,
        'encoder_state_dict': model.encoder.state_dict(),
        'model_state_dict': model.state_dict(),
        'config': asdict(config),
    }, output_path / 'final_cpc_model.pt')

    total_time = time.time() - start_time

    # Save training results
    results = {
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1],
        'final_train_acc': history['train_acc'][-1],
        'final_val_acc': history['val_acc'][-1],
        'total_time_seconds': total_time,
        'n_params': n_params,
        'config': asdict(config),
        'history': history,
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


def load_pretrained_encoder(
    model_path: str,
    device: str = 'cuda',
) -> Tuple[CPCEncoder, CPCConfig]:
    """
    Load pretrained encoder for Phase 2 fine-tuning.

    Args:
        model_path: Path to saved CPC model
        device: Device to load model on

    Returns:
        Tuple of (encoder, config)
    """
    checkpoint = torch.load(model_path, map_location=device)
    config_dict = checkpoint['config']
    config = CPCConfig(**config_dict)

    encoder = CPCEncoder(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        embed_dim=config.embed_dim,
        lstm_layers=config.lstm_layers,
        n_heads=config.n_heads,
        ff_dim=config.ff_dim,
        dropout=config.dropout,
    )

    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder = encoder.to(device)

    print(f"Loaded pretrained encoder from {model_path}")
    print(f"  Embed dim: {config.embed_dim}")
    print(f"  Hidden dim: {config.hidden_dim}")

    return encoder, config
