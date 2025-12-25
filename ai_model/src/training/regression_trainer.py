"""
Regression Trainer: Phase 2 supervised fine-tuning.

Fine-tunes the pretrained CPC encoder with a probabilistic
return prediction head using Gaussian NLL loss.

Multi-task learning: predicts both return and drawdown.

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
from cpc_regression.cpc_model import CPCModel, CPCRegressionModel
from cpc_regression.return_head import (
    ProbabilisticReturnHead, MultiTaskLoss, CalibrationMetrics
)
from cpc_regression.config import (
    CPCConfig, RegressionConfig, get_config_for_gpu
)
from training.cpc_trainer import load_pretrained_encoder


class RegressionDataset(Dataset):
    """
    Dataset for return regression training.

    Uses both features and return/drawdown targets.

    Args:
        samples: List of sample dictionaries
        max_seq_len: Maximum sequence length. Default 128.
    """

    def __init__(
        self,
        samples: List[Dict],
        max_seq_len: int = 128,
    ):
        self.samples = samples
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        features = sample['features']  # [seq_len, 14]

        # Truncate if needed
        if features.shape[0] > self.max_seq_len:
            features = features[-self.max_seq_len:]
            features = features.copy()
            features[:, 0] = features[:, 0] - features[0, 0]

        seq_len = features.shape[0]

        return {
            'features': torch.FloatTensor(features),
            'seq_length': seq_len,
            'return_target': torch.FloatTensor([sample['potential_profit_pct']]),
            'drawdown_target': torch.FloatTensor([sample['drawdown_pct']]),
        }


def collate_regression(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for regression training.

    Args:
        batch: List of sample dictionaries

    Returns:
        Batched tensors with padding
    """
    max_len = max(s['seq_length'] for s in batch)
    batch_size = len(batch)
    feature_dim = batch[0]['features'].shape[-1]

    features = torch.zeros(batch_size, max_len, feature_dim)
    seq_lengths = torch.zeros(batch_size, dtype=torch.long)
    return_targets = torch.zeros(batch_size)
    drawdown_targets = torch.zeros(batch_size)

    for i, sample in enumerate(batch):
        seq_len = sample['seq_length']
        features[i, :seq_len] = sample['features']
        seq_lengths[i] = seq_len
        return_targets[i] = sample['return_target']
        drawdown_targets[i] = sample['drawdown_target']

    return {
        'features': features,
        'seq_lengths': seq_lengths,
        'return_target': return_targets,
        'drawdown_target': drawdown_targets,
    }


class BalancedReturnSampler(torch.utils.data.Sampler):
    """
    Sampler that balances samples by return distribution.

    Addresses the class imbalance in return values
    by oversampling rare return ranges.

    Args:
        samples: List of sample dictionaries
        n_bins: Number of bins for return distribution
    """

    def __init__(self, samples: List[Dict], n_bins: int = 10):
        self.samples = samples
        returns = [s['potential_profit_pct'] for s in samples]

        # Create bins
        self.bins = np.histogram_bin_edges(returns, bins=n_bins)
        self.bin_indices = [[] for _ in range(n_bins)]

        for idx, r in enumerate(returns):
            bin_idx = np.digitize(r, self.bins) - 1
            bin_idx = min(bin_idx, n_bins - 1)
            self.bin_indices[bin_idx].append(idx)

        # Calculate sampling weights
        non_empty_bins = [b for b in self.bin_indices if len(b) > 0]
        self.n_samples = len(samples)

    def __iter__(self):
        # Sample equally from each non-empty bin
        indices = []
        samples_per_bin = self.n_samples // len(self.bin_indices)

        for bin_idx in self.bin_indices:
            if len(bin_idx) > 0:
                sampled = np.random.choice(
                    bin_idx,
                    size=min(samples_per_bin, len(bin_idx) * 2),
                    replace=True,
                )
                indices.extend(sampled.tolist())

        np.random.shuffle(indices)
        return iter(indices[:self.n_samples])

    def __len__(self) -> int:
        return self.n_samples


def freeze_encoder(model: nn.Module):
    """Freeze encoder parameters."""
    for param in model.encoder.parameters():
        param.requires_grad = False


def unfreeze_encoder(model: nn.Module):
    """Unfreeze encoder parameters."""
    for param in model.encoder.parameters():
        param.requires_grad = True


def unfreeze_top_layers(model: nn.Module, n_layers: int = 1):
    """Unfreeze top n layers of encoder (output proj, ffn, attention)."""
    # Get all named parameters
    params = list(model.encoder.named_parameters())

    # Unfreeze output projection
    for name, param in params:
        if 'output_proj' in name or 'ffn' in name or 'attn' in name:
            param.requires_grad = True


def train_regression(
    pretrained_encoder_path: str,
    data_dir: str,
    output_dir: str,
    cpc_config: Optional[CPCConfig] = None,
    reg_config: Optional[RegressionConfig] = None,
    device: str = 'cuda',
    verbose: int = 1,
) -> Dict:
    """
    Train return regression model (Phase 2: Supervised fine-tuning).

    Args:
        pretrained_encoder_path: Path to pretrained CPC encoder
        data_dir: Directory containing preprocessed data
        output_dir: Directory to save trained model
        cpc_config: CPCConfig (if None, loaded from checkpoint)
        reg_config: RegressionConfig (if None, auto-configure)
        device: Device to train on
        verbose: Verbosity level

    Returns:
        Dictionary with training results
    """
    start_time = time.time()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load pretrained encoder
    encoder, loaded_cpc_config = load_pretrained_encoder(
        pretrained_encoder_path, device
    )
    if cpc_config is None:
        cpc_config = loaded_cpc_config

    # Auto-configure regression if needed
    if reg_config is None:
        gpu_config = get_config_for_gpu()
        reg_config = RegressionConfig(**gpu_config['regression'])

    if verbose >= 1:
        print(f"Regression Training Configuration:")
        print(f"  Batch size: {reg_config.batch_size}")
        print(f"  Learning rate: {reg_config.learning_rate}")
        print(f"  Encoder LR mult: {reg_config.encoder_lr_mult}")
        print(f"  Freeze epochs: {reg_config.freeze_encoder_epochs}")
        print(f"  Total epochs: {reg_config.total_epochs}")
        print(f"  Predict drawdown: {reg_config.predict_drawdown}")

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
    train_dataset = RegressionDataset(train_samples, max_seq_len=cpc_config.max_seq_len)
    val_dataset = RegressionDataset(val_samples, max_seq_len=cpc_config.max_seq_len)

    # Create sampler for balanced training
    train_sampler = BalancedReturnSampler(train_samples)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=reg_config.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_regression,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=reg_config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_regression,
    )

    # Create return head
    return_head = ProbabilisticReturnHead(
        embed_dim=cpc_config.embed_dim,
        hidden_dims=reg_config.hidden_dims,
        predict_drawdown=reg_config.predict_drawdown,
        dropout=reg_config.dropout,
    )

    # Create combined model
    model = CPCRegressionModel(encoder, return_head)
    model = model.to(device)

    # Initially freeze encoder
    freeze_encoder(model)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose >= 1:
        print(f"Total parameters: {n_params:,}")
        print(f"Trainable parameters: {n_trainable:,}")

    # Create optimizer with different LRs for encoder and head
    optimizer = AdamW([
        {'params': model.encoder.parameters(), 'lr': reg_config.learning_rate * reg_config.encoder_lr_mult},
        {'params': model.return_head.parameters(), 'lr': reg_config.learning_rate},
    ], weight_decay=reg_config.weight_decay)

    # Create loss function
    loss_fn = MultiTaskLoss(
        return_weight=1.0,
        drawdown_weight=reg_config.drawdown_weight,
        min_log_var=reg_config.min_log_var,
        var_reg_weight=reg_config.var_reg_weight,
    )

    # Create scheduler
    steps_per_epoch = len(train_loader)
    total_steps = reg_config.total_epochs * steps_per_epoch
    warmup_steps = reg_config.warmup_epochs * steps_per_epoch

    warmup = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
    )
    cosine = CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=1e-7
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
    )

    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mae': [],
        'val_mae': [],
        'val_correlation': [],
        'calibration_error': [],
    }

    for epoch in range(reg_config.total_epochs):
        epoch_start = time.time()

        # Progressive unfreezing
        if epoch == reg_config.freeze_encoder_epochs:
            if verbose >= 1:
                print(f"Epoch {epoch+1}: Unfreezing encoder")
            unfreeze_encoder(model)
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if verbose >= 1:
                print(f"  Trainable parameters: {n_trainable:,}")

        # Training
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{reg_config.total_epochs}',
                    disable=verbose < 1)

        for batch in pbar:
            features = batch['features'].to(device)
            seq_lengths = batch['seq_lengths'].to(device)
            return_target = batch['return_target'].to(device)
            drawdown_target = batch['drawdown_target'].to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(features, seq_lengths)
            if reg_config.predict_drawdown:
                mu, log_var, dd_pred = outputs
            else:
                mu, log_var = outputs
                dd_pred = None

            # Compute loss
            loss, metrics = loss_fn(
                mu, log_var, dd_pred, return_target, drawdown_target
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), reg_config.grad_clip)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_mae += metrics['mae']
            n_batches += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mae': f'{metrics["mae"]:.4f}',
            })

        train_loss /= n_batches
        train_mae /= n_batches

        # Validation
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        all_mus = []
        all_log_vars = []
        all_targets = []
        n_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                seq_lengths = batch['seq_lengths'].to(device)
                return_target = batch['return_target'].to(device)
                drawdown_target = batch['drawdown_target'].to(device)

                outputs = model(features, seq_lengths)
                if reg_config.predict_drawdown:
                    mu, log_var, dd_pred = outputs
                else:
                    mu, log_var = outputs
                    dd_pred = None

                loss, metrics = loss_fn(
                    mu, log_var, dd_pred, return_target, drawdown_target
                )

                val_loss += loss.item()
                val_mae += metrics['mae']
                n_val_batches += 1

                all_mus.append(mu.cpu())
                all_log_vars.append(log_var.cpu())
                all_targets.append(return_target.cpu())

        val_loss /= n_val_batches
        val_mae /= n_val_batches

        # Compute calibration metrics
        all_mus = torch.cat(all_mus)
        all_log_vars = torch.cat(all_log_vars)
        all_targets = torch.cat(all_targets)
        cal_metrics = CalibrationMetrics.compute(all_mus, all_log_vars, all_targets)

        # Compute correlation
        corr = torch.corrcoef(
            torch.stack([all_mus.squeeze(), all_targets])
        )[0, 1].item()
        if np.isnan(corr):
            corr = 0.0

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        history['val_correlation'].append(corr)
        history['calibration_error'].append(cal_metrics['calibration_error'])

        epoch_time = time.time() - epoch_start

        if verbose >= 1:
            print(f'Epoch {epoch+1}: '
                  f'train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, '
                  f'val_mae={val_mae:.4f}, corr={corr:.3f}, '
                  f'calib_err={cal_metrics["calibration_error"]:.3f}, '
                  f'time={epoch_time:.1f}s')

        # Save best model
        if val_loss < best_val_loss - reg_config.min_delta:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'encoder_state_dict': model.encoder.state_dict(),
                'return_head_state_dict': model.return_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
                'correlation': corr,
                'calibration': cal_metrics,
                'cpc_config': asdict(cpc_config),
                'reg_config': asdict(reg_config),
            }, output_path / 'best_regression_model.pt')

            if verbose >= 1:
                print(f'  -> New best model saved!')
        else:
            patience_counter += 1
            if patience_counter >= reg_config.patience:
                if verbose >= 1:
                    print(f'Early stopping at epoch {epoch+1}')
                break

    # Save final model
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'encoder_state_dict': model.encoder.state_dict(),
        'return_head_state_dict': model.return_head.state_dict(),
        'cpc_config': asdict(cpc_config),
        'reg_config': asdict(reg_config),
    }, output_path / 'final_regression_model.pt')

    total_time = time.time() - start_time

    # Save results
    results = {
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1],
        'final_val_mae': history['val_mae'][-1],
        'final_correlation': history['val_correlation'][-1],
        'final_calibration_error': history['calibration_error'][-1],
        'total_time_seconds': total_time,
        'n_params': n_params,
        'history': history,
    }

    with open(output_path / 'regression_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    if verbose >= 1:
        print(f'\nRegression Training Complete!')
        print(f'  Best epoch: {best_epoch}')
        print(f'  Best val loss: {best_val_loss:.4f}')
        print(f'  Final MAE: {history["val_mae"][-1]:.4f}')
        print(f'  Final correlation: {history["val_correlation"][-1]:.3f}')
        print(f'  Total time: {total_time/60:.1f} minutes')

    return results


def load_regression_model(
    model_path: str,
    device: str = 'cuda',
) -> Tuple[CPCRegressionModel, Dict]:
    """
    Load trained regression model for inference.

    Args:
        model_path: Path to saved model
        device: Device to load on

    Returns:
        Tuple of (model, configs)
    """
    checkpoint = torch.load(model_path, map_location=device)

    cpc_config = CPCConfig(**checkpoint['cpc_config'])
    reg_config = RegressionConfig(**checkpoint['reg_config'])

    # Create encoder
    encoder = CPCEncoder(
        input_dim=cpc_config.input_dim,
        hidden_dim=cpc_config.hidden_dim,
        embed_dim=cpc_config.embed_dim,
        lstm_layers=cpc_config.lstm_layers,
        n_heads=cpc_config.n_heads,
        ff_dim=cpc_config.ff_dim,
        dropout=cpc_config.dropout,
    )

    # Create return head
    return_head = ProbabilisticReturnHead(
        embed_dim=cpc_config.embed_dim,
        hidden_dims=reg_config.hidden_dims,
        predict_drawdown=reg_config.predict_drawdown,
        dropout=0.0,  # No dropout for inference
    )

    # Create model and load weights
    model = CPCRegressionModel(encoder, return_head)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded regression model from {model_path}")
    print(f"  Val loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    print(f"  Correlation: {checkpoint.get('correlation', 'N/A'):.3f}")

    return model, {'cpc': cpc_config, 'regression': reg_config}
