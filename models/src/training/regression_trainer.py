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
from torch.utils.data import DataLoader
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
from training.cpc_utils import load_pretrained_encoder
from training.regression_dataset import (
    RegressionDataset, collate_regression, BalancedReturnSampler
)
from training.regression_utils import (
    validate_epoch, save_checkpoint,
    load_regression_model,  # noqa: F401 (re-export)
    freeze_encoder, unfreeze_encoder, unfreeze_top_layers,  # noqa: F401
)


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
    train_sampler = BalancedReturnSampler(train_samples)

    train_loader = DataLoader(
        train_dataset, batch_size=reg_config.batch_size,
        sampler=train_sampler, num_workers=4, pin_memory=True,
        collate_fn=collate_regression,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=reg_config.batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
        collate_fn=collate_regression,
    )

    # Create model
    return_head = ProbabilisticReturnHead(
        embed_dim=cpc_config.embed_dim,
        hidden_dims=reg_config.hidden_dims,
        predict_drawdown=reg_config.predict_drawdown,
        dropout=reg_config.dropout,
    )
    model = CPCRegressionModel(encoder, return_head)
    model = model.to(device)
    freeze_encoder(model)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose >= 1:
        print(f"Total parameters: {n_params:,}")
        print(f"Trainable parameters: {n_trainable:,}")

    # Optimizer, loss, scheduler
    optimizer = AdamW([
        {'params': model.encoder.parameters(), 'lr': reg_config.learning_rate * reg_config.encoder_lr_mult},
        {'params': model.return_head.parameters(), 'lr': reg_config.learning_rate},
    ], weight_decay=reg_config.weight_decay)

    loss_fn = MultiTaskLoss(
        return_weight=1.0, drawdown_weight=reg_config.drawdown_weight,
        min_log_var=reg_config.min_log_var, var_reg_weight=reg_config.var_reg_weight,
    )

    steps_per_epoch = len(train_loader)
    total_steps = reg_config.total_epochs * steps_per_epoch
    warmup_steps = reg_config.warmup_epochs * steps_per_epoch
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-7)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    history = {
        'train_loss': [], 'val_loss': [], 'train_mae': [],
        'val_mae': [], 'val_correlation': [], 'calibration_error': [],
    }

    for epoch in range(reg_config.total_epochs):
        epoch_start = time.time()

        if epoch == reg_config.freeze_encoder_epochs:
            if verbose >= 1:
                print(f"Epoch {epoch+1}: Unfreezing encoder")
            unfreeze_encoder(model)
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if verbose >= 1:
                print(f"  Trainable parameters: {n_trainable:,}")

        # Train
        model.train()
        train_loss, train_mae, n_batches = 0.0, 0.0, 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{reg_config.total_epochs}',
                    disable=verbose < 1)

        for batch in pbar:
            features = batch['features'].to(device)
            seq_lengths = batch['seq_lengths'].to(device)
            return_target = batch['return_target'].to(device)
            drawdown_target = batch['drawdown_target'].to(device)

            optimizer.zero_grad()
            outputs = model(features, seq_lengths)
            if reg_config.predict_drawdown:
                mu, log_var, dd_pred = outputs
            else:
                mu, log_var = outputs
                dd_pred = None

            loss, metrics = loss_fn(mu, log_var, dd_pred, return_target, drawdown_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), reg_config.grad_clip)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_mae += metrics['mae']
            n_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'mae': f'{metrics["mae"]:.4f}'})

        train_loss /= n_batches
        train_mae /= n_batches

        # Validate
        val_loss, val_mae, all_mus, all_log_vars, all_targets = validate_epoch(
            model, val_loader, loss_fn, reg_config, device
        )

        # Calibration and correlation
        cal_metrics = CalibrationMetrics.compute(all_mus, all_log_vars, all_targets)
        corr = torch.corrcoef(torch.stack([all_mus.squeeze(), all_targets]))[0, 1].item()
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
            save_checkpoint(model, optimizer, epoch + 1, val_loss, val_mae,
                           corr, cal_metrics, cpc_config, reg_config, output_path, 'best')
            if verbose >= 1:
                print(f'  -> New best model saved!')
        else:
            patience_counter += 1
            if patience_counter >= reg_config.patience:
                if verbose >= 1:
                    print(f'Early stopping at epoch {epoch+1}')
                break

    # Save final model
    save_checkpoint(model, None, epoch + 1, None, None, None, None,
                   cpc_config, reg_config, output_path, 'final')

    total_time = time.time() - start_time
    results = {
        'best_epoch': best_epoch, 'best_val_loss': best_val_loss,
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1],
        'final_val_mae': history['val_mae'][-1],
        'final_correlation': history['val_correlation'][-1],
        'final_calibration_error': history['calibration_error'][-1],
        'total_time_seconds': total_time, 'n_params': n_params, 'history': history,
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
