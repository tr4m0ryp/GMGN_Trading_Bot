"""
Regression Trainer Utilities: validation, checkpointing, and model loading.

Extracted from regression_trainer.py to keep files under 300 lines.

Dependencies:
    torch, numpy

Date: 2025-12-25
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cpc_regression.encoder import CPCEncoder
from cpc_regression.cpc_model import CPCRegressionModel
from cpc_regression.return_head import (
    ProbabilisticReturnHead, CalibrationMetrics
)
from cpc_regression.config import CPCConfig, RegressionConfig


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
    params = list(model.encoder.named_parameters())
    for name, param in params:
        if 'output_proj' in name or 'ffn' in name or 'attn' in name:
            param.requires_grad = True


def validate_epoch(model, val_loader, loss_fn, reg_config, device):
    """
    Run validation epoch and return metrics and collected tensors.

    Args:
        model: Model to validate.
        val_loader: Validation data loader.
        loss_fn: Loss function.
        reg_config: Regression configuration.
        device: Compute device.

    Returns:
        Tuple of (val_loss, val_mae, all_mus, all_log_vars, all_targets).
    """
    model.eval()
    val_loss, val_mae, n_val_batches = 0.0, 0.0, 0
    all_mus, all_log_vars, all_targets = [], [], []

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

            loss, metrics = loss_fn(mu, log_var, dd_pred, return_target, drawdown_target)
            val_loss += loss.item()
            val_mae += metrics['mae']
            n_val_batches += 1
            all_mus.append(mu.cpu())
            all_log_vars.append(log_var.cpu())
            all_targets.append(return_target.cpu())

    val_loss /= n_val_batches
    val_mae /= n_val_batches
    return val_loss, val_mae, torch.cat(all_mus), torch.cat(all_log_vars), torch.cat(all_targets)


def save_checkpoint(model, optimizer, epoch, val_loss, val_mae,
                    corr, cal_metrics, cpc_config, reg_config, output_path, tag):
    """
    Save model checkpoint.

    Args:
        model: Model to save.
        optimizer: Optimizer (can be None for final checkpoint).
        epoch: Current epoch number.
        val_loss: Validation loss (can be None).
        val_mae: Validation MAE (can be None).
        corr: Correlation metric (can be None).
        cal_metrics: Calibration metrics dict (can be None).
        cpc_config: CPC configuration.
        reg_config: Regression configuration.
        output_path: Directory to save to.
        tag: Checkpoint tag (e.g. 'best', 'final').
    """
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'encoder_state_dict': model.encoder.state_dict(),
        'return_head_state_dict': model.return_head.state_dict(),
        'cpc_config': asdict(cpc_config),
        'reg_config': asdict(reg_config),
    }
    if optimizer is not None:
        state['optimizer_state_dict'] = optimizer.state_dict()
    if val_loss is not None:
        state['val_loss'] = val_loss
        state['val_mae'] = val_mae
        state['correlation'] = corr
        state['calibration'] = cal_metrics
    torch.save(state, output_path / f'{tag}_regression_model.pt')


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

    encoder = CPCEncoder(
        input_dim=cpc_config.input_dim, hidden_dim=cpc_config.hidden_dim,
        embed_dim=cpc_config.embed_dim, lstm_layers=cpc_config.lstm_layers,
        n_heads=cpc_config.n_heads, ff_dim=cpc_config.ff_dim,
        dropout=cpc_config.dropout,
    )

    return_head = ProbabilisticReturnHead(
        embed_dim=cpc_config.embed_dim, hidden_dims=reg_config.hidden_dims,
        predict_drawdown=reg_config.predict_drawdown, dropout=0.0,
    )

    model = CPCRegressionModel(encoder, return_head)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded regression model from {model_path}")
    print(f"  Val loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    print(f"  Correlation: {checkpoint.get('correlation', 'N/A'):.3f}")

    return model, {'cpc': cpc_config, 'regression': reg_config}
