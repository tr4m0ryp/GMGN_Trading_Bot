"""
ProbabilisticReturnHead: Multi-task return and drawdown prediction.

This module implements probabilistic prediction heads for:
1. Expected return (mean + variance) using Gaussian NLL
2. Expected drawdown (auxiliary task)

The variance output enables uncertainty quantification for
Kelly criterion position sizing.

Dependencies:
    torch

Date: 2025-12-25
"""

from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """
    Attention-weighted pooling over sequence dimension.

    Learns to weight different timesteps differently when
    aggregating sequence embeddings.

    Args:
        embed_dim: Embedding dimension

    Example:
        >>> pool = AttentionPooling(embed_dim=512)
        >>> z = torch.randn(32, 128, 512)
        >>> pooled = pool(z)  # [32, 512]
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.Tanh(),
            nn.Linear(embed_dim // 4, 1),
        )

    def forward(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pool sequence using learned attention weights.

        Args:
            z: Embeddings [batch, seq_len, embed_dim]
            mask: Boolean mask [batch, seq_len] where True = ignore

        Returns:
            Pooled embeddings [batch, embed_dim]
        """
        # Compute attention scores
        scores = self.attention(z).squeeze(-1)  # [batch, seq_len]

        # Mask padded positions
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        # Softmax over sequence
        weights = F.softmax(scores, dim=-1)  # [batch, seq_len]

        # Weighted sum
        pooled = torch.bmm(weights.unsqueeze(1), z).squeeze(1)  # [batch, embed_dim]

        return pooled


class ProbabilisticReturnHead(nn.Module):
    """
    Multi-task prediction head for return and drawdown.

    Outputs:
        - mu: Predicted mean return (unbounded)
        - log_var: Predicted log-variance (for uncertainty)
        - drawdown: Predicted drawdown (optional auxiliary task)

    Architecture:
        Encoder output -> Attention Pooling -> Shared MLP
            -> Mean head (1 output)
            -> Variance head (1 output)
            -> Drawdown head (1 output, optional)

    Args:
        embed_dim: Input embedding dimension. Default is 512.
        hidden_dims: Hidden layer dimensions. Default [256, 128].
        predict_drawdown: Whether to predict drawdown. Default True.
        dropout: Dropout probability. Default 0.1.

    Example:
        >>> head = ProbabilisticReturnHead(embed_dim=512)
        >>> z = torch.randn(32, 128, 512)
        >>> mu, log_var, dd = head(z)
    """

    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dims: list = None,
        predict_drawdown: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.embed_dim = embed_dim
        self.predict_drawdown = predict_drawdown

        # Attention pooling over sequence
        self.attention_pool = AttentionPooling(embed_dim)

        # Shared representation
        layers = []
        in_dim = embed_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        self.shared = nn.Sequential(*layers)
        final_dim = hidden_dims[-1]

        # Mean prediction head (unbounded output)
        self.mean_head = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.GELU(),
            nn.Linear(final_dim // 2, 1),
        )

        # Log-variance prediction head
        # Output log(sigma^2) for numerical stability
        self.var_head = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.GELU(),
            nn.Linear(final_dim // 2, 1),
        )

        # Drawdown prediction head (optional)
        if predict_drawdown:
            self.drawdown_head = nn.Sequential(
                nn.Linear(final_dim, final_dim // 2),
                nn.GELU(),
                nn.Linear(final_dim // 2, 1),
            )

        # Initialize output layers
        self._init_output_layers()

    def _init_output_layers(self):
        """Initialize output layer weights."""
        for head in [self.mean_head, self.var_head]:
            for layer in head:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    nn.init.zeros_(layer.bias)

        if self.predict_drawdown:
            for layer in self.drawdown_head:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    nn.init.zeros_(layer.bias)

    def forward(
        self,
        z: torch.Tensor,
        seq_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass for return prediction.

        Args:
            z: Encoder embeddings [batch, seq_len, embed_dim]
            seq_lengths: Actual sequence lengths [batch]

        Returns:
            mu: Predicted mean return [batch, 1]
            log_var: Predicted log-variance [batch, 1]
            drawdown: Predicted drawdown [batch, 1] (if enabled)
        """
        batch_size, seq_len, _ = z.shape

        # Create mask for padding
        mask = None
        if seq_lengths is not None:
            device = z.device
            range_tensor = torch.arange(seq_len, device=device)
            mask = range_tensor.unsqueeze(0) >= seq_lengths.unsqueeze(1)

        # Attention pooling
        pooled = self.attention_pool(z, mask)  # [batch, embed_dim]

        # Shared representation
        shared = self.shared(pooled)  # [batch, final_dim]

        # Predict mean and variance
        mu = self.mean_head(shared)  # [batch, 1]
        log_var = self.var_head(shared)  # [batch, 1]

        if self.predict_drawdown:
            drawdown = self.drawdown_head(shared)  # [batch, 1]
            return mu, log_var, drawdown

        return mu, log_var


def gaussian_nll_loss(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
    min_log_var: float = -6.0,
    max_log_var: float = 6.0,
    var_reg_weight: float = 0.1,
) -> torch.Tensor:
    """
    Gaussian negative log-likelihood loss with variance regularization.

    L = 0.5 * [log_var + (target - mu)^2 / exp(log_var)]

    Includes regularization to prevent variance collapse (overconfidence).

    Args:
        mu: Predicted mean [batch, 1]
        log_var: Predicted log-variance [batch, 1]
        target: Target values [batch] or [batch, 1]
        min_log_var: Minimum log-variance (floor). Default -6.0
        max_log_var: Maximum log-variance (cap). Default 6.0
        var_reg_weight: Regularization weight for variance floor. Default 0.1

    Returns:
        Scalar loss value
    """
    # Ensure target has correct shape
    if target.dim() == 1:
        target = target.unsqueeze(1)

    # Clamp log_var for numerical stability
    log_var_clamped = torch.clamp(log_var, min=min_log_var, max=max_log_var)

    # Gaussian NLL: 0.5 * [log(sigma^2) + (y - mu)^2 / sigma^2]
    var = torch.exp(log_var_clamped)
    nll = 0.5 * (log_var_clamped + (target - mu) ** 2 / var)

    # Regularization: penalize if variance is too low (overconfidence)
    var_penalty = var_reg_weight * F.relu(min_log_var - log_var)

    return nll.mean() + var_penalty.mean()


class MultiTaskLoss(nn.Module):
    """
    Combined loss for return and drawdown prediction.

    L_total = L_return + lambda_dd * L_drawdown

    Where L_return is Gaussian NLL and L_drawdown is MSE.

    Args:
        return_weight: Weight for return loss. Default 1.0
        drawdown_weight: Weight for drawdown loss. Default 0.3
        min_log_var: Minimum log-variance. Default -6.0
        var_reg_weight: Variance regularization weight. Default 0.1

    Example:
        >>> loss_fn = MultiTaskLoss(drawdown_weight=0.3)
        >>> loss, metrics = loss_fn(mu, log_var, dd_pred, return_target, dd_target)
    """

    def __init__(
        self,
        return_weight: float = 1.0,
        drawdown_weight: float = 0.3,
        min_log_var: float = -6.0,
        var_reg_weight: float = 0.1,
    ):
        super().__init__()
        self.return_weight = return_weight
        self.drawdown_weight = drawdown_weight
        self.min_log_var = min_log_var
        self.var_reg_weight = var_reg_weight

    def forward(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        drawdown_pred: Optional[torch.Tensor],
        return_target: torch.Tensor,
        drawdown_target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.

        Args:
            mu: Predicted mean return [batch, 1]
            log_var: Predicted log-variance [batch, 1]
            drawdown_pred: Predicted drawdown [batch, 1] (optional)
            return_target: Target return [batch]
            drawdown_target: Target drawdown [batch] (optional)

        Returns:
            total_loss: Combined scalar loss
            metrics: Dictionary with component losses
        """
        # Return loss (Gaussian NLL)
        loss_return = gaussian_nll_loss(
            mu, log_var, return_target,
            min_log_var=self.min_log_var,
            var_reg_weight=self.var_reg_weight,
        )

        total_loss = self.return_weight * loss_return
        metrics = {
            'loss_return': loss_return.item(),
        }

        # Drawdown loss (MSE)
        if drawdown_pred is not None and drawdown_target is not None:
            if drawdown_target.dim() == 1:
                drawdown_target = drawdown_target.unsqueeze(1)

            loss_drawdown = F.mse_loss(drawdown_pred, drawdown_target)
            total_loss = total_loss + self.drawdown_weight * loss_drawdown
            metrics['loss_drawdown'] = loss_drawdown.item()

        # Additional metrics
        with torch.no_grad():
            # Mean absolute error
            mae = torch.abs(mu.squeeze() - return_target).mean().item()
            metrics['mae'] = mae

            # Mean predicted variance
            mean_var = torch.exp(log_var).mean().item()
            metrics['mean_variance'] = mean_var

            # Correlation between predicted and actual
            if mu.numel() > 1:
                mu_flat = mu.squeeze()
                corr = torch.corrcoef(
                    torch.stack([mu_flat, return_target])
                )[0, 1].item()
                if not torch.isnan(torch.tensor(corr)):
                    metrics['correlation'] = corr

        metrics['loss_total'] = total_loss.item()
        return total_loss, metrics


class CalibrationMetrics:
    """
    Metrics for evaluating prediction calibration.

    A well-calibrated model should have:
    - Predictions falling within 1 sigma ~68% of time
    - Predictions falling within 2 sigma ~95% of time

    Example:
        >>> calib = CalibrationMetrics()
        >>> metrics = calib.compute(mu, log_var, targets)
    """

    @staticmethod
    def compute(
        mu: torch.Tensor,
        log_var: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute calibration metrics.

        Args:
            mu: Predicted means [N, 1] or [N]
            log_var: Predicted log-variances [N, 1] or [N]
            target: Actual values [N]

        Returns:
            Dictionary with calibration metrics
        """
        mu = mu.squeeze()
        log_var = log_var.squeeze()

        sigma = torch.exp(0.5 * log_var)
        z_scores = torch.abs((target - mu) / sigma)

        # Fraction within k standard deviations
        within_1_sigma = (z_scores <= 1).float().mean().item()
        within_2_sigma = (z_scores <= 2).float().mean().item()
        within_3_sigma = (z_scores <= 3).float().mean().item()

        # Expected: 68.27%, 95.45%, 99.73%
        # Calibration error
        cal_error_1 = abs(within_1_sigma - 0.6827)
        cal_error_2 = abs(within_2_sigma - 0.9545)

        return {
            'within_1_sigma': within_1_sigma,
            'within_2_sigma': within_2_sigma,
            'within_3_sigma': within_3_sigma,
            'calibration_error': (cal_error_1 + cal_error_2) / 2,
            'mean_z_score': z_scores.mean().item(),
        }
