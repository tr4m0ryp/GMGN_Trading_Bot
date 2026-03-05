"""
Return Prediction Layers: Loss functions and calibration metrics.

Contains the Gaussian NLL loss function, MultiTaskLoss, and
CalibrationMetrics for the return prediction head.

Dependencies:
    torch

Date: 2025-12-25
"""

from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    if target.dim() == 1:
        target = target.unsqueeze(1)

    log_var_clamped = torch.clamp(log_var, min=min_log_var, max=max_log_var)
    var = torch.exp(log_var_clamped)
    nll = 0.5 * (log_var_clamped + (target - mu) ** 2 / var)
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
        loss_return = gaussian_nll_loss(
            mu, log_var, return_target,
            min_log_var=self.min_log_var,
            var_reg_weight=self.var_reg_weight,
        )

        total_loss = self.return_weight * loss_return
        metrics = {'loss_return': loss_return.item()}

        if drawdown_pred is not None and drawdown_target is not None:
            if drawdown_target.dim() == 1:
                drawdown_target = drawdown_target.unsqueeze(1)
            loss_drawdown = F.mse_loss(drawdown_pred, drawdown_target)
            total_loss = total_loss + self.drawdown_weight * loss_drawdown
            metrics['loss_drawdown'] = loss_drawdown.item()

        with torch.no_grad():
            mae = torch.abs(mu.squeeze() - return_target).mean().item()
            metrics['mae'] = mae
            mean_var = torch.exp(log_var).mean().item()
            metrics['mean_variance'] = mean_var

            if mu.numel() > 1:
                mu_flat = mu.squeeze()
                corr = torch.corrcoef(torch.stack([mu_flat, return_target]))[0, 1].item()
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

        within_1_sigma = (z_scores <= 1).float().mean().item()
        within_2_sigma = (z_scores <= 2).float().mean().item()
        within_3_sigma = (z_scores <= 3).float().mean().item()

        cal_error_1 = abs(within_1_sigma - 0.6827)
        cal_error_2 = abs(within_2_sigma - 0.9545)

        return {
            'within_1_sigma': within_1_sigma,
            'within_2_sigma': within_2_sigma,
            'within_3_sigma': within_3_sigma,
            'calibration_error': (cal_error_1 + cal_error_2) / 2,
            'mean_z_score': z_scores.mean().item(),
        }
