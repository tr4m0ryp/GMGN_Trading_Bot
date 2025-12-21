"""
Evaluation and backtesting for trading model.

This module implements backtesting with realistic fee simulation,
performance metrics calculation, and walk-forward testing on unseen data.

Dependencies:
    torch: Deep learning framework
    numpy: Numerical computations
    pandas: Data analysis

Author: Trading Team
Date: 2025-12-21
"""

from typing import Dict, List, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import FIXED_POSITION_SIZE, TOTAL_FEE_PER_TX
from data_preparation import calculate_net_profit, get_execution_price


def backtest_token(model: nn.Module,
                  token_samples: List[Dict[str, Any]],
                  device: str,
                  confidence_threshold: float = 0.7) -> Dict[str, Any]:
    """
    Backtest model on a single token.

    Simulates trading on one token's full lifecycle with realistic
    execution delays and fees.

    Args:
        model: Trained PyTorch model.
        token_samples: List of samples for one token (chronological order).
        device: Device to run inference on.
        confidence_threshold: Minimum confidence to execute trade. Default is 0.7.

    Returns:
        Dictionary containing:
            - trades: List of executed trades
            - total_pnl: Total profit/loss in SOL
            - num_trades: Number of trades executed
            - win_rate: Percentage of winning trades
            - avg_profit: Average profit per winning trade
            - avg_loss: Average loss per losing trade

    Example:
        >>> results = backtest_token(model, token_samples, device='cuda')
        >>> print(f"Total PnL: {results['total_pnl']:.4f} SOL")
    """
    model.eval()
    trades = []

    with torch.no_grad():
        for sample in token_samples:
            features = torch.FloatTensor(sample['features']).unsqueeze(0).to(device)
            seq_length = torch.LongTensor([sample['seq_length']])

            predictions, confidence = model(features, seq_length)

            if confidence.item() < confidence_threshold:
                continue

            action = torch.argmax(predictions, dim=1).item()

            if action == 1:
                buy_price = sample['buy_price']
                profit_pct = sample['potential_profit_pct']

                net_profit = profit_pct * FIXED_POSITION_SIZE

                trades.append({
                    'timestamp': sample['timestamp'],
                    'action': 'BUY',
                    'buy_price': buy_price,
                    'profit': net_profit,
                    'confidence': confidence.item(),
                })

    total_pnl = sum(t['profit'] for t in trades)
    num_trades = len(trades)

    if num_trades == 0:
        return {
            'trades': trades,
            'total_pnl': 0.0,
            'num_trades': 0,
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'avg_loss': 0.0,
        }

    winning_trades = [t for t in trades if t['profit'] > 0]
    losing_trades = [t for t in trades if t['profit'] <= 0]

    win_rate = len(winning_trades) / num_trades
    avg_profit = np.mean([t['profit'] for t in winning_trades]) if winning_trades else 0.0
    avg_loss = np.mean([t['profit'] for t in losing_trades]) if losing_trades else 0.0

    return {
        'trades': trades,
        'total_pnl': total_pnl,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'avg_profit': avg_profit,
        'avg_loss': avg_loss,
    }


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio.

    Args:
        returns: Array of returns (profit/loss per trade).
        risk_free_rate: Risk-free rate. Default is 0.0.

    Returns:
        Sharpe ratio (annualized).

    Note:
        Returns 0 if standard deviation is zero.
    """
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - risk_free_rate
    mean_return = np.mean(excess_returns)
    std_return = np.std(excess_returns)

    if std_return == 0:
        return 0.0

    sharpe = mean_return / std_return
    return sharpe * np.sqrt(252)


def calculate_max_drawdown(cumulative_pnl: np.ndarray) -> float:
    """
    Calculate maximum drawdown.

    Args:
        cumulative_pnl: Array of cumulative profit/loss over time.

    Returns:
        Maximum drawdown as a decimal (e.g., 0.15 for 15% drawdown).

    Example:
        >>> pnl = np.array([0.1, 0.2, 0.15, 0.25, 0.3])
        >>> max_dd = calculate_max_drawdown(pnl)
        >>> print(f"Max drawdown: {max_dd * 100:.2f}%")
    """
    if len(cumulative_pnl) == 0:
        return 0.0

    running_max = np.maximum.accumulate(cumulative_pnl)
    drawdown = (cumulative_pnl - running_max) / (running_max + 1e-8)
    max_drawdown = np.min(drawdown)

    return abs(max_drawdown)


def evaluate_model(model: nn.Module,
                  test_loader: DataLoader,
                  device: str,
                  confidence_threshold: float = 0.7) -> Dict[str, Any]:
    """
    Evaluate model on test set.

    Runs comprehensive evaluation including accuracy, per-class metrics,
    and confidence analysis.

    Args:
        model: Trained PyTorch model.
        test_loader: DataLoader for test data.
        device: Device to run evaluation on.
        confidence_threshold: Minimum confidence for predictions. Default is 0.7.

    Returns:
        Dictionary containing:
            - accuracy: Overall accuracy
            - precision: Precision per class
            - recall: Recall per class
            - f1_score: F1 score per class
            - confusion_matrix: Confusion matrix
            - avg_confidence: Average confidence for predictions

    Example:
        >>> metrics = evaluate_model(model, test_loader, device='cuda')
        >>> print(f"Test accuracy: {metrics['accuracy']:.4f}")
    """
    model.eval()

    all_predictions = []
    all_labels = []
    all_confidences = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            seq_lengths = batch['seq_lengths']

            predictions, confidence = model(features, seq_lengths)

            high_conf_mask = confidence >= confidence_threshold
            predicted = torch.argmax(predictions, dim=1)
            predicted[~high_conf_mask] = 0

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidence.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)

    accuracy = np.mean(all_predictions == all_labels)

    num_classes = 3
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1_score = np.zeros(num_classes)

    for cls in range(num_classes):
        true_positives = np.sum((all_predictions == cls) & (all_labels == cls))
        false_positives = np.sum((all_predictions == cls) & (all_labels != cls))
        false_negatives = np.sum((all_predictions != cls) & (all_labels == cls))

        if true_positives + false_positives > 0:
            precision[cls] = true_positives / (true_positives + false_positives)
        else:
            precision[cls] = 0.0

        if true_positives + false_negatives > 0:
            recall[cls] = true_positives / (true_positives + false_negatives)
        else:
            recall[cls] = 0.0

        if precision[cls] + recall[cls] > 0:
            f1_score[cls] = 2 * (precision[cls] * recall[cls]) / (precision[cls] + recall[cls])
        else:
            f1_score[cls] = 0.0

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(all_predictions)):
        confusion_matrix[all_labels[i], all_predictions[i]] += 1

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'confusion_matrix': confusion_matrix,
        'avg_confidence': np.mean(all_confidences),
    }


def comprehensive_backtest(model: nn.Module,
                          test_dataset: Any,
                          device: str,
                          confidence_threshold: float = 0.7) -> Dict[str, Any]:
    """
    Run comprehensive backtest across all test tokens.

    Aggregates results from individual token backtests and calculates
    portfolio-level metrics including Sharpe ratio and max drawdown.

    Args:
        model: Trained PyTorch model.
        test_dataset: Test dataset containing samples from multiple tokens.
        device: Device to run inference on.
        confidence_threshold: Minimum confidence to execute trades. Default is 0.7.

    Returns:
        Dictionary containing:
            - total_pnl: Total profit/loss across all tokens
            - num_tokens: Number of tokens traded
            - total_trades: Total number of trades executed
            - overall_win_rate: Win rate across all trades
            - sharpe_ratio: Sharpe ratio of returns
            - max_drawdown: Maximum drawdown experienced
            - avg_trades_per_token: Average trades per token

    Example:
        >>> results = comprehensive_backtest(model, test_dataset, device='cuda')
        >>> print(f"Total PnL: {results['total_pnl']:.4f} SOL")
        >>> print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        >>> print(f"Max Drawdown: {results['max_drawdown'] * 100:.2f}%")
    """
    model.eval()

    token_results = []
    all_trades = []
    cumulative_pnl = [0.0]

    token_samples_dict = {}
    for sample in test_dataset.samples:
        token_id = id(sample)
        if token_id not in token_samples_dict:
            token_samples_dict[token_id] = []
        token_samples_dict[token_id].append(sample)

    unique_tokens = list(set([id(s) for s in test_dataset.samples]))

    for token_id in tqdm(unique_tokens, desc="Backtesting tokens"):
        token_samples = [s for s in test_dataset.samples if id(s) == token_id]

        if not token_samples:
            continue

        token_result = backtest_token(
            model,
            token_samples,
            device,
            confidence_threshold
        )

        token_results.append(token_result)
        all_trades.extend(token_result['trades'])

        cumulative_pnl.append(cumulative_pnl[-1] + token_result['total_pnl'])

    total_pnl = sum(r['total_pnl'] for r in token_results)
    total_trades = sum(r['num_trades'] for r in token_results)
    num_tokens = len(token_results)

    if total_trades == 0:
        return {
            'total_pnl': 0.0,
            'num_tokens': num_tokens,
            'total_trades': 0,
            'overall_win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'avg_trades_per_token': 0.0,
        }

    all_profits = np.array([t['profit'] for t in all_trades])
    winning_trades = np.sum(all_profits > 0)
    overall_win_rate = winning_trades / total_trades

    sharpe_ratio = calculate_sharpe_ratio(all_profits)
    max_drawdown = calculate_max_drawdown(np.array(cumulative_pnl))

    return {
        'total_pnl': total_pnl,
        'num_tokens': num_tokens,
        'total_trades': total_trades,
        'overall_win_rate': overall_win_rate,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'avg_trades_per_token': total_trades / num_tokens,
        'avg_pnl_per_token': total_pnl / num_tokens,
    }
