"""
Performance metrics for backtesting.

Calculates and reports trading performance metrics.

Author: Trading Team
Date: 2025-12-29
"""

from typing import Dict, Any, List
from dataclasses import dataclass

import numpy as np

from .backtester import BacktestResult, Trade, TradeStatus


def calculate_metrics(result: BacktestResult) -> Dict[str, Any]:
    """
    Calculate comprehensive performance metrics.

    Args:
        result: BacktestResult from backtest run.

    Returns:
        Dictionary of metrics.
    """
    completed = [t for t in result.trades if t.status == TradeStatus.COMPLETED]

    if not completed:
        return {"error": "No completed trades"}

    returns = [t.net_pnl_pct for t in completed]
    equity = np.cumsum(returns)

    metrics = {
        # Count metrics
        "total_tokens": result.total_tokens,
        "screened_in": result.tokens_screened_in,
        "screened_in_pct": result.tokens_screened_in / result.total_tokens if result.total_tokens > 0 else 0,
        "entered": result.tokens_entered,
        "completed": result.tokens_completed,

        # Win/loss
        "wins": result.winning_trades,
        "losses": result.losing_trades,
        "win_rate": result.win_rate,

        # P&L
        "total_profit": result.total_profit,
        "total_loss": result.total_loss,
        "net_pnl": result.net_pnl,
        "avg_trade_pnl": np.mean(returns) if returns else 0,

        # Risk metrics
        "max_drawdown": result.max_drawdown,
        "sharpe_ratio": result.sharpe_ratio,
        "profit_factor": result.profit_factor,

        # Distribution
        "return_mean": np.mean(returns),
        "return_std": np.std(returns),
        "return_median": np.median(returns),
        "return_min": np.min(returns),
        "return_max": np.max(returns),

        # Final equity
        "final_equity": 1.0 + equity[-1] if len(equity) > 0 else 1.0,
    }

    # Exit reason distribution
    exit_reasons = {}
    for t in completed:
        reason = str(t.exit_reason.value) if t.exit_reason else "unknown"
        if reason not in exit_reasons:
            exit_reasons[reason] = 0
        exit_reasons[reason] += 1
    metrics["exit_reasons"] = exit_reasons

    return metrics


def print_metrics_report(result: BacktestResult) -> None:
    """
    Print formatted metrics report.

    Args:
        result: BacktestResult from backtest run.
    """
    metrics = calculate_metrics(result)

    if "error" in metrics:
        print(f"Error: {metrics['error']}")
        return

    print("\n" + "=" * 70)
    print("PERFORMANCE REPORT")
    print("=" * 70)

    print("\n--- Pipeline Efficiency ---")
    print(f"Total tokens analyzed:     {metrics['total_tokens']}")
    print(f"Passed screener:           {metrics['screened_in']} ({metrics['screened_in_pct']:.1%})")
    print(f"Entered positions:         {metrics['entered']}")
    print(f"Completed trades:          {metrics['completed']}")

    print("\n--- Trade Outcomes ---")
    print(f"Winning trades:            {metrics['wins']}")
    print(f"Losing trades:             {metrics['losses']}")
    print(f"Win rate:                  {metrics['win_rate']:.1%}")

    print("\n--- Profit & Loss ---")
    print(f"Total profit:              {metrics['total_profit']:.4f} SOL")
    print(f"Total loss:                {metrics['total_loss']:.4f} SOL")
    print(f"Net P&L:                   {metrics['net_pnl']:.4f} SOL")
    print(f"Average trade P&L:         {metrics['avg_trade_pnl']:.2%}")

    print("\n--- Risk Metrics ---")
    print(f"Maximum drawdown:          {metrics['max_drawdown']:.2%}")
    print(f"Sharpe ratio:              {metrics['sharpe_ratio']:.2f}")
    print(f"Profit factor:             {metrics['profit_factor']:.2f}")

    print("\n--- Return Distribution ---")
    print(f"Mean return:               {metrics['return_mean']:.2%}")
    print(f"Std deviation:             {metrics['return_std']:.2%}")
    print(f"Median return:             {metrics['return_median']:.2%}")
    print(f"Min return:                {metrics['return_min']:.2%}")
    print(f"Max return:                {metrics['return_max']:.2%}")

    print("\n--- Exit Reasons ---")
    for reason, count in metrics["exit_reasons"].items():
        pct = count / metrics["completed"] * 100
        print(f"  {reason:20s} {count:4d} ({pct:.1f}%)")

    print("\n--- Final Results ---")
    print(f"Final equity:              {metrics['final_equity']:.4f} (started at 1.0)")
    print(f"Total return:              {(metrics['final_equity'] - 1) * 100:.2f}%")

    print("\n" + "=" * 70)

    # Target comparison
    print("\n--- Target Performance Comparison ---")
    targets = {
        "Win rate": (metrics["win_rate"], 0.60, ">="),
        "Avg win/loss ratio": (
            result.avg_win / result.avg_loss if result.avg_loss > 0 else 0,
            2.0,
            ">="
        ),
        "Max drawdown": (metrics["max_drawdown"], 0.30, "<="),
    }

    for name, (actual, target, op) in targets.items():
        if op == ">=":
            passed = actual >= target
        else:
            passed = actual <= target

        status = "PASS" if passed else "FAIL"
        print(f"  {name:25s} Actual: {actual:.2f} | Target: {target:.2f} | {status}")

    print("=" * 70)
