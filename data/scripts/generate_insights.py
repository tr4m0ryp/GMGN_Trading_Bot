#!/usr/bin/env python3
"""
Generate detailed insights and statistics from token data
Creates a comprehensive report for strategy validation
"""

import csv
import json
import statistics
from collections import defaultdict

def analyze_all_tokens():
    """Comprehensive analysis of all tokens"""

    csv_file = '/home/tr4moryp/script/gmgn_trading/ai_data/data/tokens_2025-12-21.csv'

    results = {
        'total_tokens': 0,
        'profitable_tokens': 0,
        'gains': [],
        'durations': [],
        'discovery_ages': [],
        'volatilities': [],
        'gain_by_duration': defaultdict(list),
        'gain_by_discovery_age': defaultdict(list),
        'multi_trade_profits': [],
    }

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                candles = json.loads(row['candles'])
            except (json.JSONDecodeError, TypeError):
                continue

            if not candles or not isinstance(candles, list):
                continue

            results['total_tokens'] += 1

            # Basic metrics
            entry_price = candles[0]['c']
            all_highs = [c['h'] for c in candles]
            all_closes = [c['c'] for c in candles]
            max_price = max(all_highs)

            max_gain_pct = ((max_price - entry_price) / entry_price) * 100
            duration = len(candles)
            discovery_age = int(row['discovered_age_sec'])
            volatility = statistics.stdev(all_closes) if len(all_closes) > 1 else 0

            results['gains'].append(max_gain_pct)
            results['durations'].append(duration)
            results['discovery_ages'].append(discovery_age)
            results['volatilities'].append(volatility)

            if max_gain_pct > 0:
                results['profitable_tokens'] += 1

            # Bucket by duration
            duration_bucket = (duration // 60) * 60  # Round to nearest 60s
            results['gain_by_duration'][duration_bucket].append(max_gain_pct)

            # Bucket by discovery age
            age_bucket = (discovery_age // 10) * 10  # Round to nearest 10s
            results['gain_by_discovery_age'][age_bucket].append(max_gain_pct)

            # Simulate multi-trade strategy
            multi_trade_profit = simulate_multi_trades(candles)
            results['multi_trade_profits'].append(multi_trade_profit)

    return results

def simulate_multi_trades(candles, num_trades=3):
    """
    Simulate a simple multi-trade strategy
    Returns total profit percentage from multiple trades
    """
    if len(candles) < 10:
        return 0

    total_profit = 0
    trades_executed = 0
    in_position = False
    entry_price = 0
    entry_idx = 0

    target_profit = 0.20  # 20% target
    stop_loss = -0.08  # -8% stop

    for i in range(1, len(candles)):
        current_price = candles[i]['c']

        if not in_position:
            # Look for entry (local minimum)
            if i > 0 and i < len(candles) - 1:
                if candles[i]['c'] < candles[i-1]['c'] and candles[i]['c'] < candles[i+1]['c']:
                    # Enter position
                    in_position = True
                    entry_price = current_price
                    entry_idx = i
        else:
            # Check exit conditions
            pnl_pct = (current_price - entry_price) / entry_price

            # Take profit or stop loss
            if pnl_pct >= target_profit or pnl_pct <= stop_loss or (i - entry_idx) > 30:
                total_profit += pnl_pct * 100
                trades_executed += 1
                in_position = False

                if trades_executed >= num_trades:
                    break

    return total_profit

def print_report(results):
    """Print comprehensive analysis report"""

    print("=" * 70)
    print("GMGN TOKEN ANALYSIS REPORT")
    print("=" * 70)
    print()

    # Overall statistics
    print("OVERALL STATISTICS")
    print("-" * 70)
    print(f"Total tokens analyzed: {results['total_tokens']}")
    print(f"Profitable tokens: {results['profitable_tokens']} ({results['profitable_tokens']/results['total_tokens']*100:.1f}%)")
    print()

    # Gain statistics
    print("GAIN DISTRIBUTION")
    print("-" * 70)
    gains = results['gains']
    print(f"Average max gain: {statistics.mean(gains):.2f}%")
    print(f"Median max gain: {statistics.median(gains):.2f}%")
    print(f"Min max gain: {min(gains):.2f}%")
    print(f"Max max gain: {max(gains):.2f}%")
    print(f"Std dev: {statistics.stdev(gains):.2f}%")
    print()

    # Percentiles
    gains_sorted = sorted(gains)
    print("Gain Percentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        idx = int(len(gains_sorted) * p / 100)
        print(f"  {p}th percentile: {gains_sorted[idx]:.2f}%")
    print()

    # Duration statistics
    print("DURATION ANALYSIS")
    print("-" * 70)
    durations = results['durations']
    print(f"Average duration: {statistics.mean(durations):.1f} seconds")
    print(f"Median duration: {statistics.median(durations):.1f} seconds")
    print(f"Min duration: {min(durations)} seconds")
    print(f"Max duration: {max(durations)} seconds")
    print()

    # Discovery age statistics
    print("DISCOVERY AGE ANALYSIS")
    print("-" * 70)
    ages = results['discovery_ages']
    print(f"Average discovery age: {statistics.mean(ages):.1f} seconds")
    print(f"Median discovery age: {statistics.median(ages):.1f} seconds")
    print(f"Min discovery age: {min(ages)} seconds")
    print(f"Max discovery age: {max(ages)} seconds")
    print()

    # Gain by duration
    print("AVERAGE GAIN BY DURATION")
    print("-" * 70)
    for duration_bucket in sorted(results['gain_by_duration'].keys())[:10]:
        avg_gain = statistics.mean(results['gain_by_duration'][duration_bucket])
        count = len(results['gain_by_duration'][duration_bucket])
        print(f"{duration_bucket:3d}-{duration_bucket+59:3d}s: {avg_gain:7.2f}% (n={count})")
    print()

    # Gain by discovery age
    print("AVERAGE GAIN BY DISCOVERY AGE")
    print("-" * 70)
    for age_bucket in sorted(results['gain_by_discovery_age'].keys())[:15]:
        avg_gain = statistics.mean(results['gain_by_discovery_age'][age_bucket])
        count = len(results['gain_by_discovery_age'][age_bucket])
        print(f"{age_bucket:3d}-{age_bucket+9:3d}s: {avg_gain:7.2f}% (n={count})")
    print()

    # Multi-trade simulation
    print("MULTI-TRADE STRATEGY SIMULATION")
    print("-" * 70)
    multi_profits = [p for p in results['multi_trade_profits'] if p != 0]
    if multi_profits:
        print(f"Tokens with profitable trades: {len(multi_profits)} ({len(multi_profits)/results['total_tokens']*100:.1f}%)")
        print(f"Average profit (3 trades): {statistics.mean(multi_profits):.2f}%")
        print(f"Median profit (3 trades): {statistics.median(multi_profits):.2f}%")
        print(f"Best performance: {max(multi_profits):.2f}%")
        print(f"Worst performance: {min(multi_profits):.2f}%")
    print()

    # Profitability breakdown
    print("PROFITABILITY BREAKDOWN")
    print("-" * 70)
    gain_ranges = [
        (0, 10, "0-10%"),
        (10, 25, "10-25%"),
        (25, 50, "25-50%"),
        (50, 100, "50-100%"),
        (100, 200, "100-200%"),
        (200, 500, "200-500%"),
        (500, 1000, "500-1000%"),
        (1000, float('inf'), "1000%+"),
    ]

    for min_g, max_g, label in gain_ranges:
        count = len([g for g in gains if min_g <= g < max_g])
        pct = count / len(gains) * 100
        print(f"{label:15s}: {count:4d} tokens ({pct:5.1f}%)")
    print()

    # Expected value calculation
    print("EXPECTED VALUE ANALYSIS (Conservative Strategy)")
    print("-" * 70)
    win_rate = 0.60
    avg_win = 0.15  # 15%
    avg_loss = -0.08  # -8%
    trades_per_token = 3
    position_size = 0.01  # SOL

    ev_per_trade = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    ev_per_token = ev_per_trade * trades_per_token
    profit_per_token = position_size * trades_per_token * ev_per_trade

    print(f"Assumptions:")
    print(f"  Win rate: {win_rate*100:.0f}%")
    print(f"  Average win: {avg_win*100:.0f}%")
    print(f"  Average loss: {avg_loss*100:.0f}%")
    print(f"  Trades per token: {trades_per_token}")
    print(f"  Position size: {position_size} SOL")
    print()
    print(f"Results:")
    print(f"  EV per trade: {ev_per_trade*100:.2f}%")
    print(f"  EV per token: {ev_per_token*100:.2f}%")
    print(f"  Profit per token: {profit_per_token:.6f} SOL")
    print()
    print(f"Daily projections (50 tokens):")
    print(f"  Total profit: {profit_per_token * 50:.4f} SOL")
    print(f"  Total invested: {position_size * trades_per_token * 50:.2f} SOL")
    print(f"  ROI: {ev_per_token*100:.2f}%")
    print()

    print("=" * 70)

if __name__ == '__main__':
    results = analyze_all_tokens()
    print_report(results)
