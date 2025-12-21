#!/usr/bin/env python3
"""
Token data analysis script for GMGN trading bot
Analyzes historical token data to identify profitable patterns
"""

import csv
import json
import statistics
from datetime import datetime

def analyze_candle_data(candles_json):
    """
    Analyze candle data to extract key metrics

    Returns dict with:
    - max_price: highest price reached
    - min_price: lowest price reached
    - entry_price: price at first candle (discovered_at)
    - final_price: price at last candle
    - max_gain_pct: maximum potential gain from entry
    - total_duration: total seconds of data
    - volatility: standard deviation of close prices
    - volume_total: total trading volume
    """
    try:
        candles = json.loads(candles_json)
    except (json.JSONDecodeError, TypeError):
        return None

    if not candles or not isinstance(candles, list):
        return None

    entry_price = candles[0]['c']
    final_price = candles[-1]['c']

    all_highs = [c['h'] for c in candles]
    all_lows = [c['l'] for c in candles]
    all_closes = [c['c'] for c in candles]
    all_volumes = [c['v'] for c in candles]

    max_price = max(all_highs)
    min_price = min(all_lows)

    max_gain_pct = ((max_price - entry_price) / entry_price) * 100
    final_gain_pct = ((final_price - entry_price) / entry_price) * 100

    volatility = statistics.stdev(all_closes) if len(all_closes) > 1 else 0

    return {
        'entry_price': entry_price,
        'max_price': max_price,
        'min_price': min_price,
        'final_price': final_price,
        'max_gain_pct': max_gain_pct,
        'final_gain_pct': final_gain_pct,
        'total_duration': len(candles),
        'volatility': volatility,
        'volume_total': sum(all_volumes),
        'avg_volume': statistics.mean(all_volumes)
    }

def find_optimal_entry_exit(candles_json, max_trades=5):
    """
    Find optimal entry/exit points for multiple trades
    Returns list of (entry_idx, exit_idx, gain_pct) tuples
    """
    try:
        candles = json.loads(candles_json)
    except (json.JSONDecodeError, TypeError):
        return []

    if not candles or not isinstance(candles, list) or len(candles) < 2:
        return []

    trades = []

    # Simple strategy: buy on local minima, sell on local maxima
    for i in range(1, len(candles) - 1):
        # Check if local minimum
        if candles[i]['c'] < candles[i-1]['c'] and candles[i]['c'] < candles[i+1]['c']:
            # Look for next local maximum
            for j in range(i+1, len(candles) - 1):
                if candles[j]['c'] > candles[j-1]['c'] and candles[j]['c'] > candles[j+1]['c']:
                    gain_pct = ((candles[j]['c'] - candles[i]['c']) / candles[i]['c']) * 100
                    if gain_pct > 5:  # Only consider trades with >5% gain
                        trades.append({
                            'entry_idx': i,
                            'exit_idx': j,
                            'entry_price': candles[i]['c'],
                            'exit_price': candles[j]['c'],
                            'gain_pct': gain_pct,
                            'duration': j - i
                        })
                        break

    # Sort by gain and take top trades
    trades.sort(key=lambda x: x['gain_pct'], reverse=True)
    return trades[:max_trades]

def main():
    csv_file = '/home/tr4moryp/script/gmgn_trading/ai_data/data/tokens_2025-12-21.csv'

    tokens_analyzed = 0
    total_max_gain = 0
    profitable_tokens = 0

    max_gain_token = None
    max_gain_value = 0

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            tokens_analyzed += 1

            metrics = analyze_candle_data(row['candles'])
            if not metrics:
                continue

            total_max_gain += metrics['max_gain_pct']

            if metrics['max_gain_pct'] > 0:
                profitable_tokens += 1

            if metrics['max_gain_pct'] > max_gain_value:
                max_gain_value = metrics['max_gain_pct']
                max_gain_token = {
                    'address': row['token_address'],
                    'symbol': row['symbol'],
                    'discovered_age': row['discovered_age_sec'],
                    **metrics
                }

            # Print first 5 tokens for detailed view
            if tokens_analyzed <= 5:
                print(f"\n{'='*60}")
                print(f"Token: {row['symbol']} ({row['token_address'][:8]}...)")
                print(f"Discovered at: {row['discovered_age_sec']}s after creation")
                print(f"Entry price: ${metrics['entry_price']:.2f}")
                print(f"Max price: ${metrics['max_price']:.2f}")
                print(f"Final price: ${metrics['final_price']:.2f}")
                print(f"Max gain: {metrics['max_gain_pct']:.2f}%")
                print(f"Final gain: {metrics['final_gain_pct']:.2f}%")
                print(f"Duration: {metrics['total_duration']} seconds")
                print(f"Volatility: ${metrics['volatility']:.2f}")

                # Find optimal trades
                optimal_trades = find_optimal_entry_exit(row['candles'], max_trades=3)
                if optimal_trades:
                    print(f"\nOptimal trades found: {len(optimal_trades)}")
                    for idx, trade in enumerate(optimal_trades, 1):
                        print(f"  Trade {idx}: Buy @${trade['entry_price']:.2f} "
                              f"-> Sell @${trade['exit_price']:.2f} "
                              f"= {trade['gain_pct']:.2f}% gain ({trade['duration']}s)")

    # Summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"Total tokens analyzed: {tokens_analyzed}")
    print(f"Profitable tokens (max_gain > 0): {profitable_tokens} ({profitable_tokens/tokens_analyzed*100:.1f}%)")
    print(f"Average max gain: {total_max_gain/tokens_analyzed:.2f}%")

    print(f"\nBest performing token:")
    print(f"  Symbol: {max_gain_token['symbol']}")
    print(f"  Address: {max_gain_token['address']}")
    print(f"  Max gain: {max_gain_token['max_gain_pct']:.2f}%")
    print(f"  Entry price: ${max_gain_token['entry_price']:.2f}")
    print(f"  Max price: ${max_gain_token['max_price']:.2f}")
    print(f"  Duration: {max_gain_token['total_duration']} seconds")

if __name__ == '__main__':
    main()
