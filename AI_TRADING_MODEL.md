# Solana AI Trading Bot - Complete Documentation

> **AI-powered high-frequency trading bot with Jito Bundle Integration**

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Data Analysis Results](#data-analysis-results)
3. [Jito Trading Infrastructure](#jito-trading-infrastructure)
4. [Trading Strategy Design](#trading-strategy-design)
5. [AI Model Architecture](#ai-model-architecture)
6. [Model Self-Improvement System](#model-self-improvement-system)
7. [Risk Management](#risk-management)
8. [Implementation Guide](#implementation-guide)
9. [Quick Start](#quick-start)
10. [Appendix](#appendix)

---

# Executive Summary

## Project Overview

This document outlines a complete AI-powered trading bot for Solana token markets using Jito Bundle Integration for ultra-fast execution. The bot uses machine learning to identify profitable entry/exit points on newly discovered tokens, executing rapid trades to capture volatility-driven gains with minimal latency and MEV protection.

## Key Findings at a Glance

| Metric | Value |
|--------|-------|
| **Tokens analyzed** | 328 |
| **Profitable tokens** | 326 (99.4%) |
| **Average max gain** | 174.49% |
| **Best performing token** | 1247.81% |
| **Typical profitable window** | 60-400 seconds |

## Critical Training Constraint

**Historical Data Only**: During model training, the AI can only observe historical price data of live tokens. The model must learn to predict future price movements based solely on past patterns - it cannot see future data during training or inference. This fundamental constraint reflects real-world trading conditions.

## Revolutionary Training Approach: Full Historical Context

**Unlike traditional fixed-window models**, this system uses **variable-length sequences** where the model sees ALL historical data from token discovery to the current timestamp, exactly simulating how a real trader watches a live chart evolve.

**Key Innovation:**
- At T=31s: Model sees 31 candles [0-30] - knows token is brand new
- At T=100s: Model sees 100 candles [0-99] - sees if it already pumped 300%
- At T=300s: Model sees 300 candles [0-299] - full lifecycle visible

This approach is vastly superior to fixed 30-second windows because:
1. **Real Market Simulation**: Exactly mimics watching a live trading chart
2. **Full Context**: Model knows if token already pumped 500% or just starting
3. **Lifecycle Awareness**: Sequence length itself indicates token maturity
4. **Pattern Recognition**: Can identify pump cycles, support/resistance across full history
5. **No Artificial Constraints**: Removes the arbitrary limitation of fixed windows

## Recommended Strategy Summary

| Parameter | Value |
|-----------|-------|
| Position size | **0.01 SOL (fixed)** |
| Trades per token | 3-5 |
| Target gain (GROSS) | 30-40% (dynamic, reevaluated) |
| Stop loss | -8% (dynamic, reevaluated) |
| Expected NET profit | 3-6% per token |
| Transaction delay (Jito) | 0.3-0.8 seconds |

### Critical Model Behavior: Iterative Self-Improvement

**After each token trading session**, the model MUST:
1. **Reevaluate performance** - Analyze actual vs expected profit/loss
2. **Adjust parameters** - Dynamically update take-profit and stop-loss thresholds based on token behavior
3. **Learn from outcomes** - Investigate what worked and what failed to improve future decisions
4. **Optimize strategy** - Adapt entry/exit timing based on observed patterns

**Current Performance Baseline** (with 20% profit targets and -8% stop losses):

| Metric | Result |
|--------|--------|
| Success rate | 324/326 tokens (99.4%) |
| Average profit | 43.25% per token |
| Median profit | 42.05% per token |
| Best performance | 214.17% on single token |
| Worst performance | -43.67% on single token |

**Key Insight**: While initial take-profit (30-40%) and stop-loss (-8%) parameters are defined, the model should continuously reevaluate and adjust these thresholds based on real-time token behavior patterns to maximize gains and minimize losses.

---

# Data Analysis Results

## Profitability Statistics

**Dataset:** 326 tokens from GMGN.ai (December 2025)
**Analysis Period:** First 14-501 seconds after discovery
**Success Rate:** 100% of tokens showed profitable opportunities

| Metric | Value |
|--------|-------|
| Profitable tokens | 326/326 (100%) |
| Average max gain | 175.56% |
| Median max gain | 123.74% |
| 75th percentile gain | 235.41% |
| Best performing token | 1247.81% |

## Gain Distribution

- **25% of tokens** achieved >235% max gain
- **50% of tokens** achieved >123% max gain
- **75% of tokens** achieved >67% max gain
- **90% of tokens** achieved >43% max gain

## Duration Insights

Tokens with longer active trading periods show significantly higher gains:

| Duration | Average Max Gain | Sample Size |
|----------|-----------------|-------------|
| 0-60s | 70.47% | 56 tokens |
| 60-120s | 124.47% | 133 tokens |
| 120-180s | 212.44% | 75 tokens |
| 180-240s | 300.88% | 33 tokens |
| 240-300s | 307.29% | 11 tokens |
| 300-360s | 452.07% | 6 tokens |

**Insight:** Tokens that remain active for 2+ minutes show 200-450% average gains.

## Price Movement Patterns

1. **Extreme Volatility Window**: First 30-200 seconds after discovery show highest volatility
2. **Multiple Pump Cycles**: Tokens typically experience 2-5 distinct price pumps
3. **Rapid Gains**: Most significant gains occur within 2-10 seconds
4. **Local Minima/Maxima**: Clear patterns of local bottoms and tops suitable for swing trading

## Profit Potential Examples

**Token FOOL:** 281% max gain with 3 optimal trades:
- Trade 1: 79.32% gain in 5 seconds
- Trade 2: 32.30% gain in 10 seconds
- Trade 3: 29.86% gain in 15 seconds

**Token MOTION:** 91.55% max gain with 3 optimal trades:
- Trade 1: 39.42% gain in 3 seconds
- Trade 2: 37.77% gain in 2 seconds
- Trade 3: 34.30% gain in 4 seconds

## Multi-Trade Strategy Simulation

Simulated strategy: 3 trades per token with 20% profit targets and -8% stop losses

| Metric | Result |
|--------|--------|
| Success rate | 324/326 tokens (99.4%) |
| Average profit | 43.25% per token |
| Median profit | 42.05% per token |
| Best performance | 214.17% on single token |
| Worst performance | -43.67% on single token |

---

# Jito Trading Infrastructure

## Fixed Position Size

**All trades use a fixed buy amount of 0.01 SOL.** This standardized position size:
- Simplifies risk management
- Ensures consistent exposure per trade
- Allows accurate fee calculation
- Makes backtesting results directly comparable

## Jito Bundle Integration

Jito provides the fastest transaction execution on Solana through MEV infrastructure, enabling sub-second confirmations and MEV protection.

**Key Benefits:**
| Feature | Benefit |
|---------|---------|
| Direct validator routing | Bypasses public mempool |
| Bundle atomicity | All-or-nothing execution |
| Parallel auctions (50ms ticks) | Near-instant inclusion |
| MEV protection | Prevents sandwich attacks |

**Jito Endpoints by Region:**
```
Mainnet (Global):     https://mainnet.block-engine.jito.wtf
Amsterdam (EU):       https://amsterdam.mainnet.block-engine.jito.wtf
Frankfurt (EU):       https://frankfurt.mainnet.block-engine.jito.wtf
New York (US):        https://ny.mainnet.block-engine.jito.wtf
Tokyo (Asia):         https://tokyo.mainnet.block-engine.jito.wtf
Singapore (Asia):     https://singapore.mainnet.block-engine.jito.wtf
```

**Jito Tip Structure:**
- Minimum tip: **1000 lamports** (0.000001 SOL)
- Recommended for high-priority: **10,000-100,000 lamports** (0.00001-0.0001 SOL)
- Include tip in main transaction for atomicity

**Tip Accounts** (pick one randomly to reduce contention):
```
96gYZGLnJYVFmbjzopPSU6QiEV5fGqZNyN9nmNhvrZU5
HFqU5x63VTqvQss8hp11i4wVV8bD44PvwucfZ2bU7gRe
Cw8CFyM9FkoMi7K7Crf6HNQqf4uEMzpKw6QNghXLvLkY
ADaUMid9yfUytqMBgopwjb2DTLSokTSzL1zt6iGPaS49
DfXygSm4jCyNCybVYYK6DwvWqjKee8pbDmJGcLWNDXjh
ADuUkR4vqLUMWXxW9gh6D6L8pMSawimctcNZ5pGwDcEt
DttWaMuVvTiduZRnguLF7jNxTgiMBZ1hyAumKUiL2KRL
3AVi9Tg9Uo68tJfuvoKvqKNWKkC5wPdSSdeBnizKZ6jT
```

## Transaction Delay Simulation (Jito)

With Jito's optimized infrastructure, transaction delays are significantly reduced:

| Metric | Value |
|--------|-------|
| Average delay | 0.3-0.8 seconds |
| Delay range | 0.2-1.2 seconds |
| Simulation window | 1 second (conservative) |
| Simulation approach | Use HIGHEST price in delay window |

**Delay Simulation Logic:**
```
When simulating a BUY order at time T:
  - Look at all prices from T to T+1 second (Jito optimized)
  - Use the HIGHEST price in this window as actual buy price
  - This represents worst-case slippage scenario

When simulating a SELL order at time T:
  - Look at all prices from T to T+1 second
  - Use the LOWEST price in this window as actual sell price
  - This represents worst-case slippage scenario
```

This conservative approach ensures the model is trained to be profitable even under adverse execution conditions.

## Transaction Fee Structure (Jito Only)

Every BUY and SELL transaction incurs the following fees:

| Fee Type | Amount | Notes |
|----------|--------|-------|
| Jito Tip | 0.00001-0.0001 SOL | Priority for fast inclusion |
| Solana Gas Fee | ~0.0002 SOL | Network base fee |
| Priority Fee | ~0.0001 SOL | Compute unit pricing |
| **Total per transaction** | **~0.0003-0.0005 SOL** | No platform fees |

**Fee Impact Calculation (per round trip):**
```
For 0.01 SOL position:
  BUY transaction:
    - Jito tip: ~0.00005 SOL (average)
    - Gas fee: ~0.0002 SOL
    - Priority fee: ~0.0001 SOL
    - Subtotal: ~0.00035 SOL

  SELL transaction:
    - Jito tip: ~0.00005 SOL
    - Gas fee: ~0.0002 SOL
    - Priority fee: ~0.0001 SOL
    - Subtotal: ~0.00035 SOL

  Total round-trip fees: ~0.0007 SOL (7% of position)

  MINIMUM REQUIRED GAIN TO BREAK EVEN: ~7-8%
```

**Critical Insight:** With Jito infrastructure, fees are reduced to ~7% per round trip (vs 10% with platform fees). The model must target gains of **at least 10-12%** to achieve meaningful profit after all costs, with recommended targets of **30-40%** for optimal risk/reward.

---

# Trading Strategy Design

## Core Strategy: Multi-Trade Momentum Scalping

The bot will execute multiple rapid trades on the same token, capturing profit from volatility waves.

## Position Sizing (FIXED)

- **Fixed position size: 0.01 SOL per trade (non-negotiable)**
- Maximum concurrent positions on single token: 3-5 trades
- Total risk per token: 0.03-0.05 SOL
- With 174% average max gain and realistic fees (~10%), expected net profit: 0.042-0.070 SOL per token

## Entry Signals

The model should identify entry points using these indicators:

### 1. Local Minimum Detection
- Price < previous candle AND price < next candle
- Volume spike (>150% of average)
- Minimum 5% drop from recent high

### 2. Momentum Reversal
- RSI < 30 (oversold)
- Price bouncing off support level
- Increasing buy volume

### 3. Early Entry Optimization
- First trade: Immediate entry at discovery (if volatility indicators are positive)
- Subsequent trades: Wait for pullbacks (3-7% dips)

## Exit Signals

The model should exit positions when:

### 1. Profit Target Hit
- Conservative: 15% gain (1.5 seconds typical)
- Moderate: 25% gain (3-5 seconds typical)
- Aggressive: 40% gain (5-10 seconds typical)

### 2. Stop Loss Triggered
- Hard stop: -8% from entry
- Trailing stop: Activate at +10% gain, trail by 5%

### 3. Time-Based Exit
- If no movement after 20 seconds, exit at breakeven or small loss
- Maximum hold time: 60 seconds per trade

## Multi-Trade Logic

Execute multiple trades per token:

```
Trade 1 (Aggressive Entry):
  - Entry: At discovery or first pullback
  - Target: 15-20% gain
  - Stop: -8%

Trade 2 (Main Position):
  - Entry: After 5-15 second pullback (3-7% dip)
  - Target: 25-35% gain
  - Stop: -8%

Trade 3 (Swing Trade):
  - Entry: Second major pullback (5-10% dip)
  - Target: 40-60% gain
  - Stop: -8%

Trade 4-5 (Opportunistic):
  - Entry: Only if strong momentum continues
  - Target: 20-30% gain
  - Stop: -5%
```

---

# AI Model Architecture

## Recommended Approach: Variable-Length Sequence Models

The model receives ALL historical data from token discovery to current timestamp, simulating real market conditions where traders see the full chart.

### 1. Pattern Recognition (LSTM/Transformer with Variable Lengths)

**Critical Insight:** Unlike traditional fixed-window approaches, this model sees the ENTIRE price history from token discovery, just like a real trader watching a live chart.

```
Input: All historical OHLCV data from discovery to current time (variable length)
  - At T=31s:  Model sees 31 candles [0-30]
  - At T=60s:  Model sees 60 candles [0-59]
  - At T=200s: Model sees 200 candles [0-199]

Architecture Option A (LSTM - Handles variable lengths natively):
  - Input layer: (batch, seq_len, 11 features)
  - LSTM layers: 2-3 layers with 128-256 hidden units
  - Dropout: 0.2-0.3 for regularization
  - Dense layers: Final classification (BUY/SELL/HOLD)

Architecture Option B (Transformer - Better for long sequences):
  - Input embedding: Project 11 features to d_model=128
  - Positional encoding: Add temporal information
  - Transformer encoder: 4-6 layers, 8 attention heads
  - Output layer: Classification head

Output: Trade signal + confidence score
```

**Why Variable-Length is Superior:**

| Aspect | Fixed 30s Window | Variable-Length (All History) |
|--------|------------------|-------------------------------|
| Realism | Partial view only | Full market view like real traders |
| Context | No lifecycle awareness | Sees if token already pumped 500% |
| Time awareness | Needs explicit feature | Implicit from sequence length |
| Pattern recognition | Local patterns only | Global trends + local patterns |
| Training alignment | Artificial constraint | Matches real trading exactly |

### 2. Feature Engineering

**Technical Indicators:**
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Volume-weighted average price (VWAP)
- Price momentum (rate of change)
- Volatility (standard deviation of price)

**Pattern Features:**
- Local minima/maxima count
- Average pump amplitude
- Time since last pump
- Support/resistance levels

### 3. Model Input Features

Each training sample consists of ALL historical data from token discovery to current timestamp.

**Raw OHLCV Data (variable-length sequence):**
- Open, High, Low, Close, Volume for EVERY second from discovery to current time
- Minimum sequence length: 30 candles (30 seconds minimum history required)
- Maximum sequence length: Up to 500+ candles for long-lived tokens

**Technical Indicators (computed per candle):**
- RSI (14-period, adjusted for available data length)
- MACD (12, 26, 9)
- Bollinger Bands (upper, middle, lower, 20-period)
- VWAP (Volume-Weighted Average Price)
- Momentum (Rate of Change)
- Volatility (Rolling standard deviation)

**Derived Features (per timestep):**
- Cumulative max price seen (resistance level identification)
- Cumulative min price seen (support level identification)
- Volume trend ratio (current vs average)
- Price position in historical range (0.0 to 1.0)
- Seconds since discovery (implicit from sequence length)

**Input shape:** `(batch, variable_seq_len, 11)` where:
- `variable_seq_len` = current_timestamp - discovery_timestamp (in seconds)
- 11 features per timestep: [Open, High, Low, Close, Volume, RSI, MACD, BB_upper, BB_lower, VWAP, Momentum]

**Example Shapes:**
```
Token at 31s:  (batch, 31, 11)   - Early stage, high volatility expected
Token at 60s:  (batch, 60, 11)   - First minute complete, patterns emerging
Token at 200s: (batch, 200, 11)  - Mature token, model sees full pump history
```

**Critical Advantage:** The sequence length itself is a powerful signal:
- Short sequences (30-60s): Token is fresh, aggressive entry viable
- Medium sequences (60-180s): Prime trading window, multiple cycles visible
- Long sequences (180s+): Token maturing, model sees if it's dying or continuing

### 4. Model Output

Binary classification for each timestep:
- **BUY (1)**: Enter position (if no position) or add to position
- **SELL (-1)**: Exit position (if position exists)
- **HOLD (0)**: No action

Plus confidence score: `[0.0, 1.0]`

**Only execute trades when confidence > 0.7**

### 5. Reinforcement Learning (Optional Advanced)

Train an agent to:
- **State:** Current price, position, unrealized PnL, technical indicators
- **Actions:** BUY, SELL, HOLD
- **Reward:** Total profit - (number of trades * transaction cost)
- **Algorithm:** PPO (Proximal Policy Optimization) or DQN

## Training Data Preparation

From the 328 tokens, create labeled dataset with **realistic execution simulation and full historical context**:

```python
# Constants for realistic simulation (Jito-optimized)
FIXED_POSITION_SIZE = 0.01  # SOL
DELAY_SECONDS = 1  # Jito transaction confirmation delay
JITO_TIP_AVG = 0.00005  # Average Jito tip per transaction (50,000 lamports)
GAS_FEE_FIXED = 0.0002  # Solana gas fee
PRIORITY_FEE = 0.0001  # Priority fee for compute units
TOTAL_FEE_PER_TX = JITO_TIP_AVG + GAS_FEE_FIXED + PRIORITY_FEE  # ~0.00035 SOL
MIN_PROFIT_TARGET = 0.10  # 10% minimum to account for fees
MIN_HISTORY_LENGTH = 30  # Require at least 30 seconds of history

def get_execution_price(candles, start_idx, is_buy=True):
    """
    Simulate transaction delay by finding worst-case price
    in the 1 second execution window (Jito-optimized).
    """
    end_idx = min(start_idx + DELAY_SECONDS + 1, len(candles))
    delay_candles = candles[start_idx:end_idx]

    if not delay_candles:
        return candles[start_idx]['c']

    if is_buy:
        # For BUY: use highest price (worst case slippage)
        return max(c['h'] for c in delay_candles)
    else:
        # For SELL: use lowest price (worst case slippage)
        return min(c['l'] for c in delay_candles)

def calculate_net_profit(buy_price, sell_price, position_size=FIXED_POSITION_SIZE):
    """
    Calculate actual profit after all fees (Jito infrastructure).
    """
    # Tokens received after buy
    tokens = position_size / buy_price

    # SOL received from sell
    sell_value = tokens * sell_price

    # Subtract transaction fees for both BUY and SELL
    net_value = sell_value - (2 * TOTAL_FEE_PER_TX)

    return net_value - position_size  # Profit/Loss

def prepare_realistic_training_data(token_candles):
    """
    Prepare training data with FULL HISTORICAL CONTEXT at each timestep.

    This simulates real trading: at each moment, the model sees the entire
    price history from token discovery to current time, just like watching
    a live chart.
    """
    samples = []

    # Start from candle 30 (minimum history requirement)
    for current_time in range(MIN_HISTORY_LENGTH, len(token_candles)):
        # CRITICAL: Model sees ALL history from discovery to current_time
        # This is the key difference from fixed-window approaches
        historical_data = token_candles[0:current_time]  # Variable length!

        # Extract features from all historical candles
        features = extract_features(historical_data)  # Returns (current_time, 11)

        # Simulate buy execution with realistic 1-second delay
        buy_price = get_execution_price(token_candles, current_time, is_buy=True)

        # Look ahead to find optimal sell point (for labeling ONLY)
        # Model will NEVER see this during inference
        future_window_end = min(current_time + 20, len(token_candles))
        future_candles = token_candles[current_time + DELAY_SECONDS:future_window_end]

        if not future_candles:
            continue  # Not enough future data for labeling

        # Find best sell price in next 20 seconds (for label only)
        best_sell_price = max(c['h'] for c in future_candles)
        worst_sell_price = min(c['l'] for c in future_candles)

        # Calculate potential NET profit (after fees)
        best_net_profit = calculate_net_profit(buy_price, best_sell_price)
        best_profit_pct = best_net_profit / FIXED_POSITION_SIZE

        # Determine optimal label based on net profit potential
        if best_profit_pct > 0.10:  # >10% net profit possible
            label = 1  # BUY
        elif best_profit_pct < 0.03:  # <3% profit, not worth the risk
            label = 0  # HOLD
        else:
            label = 2  # SELL (if we have position)

        samples.append({
            'features': features,  # Shape: (current_time, 11) - Variable!
            'label': label,
            'seq_length': current_time,  # Actual sequence length
            'timestamp': current_time,
            'buy_price': buy_price,
            'best_future_price': best_sell_price,
            'potential_profit_pct': best_profit_pct
        })

    return samples

def extract_features(candles):
    """
    Extract technical indicators from historical candles.

    Args:
        candles: List of OHLCV candles from discovery to current time

    Returns:
        numpy array of shape (len(candles), 11) with features per timestep
    """
    import numpy as np
    import pandas as pd

    df = pd.DataFrame(candles)
    features = np.zeros((len(df), 11))

    # Raw OHLCV (5 features)
    features[:, 0] = df['o'].values  # Open
    features[:, 1] = df['h'].values  # High
    features[:, 2] = df['l'].values  # Low
    features[:, 3] = df['c'].values  # Close
    features[:, 4] = df['v'].values  # Volume

    # Technical indicators (6 features)
    features[:, 5] = calculate_rsi(df['c'], period=14)
    features[:, 6] = calculate_macd(df['c'])
    features[:, 7] = calculate_bollinger_upper(df['c'], period=20)
    features[:, 8] = calculate_bollinger_lower(df['c'], period=20)
    features[:, 9] = calculate_vwap(df['c'], df['v'])
    features[:, 10] = calculate_momentum(df['c'], period=10)

    return features

# Process all tokens
all_training_samples = []
for token in tokens:
    token_samples = prepare_realistic_training_data(token['candles'])
    all_training_samples.extend(token_samples)

print(f"Generated {len(all_training_samples)} training samples")
print(f"Sequence lengths range from {MIN_HISTORY_LENGTH} to {max(s['seq_length'] for s in all_training_samples)} candles")
```

**Key Training Principles:**

1. **Full Historical Context**: Model sees ALL candles from discovery to current time
   - At 31s: sees 31 candles [0-30]
   - At 100s: sees 100 candles [0-99]
   - At 300s: sees 300 candles [0-299]

2. **Real Market Simulation**: Exactly mimics watching a live chart evolve
   - No artificial fixed-window constraints
   - Model knows token age implicitly from sequence length
   - Can identify if token already pumped 500% vs just starting

3. **Realistic Execution**: Simulates 1-second delay with worst-case pricing
   - Buy orders: highest price in delay window (worst slippage)
   - Sell orders: lowest price in delay window (worst slippage)

4. **Fee-Aware Labels**: All labels based on NET profit potential
   - Accounts for Jito tips, gas fees, priority fees (~7% total)
   - BUY only when >10% net profit possible
   - Ensures trained model is profitable in real trading

5. **Variable-Length Sequences**: Each sample has different length
   - Requires padding/masking for batching
   - LSTM handles natively with pack_padded_sequence
   - Transformers use attention masks

## Model Implementation Examples

### LSTM Implementation (Recommended for Start)

```python
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class VariableLengthLSTMTrader(nn.Module):
    """
    LSTM-based trading model that handles variable-length sequences.
    Sees all historical data from token discovery to current time.
    """
    def __init__(self, input_size=11, hidden_size=128, num_layers=2, num_classes=3):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers (handles temporal dependencies)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=False
        )

        # Fully connected layers for classification
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)  # BUY/HOLD/SELL
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, lengths):
        """
        Args:
            x: Padded sequences (batch, max_seq_len, features)
            lengths: Actual length of each sequence (batch,)

        Returns:
            predictions: (batch, num_classes)
            confidence: (batch,)
        """
        # Pack padded sequences (handles variable lengths efficiently)
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Pass through LSTM
        lstm_out, (hidden, cell) = self.lstm(packed)

        # Use final hidden state (last timestep)
        final_hidden = hidden[-1]  # (batch, hidden_size)

        # Classification layers
        out = self.fc1(final_hidden)
        out = self.relu(out)
        out = self.dropout(out)
        logits = self.fc2(out)
        predictions = self.softmax(logits)

        # Confidence is max probability
        confidence, _ = torch.max(predictions, dim=1)

        return predictions, confidence

# Training loop example
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for batch in train_loader:
        # Unpack batch
        features = batch['features'].to(device)  # (batch, max_len, 11)
        labels = batch['labels'].to(device)      # (batch,)
        lengths = batch['seq_lengths']            # (batch,)

        # Forward pass
        predictions, confidence = model(features, lengths)

        # Calculate loss
        loss = criterion(predictions, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)
```

### Transformer Implementation (For Advanced Users)

```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Add positional information to sequences."""
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerTrader(nn.Module):
    """
    Transformer-based trading model for variable-length sequences.
    Better than LSTM for capturing long-range dependencies.
    """
    def __init__(self, input_size=11, d_model=128, nhead=8, num_layers=4, num_classes=3):
        super().__init__()

        # Project input features to d_model dimensions
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.3,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.fc1 = nn.Linear(d_model, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, features)
            mask: (batch, seq_len) - True for padding positions

        Returns:
            predictions: (batch, num_classes)
            confidence: (batch,)
        """
        # Project to d_model
        x = self.input_projection(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Pass through transformer (mask prevents attention to padding)
        x = self.transformer(x, src_key_padding_mask=mask)

        # Use last non-padded timestep for prediction
        if mask is not None:
            # Get last valid position for each sequence
            seq_lengths = (~mask).sum(dim=1) - 1
            last_outputs = x[torch.arange(x.size(0)), seq_lengths]
        else:
            last_outputs = x[:, -1, :]  # Use last timestep

        # Classification
        out = self.fc1(last_outputs)
        out = self.relu(out)
        out = self.dropout(out)
        logits = self.fc2(out)
        predictions = self.softmax(logits)

        confidence, _ = torch.max(predictions, dim=1)

        return predictions, confidence

# Create padding mask for transformer
def create_padding_mask(lengths, max_len):
    """
    Create mask where True indicates padding positions.

    Args:
        lengths: (batch,) actual sequence lengths
        max_len: maximum sequence length in batch

    Returns:
        mask: (batch, max_len) boolean tensor
    """
    batch_size = len(lengths)
    mask = torch.arange(max_len).expand(batch_size, max_len)
    mask = mask >= lengths.unsqueeze(1)
    return mask
```

### Data Collation for Variable-Length Batching

```python
from torch.nn.utils.rnn import pad_sequence

def collate_variable_length(batch):
    """
    Collate function for DataLoader to handle variable-length sequences.

    Args:
        batch: List of samples, each with 'features', 'label', 'seq_length'

    Returns:
        Dictionary with padded tensors and metadata
    """
    # Extract data
    features = [torch.FloatTensor(item['features']) for item in batch]
    labels = torch.LongTensor([item['label'] for item in batch])
    seq_lengths = torch.LongTensor([item['seq_length'] for item in batch])

    # Pad sequences to same length (pads with zeros)
    padded_features = pad_sequence(features, batch_first=True, padding_value=0.0)

    return {
        'features': padded_features,     # (batch, max_len, 11)
        'labels': labels,                # (batch,)
        'seq_lengths': seq_lengths,      # (batch,)
    }

# Usage with DataLoader
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_variable_length
)
```

## Model Evaluation Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| Win rate | >60% | Conservative, achievable |
| Avg profit/trade | >18% | Well above transaction costs |
| Sharpe ratio | >1.5 | Strong risk-adjusted returns |
| Max drawdown | <15% | Acceptable risk level |

## Transaction Speed Optimization

### Priority Fee Calculation

```python
# Get recommended Jito tip using tip floor API
def get_jito_tip():
    response = requests.get("https://bundles.jito.wtf/api/v1/bundles/tip_floor")
    data = response.json()[0]

    # Use 75th percentile for fast execution
    return data['landed_tips_75th_percentile']  # In SOL
```

**Recommended Jito Tip Strategy:**
| Urgency | Percentile | Typical Tip | Use Case |
|---------|------------|-------------|----------|
| Normal | 50th | 0.00001 SOL | Standard trades |
| Fast | 75th | 0.00005 SOL | Time-sensitive |
| Ultra | 95th | 0.0001 SOL | Critical/sniping |

### Compute Unit Optimization

```javascript
// 1. Simulate transaction to get actual CU usage
const simulation = await connection.simulateTransaction(tx);
const actualCUs = simulation.value.unitsConsumed;

// 2. Add 10% buffer
const optimizedCU = Math.ceil(actualCUs * 1.1);

// 3. Set CU limit instruction
const cuLimitIx = ComputeBudgetProgram.setComputeUnitLimit({
  units: optimizedCU  // Instead of default 200,000
});
```

**Typical CU Usage for Swaps:**
| Operation | CU Range |
|-----------|----------|
| Simple transfer | 200-500 |
| Token swap (pump.fun) | 50,000-150,000 |
| Complex DEX swap | 150,000-300,000 |

---

## Direct DEX Integration

For pump.fun tokens, bypass aggregators and interact directly with the bonding curve.

**Pump.fun Bonding Curve:**
```
Program ID: 6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P
```

**Transaction Structure:**
1. Create swap instruction directly to pump.fun program
2. Add compute budget instructions (priority fee)
3. Add Jito tip instruction
4. Sign and send via Jito bundle

**Advantages:**
- No aggregator routing delay
- No aggregator fees
- Predictable execution path
- Optimized for specific token type

---

## Sandwich Attack Protection

```javascript
// Add jitodontfront account for MEV protection
const JITO_DONT_FRONT = new PublicKey("jitodontfront111111111111111111111111111111");

// Include as read-only account in transaction
// This prevents the transaction from being sandwiched
```

---

## SDK References

**Jito SDKs:**
| Language | Repository |
|----------|------------|
| Python | https://github.com/jito-labs/jito-py-rpc |
| JavaScript | https://github.com/jito-labs/jito-js-rpc |
| Rust | https://github.com/jito-labs/jito-rust-rpc |
| Go | https://github.com/jito-labs/jito-go-rpc |

---

# Model Self-Improvement System

## Core Philosophy: Continuous Learning

The AI model MUST NOT operate with static parameters. After each token trading session, the model enters a **self-evaluation phase** to analyze performance and adjust strategy parameters dynamically.

## Post-Trade Evaluation Workflow

**After completing all trades on a token**, execute this workflow:

### 1. Performance Analysis

```python
def evaluate_token_performance(token_trades):
    """
    Analyze all trades executed on a single token.

    Returns:
        performance_metrics: Dict containing win_rate, avg_profit, max_drawdown, etc.
    """
    total_trades = len(token_trades)
    winning_trades = [t for t in token_trades if t['profit'] > 0]
    losing_trades = [t for t in token_trades if t['profit'] <= 0]

    metrics = {
        'win_rate': len(winning_trades) / total_trades,
        'avg_profit': mean([t['profit'] for t in winning_trades]),
        'avg_loss': mean([t['profit'] for t in losing_trades]),
        'total_pnl': sum([t['profit'] for t in token_trades]),
        'max_profit_missed': calculate_missed_opportunities(token_trades),
        'premature_exits': count_premature_exits(token_trades),
        'late_entries': count_late_entries(token_trades),
    }

    return metrics
```

### 2. Parameter Adjustment Strategy

Based on performance metrics, dynamically adjust take-profit and stop-loss thresholds:

| Performance Pattern | Adjustment Action |
|---------------------|-------------------|
| **High win rate (>70%) but low avg profit** | Increase take-profit target by 5-10% |
| **Low win rate (<50%) with high losses** | Tighten stop-loss by 1-2% |
| **Many premature exits** | Increase take-profit, implement trailing stop |
| **Many late entries** | Adjust entry signal sensitivity |
| **Large missed opportunities** | Analyze exit timing, consider multi-stage exits |

**Example Dynamic Adjustment:**
```python
def adjust_parameters(current_params, metrics):
    """
    Dynamically adjust trading parameters based on performance.

    Args:
        current_params: Dict with 'take_profit', 'stop_loss', etc.
        metrics: Performance metrics from evaluation

    Returns:
        adjusted_params: Updated parameters
    """
    adjusted = current_params.copy()

    # Rule 1: High win rate but low profit - increase targets
    if metrics['win_rate'] > 0.70 and metrics['avg_profit'] < 0.20:
        adjusted['take_profit'] *= 1.10  # Increase by 10%
        print(f"Adjusting take-profit: {current_params['take_profit']} -> {adjusted['take_profit']}")

    # Rule 2: Low win rate - tighten stop loss
    if metrics['win_rate'] < 0.50:
        adjusted['stop_loss'] = max(-0.05, adjusted['stop_loss'] * 0.90)  # Tighter stop
        print(f"Tightening stop-loss: {current_params['stop_loss']} -> {adjusted['stop_loss']}")

    # Rule 3: Large missed opportunities - implement staged exits
    if metrics['max_profit_missed'] > 0.30:
        adjusted['use_staged_exits'] = True
        adjusted['exit_stages'] = [0.20, 0.35, 0.50]  # Exit 33% at each level
        print("Enabling staged exits due to missed opportunities")

    # Rule 4: Premature exits detected - add trailing stop
    if metrics['premature_exits'] > 3:
        adjusted['trailing_stop_enabled'] = True
        adjusted['trailing_stop_activation'] = 0.12  # Activate at 12% gain
        adjusted['trailing_stop_distance'] = 0.05   # Trail by 5%
        print("Enabling trailing stop to capture larger gains")

    return adjusted
```

### 3. Learning from Mistakes

**Track and categorize trade failures:**

```python
def analyze_failure_modes(token_trades, price_history):
    """
    Investigate why trades failed and extract learnings.
    """
    failures = []

    for trade in losing_trades:
        failure_type = classify_failure(trade, price_history)
        failures.append({
            'type': failure_type,  # 'early_exit', 'late_entry', 'wrong_signal', etc.
            'loss_amount': trade['profit'],
            'token_characteristics': extract_token_features(trade),
        })

    # Identify patterns in failures
    common_failures = group_by_type(failures)

    return {
        'most_common_failure': max(common_failures, key=lambda x: x['count']),
        'failure_patterns': common_failures,
        'recommended_fixes': generate_recommendations(common_failures)
    }
```

**Common Failure Patterns & Fixes:**

| Failure Type | Cause | Recommended Fix |
|--------------|-------|-----------------|
| **Early Exit** | Hit take-profit too soon, price continued up | Increase take-profit OR use staged exits |
| **Late Entry** | Bought near peak, immediate drawdown | Improve entry signal OR wait for deeper pullback |
| **Whipsaw** | Stopped out then price recovered | Widen stop-loss OR reduce position size |
| **Missed Reversal** | Didn't exit before major dump | Add reversal detection OR tighter trailing stop |

### 4. Rolling Performance Window

Maintain a rolling window of the last 20-50 tokens to track long-term trends:

```python
class ModelPerformanceTracker:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.token_history = deque(maxlen=window_size)
        self.current_params = DEFAULT_PARAMS.copy()

    def update(self, token_metrics):
        """Add new token performance to history."""
        self.token_history.append(token_metrics)

        # Calculate rolling statistics
        rolling_stats = self.calculate_rolling_stats()

        # Adjust parameters if needed
        if self.should_adjust(rolling_stats):
            self.current_params = adjust_parameters(
                self.current_params,
                rolling_stats
            )

    def calculate_rolling_stats(self):
        """Calculate performance over rolling window."""
        return {
            'avg_win_rate': mean([t['win_rate'] for t in self.token_history]),
            'avg_pnl': mean([t['total_pnl'] for t in self.token_history]),
            'sharpe_ratio': calculate_sharpe(self.token_history),
            'consistency': calculate_consistency(self.token_history),
        }

    def should_adjust(self, stats):
        """Determine if parameters need adjustment."""
        # Adjust if performance degrading over 10+ tokens
        if len(self.token_history) >= 10:
            recent_pnl = mean([t['total_pnl'] for t in list(self.token_history)[-10:]])
            older_pnl = mean([t['total_pnl'] for t in list(self.token_history)[-20:-10]])

            # Performance degrading by >20%
            if recent_pnl < older_pnl * 0.80:
                return True

        return False
```

### 5. Parameter Bounds & Safety Limits

**Never allow parameters to drift outside safe ranges:**

```python
PARAMETER_BOUNDS = {
    'take_profit': (0.10, 0.80),      # Min 10%, Max 80%
    'stop_loss': (-0.15, -0.03),      # Min -15%, Max -3%
    'position_size': (0.005, 0.02),   # Min 0.005 SOL, Max 0.02 SOL
    'max_trades_per_token': (2, 8),   # Min 2, Max 8
    'confidence_threshold': (0.60, 0.90),  # Min 60%, Max 90%
}

def enforce_bounds(params):
    """Ensure parameters stay within safe ranges."""
    for key, (min_val, max_val) in PARAMETER_BOUNDS.items():
        if key in params:
            params[key] = max(min_val, min(max_val, params[key]))
    return params
```

## Continuous Improvement Metrics

Track these metrics to measure model improvement over time:

| Metric | Target Trend | Action if Degrading |
|--------|--------------|---------------------|
| Rolling win rate (30 tokens) | Stable or increasing | Retrain model on recent data |
| Average profit per token | Increasing | Continue current strategy |
| Sharpe ratio | >1.5 | Adjust risk parameters |
| Parameter stability | Low variance | Lock in successful parameters |

## Implementation Checklist

After EVERY token trading session:

- [ ] Calculate performance metrics (win rate, PnL, missed opportunities)
- [ ] Identify failure patterns (early exits, late entries, etc.)
- [ ] Adjust take-profit/stop-loss if patterns detected
- [ ] Update rolling performance window
- [ ] Check if parameter adjustment needed
- [ ] Enforce safety bounds on all parameters
- [ ] Log adjustments for audit trail
- [ ] Prepare updated parameters for next token

---

# Risk Management

## Position Limits

- Maximum 5 concurrent trades per token
- Maximum total exposure: 0.05 SOL per token
- No new trades after 3 consecutive losses on same token

## Circuit Breakers

- Stop all trading if total losses exceed 0.5 SOL in 10 minutes
- Halt if win rate drops below 40% over last 20 trades
- Emergency exit all positions if token volume drops >80%

## Risk Management Parameters

```python
# Position sizing
BASE_POSITION = 0.01  # SOL per trade
MAX_TRADES_PER_TOKEN = 5
MAX_POSITION_PER_TOKEN = 0.05  # SOL

# Stop loss / Take profit
STOP_LOSS_PCT = -0.08  # -8%
TAKE_PROFIT_CONSERVATIVE = 0.15  # 15%
TAKE_PROFIT_MODERATE = 0.25  # 25%
TAKE_PROFIT_AGGRESSIVE = 0.40  # 40%

# Trailing stop
TRAILING_STOP_ACTIVATION = 0.10  # Activate at +10%
TRAILING_STOP_DISTANCE = 0.05  # Trail by 5%

# Time limits
MAX_HOLD_TIME = 60  # seconds
INACTIVITY_EXIT = 20  # Exit if no movement for 20s

# Circuit breakers
MAX_LOSS_PER_PERIOD = 0.5  # SOL in 10 minutes
MIN_WIN_RATE = 0.40  # Over last 20 trades
VOLUME_DROP_THRESHOLD = 0.20  # Exit if volume drops to 20%
```

## Risk Assessment

### Market Risks

- **Liquidity risk:** Low liquidity on some tokens may cause high slippage
- **Volatility risk:** Extreme price movements can trigger stop losses
- **Execution risk:** Delays in trade execution can reduce profitability

**Mitigation:** Trade only high-volume tokens, use tight stop losses, optimize execution speed

### Model Risks

- **Overfitting:** Model performs well on historical data but fails live
- **Regime change:** Market conditions change, model becomes less effective
- **Data quality:** Poor quality data leads to bad predictions

**Mitigation:** Cross-validation, walk-forward testing, continuous monitoring and retraining

### Operational Risks

- **System downtime:** Connection loss, server failure
- **Bug risk:** Code errors lead to incorrect trades
- **Fat finger:** Configuration errors cause outsized positions

**Mitigation:** Redundant systems, extensive testing, position limits, kill switches

---

# Implementation Guide

## Implementation Roadmap

### Phase 1: Data Preparation
1. Clean and normalize CSV data
2. Extract features from candles
3. Label training data with optimal actions using Jito fee structure
4. Implement realistic 1-second delay simulation
5. Split data: 80% train, 10% validation, 10% test

### Phase 2: Model Development
1. Implement baseline model (simple momentum strategy)
2. Train CNN-LSTM classifier with historical data only
3. Optimize hyperparameters
4. Backtest on validation set with Jito fee structure
5. Implement self-evaluation and parameter adjustment logic

### Phase 3: Backtesting
1. Simulate trading on test dataset with realistic Jito delays
2. Calculate performance metrics (win rate, Sharpe ratio, etc.)
3. Refine entry/exit logic based on results
4. Test multi-trade scenarios with dynamic parameter adjustment
5. Verify self-improvement system works correctly

### Phase 4: Paper Trading
1. Connect to live token data feed (WebSocket or polling)
2. Execute paper trades in real-time using Jito endpoints
3. Monitor model performance and self-adjustment behavior
4. Fine-tune parameters based on live market conditions
5. Test MEV protection and sandwich attack prevention

### Phase 5: Live Trading
1. Start with 0.01 SOL minimum trades
2. Monitor all trades and log performance
3. Verify continuous self-improvement is working
4. Gradually scale up as confidence increases
5. Continuous model retraining on new data

## Project Structure

```
ai_data/
├── data/
│   ├── tokens_2025-12-21.csv       # Raw data
│   └── processed/
│       ├── features_train.npy       # Training features
│       ├── labels_train.npy         # Training labels
│       └── features_test.npy        # Test features
├── src/
│   ├── prepare_training_data.py     # Clean and prepare data
│   ├── feature_extraction.py        # Extract technical indicators
│   ├── model_cnn_lstm.py           # CNN-LSTM architecture
│   ├── model_training.py           # Training logic
│   ├── backtester.py               # Backtest engine
│   ├── paper_trade.py              # Paper trading
│   └── live_trade.py               # Live trading implementation
├── models/
│   └── best_model.pth              # Saved model weights
└── analyze_tokens.py               # Data analysis script
```

## Monitoring Metrics

Track these in real-time:

### Per-Trade Metrics
- Entry price, exit price, PnL
- Hold time
- Slippage
- Confidence score

### Per-Token Metrics
- Total trades
- Win rate
- Total PnL
- Average hold time

### Portfolio Metrics
- Total PnL (daily, weekly)
- Sharpe ratio
- Maximum drawdown
- Win rate across all trades

---

# Quick Start

## TL;DR - Key Insights

| Metric | Value |
|--------|-------|
| Historical Performance | 328 tokens analyzed |
| Success Rate | 99.4% showed profit opportunities |
| Average max gain | 174.49% |
| Best token | 1247.81% gain |
| Typical profitable window | 60-400 seconds |

## Getting Started

### 1. Prepare Training Data

```bash
cd /home/tr4moryp/script/gmgn_trading/ai_data
python3 prepare_training_data.py
```

This will create:
- `data/processed/features_train.npy` - Training features (OHLCV + indicators)
- `data/processed/labels_train.npy` - Labels (BUY/SELL/HOLD)
- `data/processed/features_test.npy` - Test features

### 2. Train the Model

```bash
python3 train_model.py --epochs 100 --batch-size 32 --learning-rate 0.001
```

Expected training time: 2-4 hours on GPU

### 3. Backtest Strategy

```bash
python3 backtest.py --model models/best_model.pth --test-data data/processed/
```

Look for:
- Win rate > 60%
- Average profit per trade > 15%
- Sharpe ratio > 1.5

### 4. Paper Trade

```bash
python3 paper_trade.py --model models/best_model.pth --live-feed
```

Monitor for 24-48 hours before going live.

### 5. Live Trading

```bash
python3 live_trade.py --model models/best_model.pth --max-position 0.01
```

Start small and scale up gradually.

## Critical Trading Rules

1. **Never trade without stop loss**
2. **Maximum 5 concurrent trades per token**
3. **Exit immediately if volume drops >80%**
4. **Halt trading after 3 consecutive losses on same token**
5. **Maximum position: 0.05 SOL per token**

---

# Appendix

## Expected Performance (WITH JITO INFRASTRUCTURE)

### Fee Impact Summary (Jito)
```
Per 0.01 SOL trade:
  Jito tips (buy + sell): ~0.0001 SOL (2 txs @ 50k lamports each)
  Gas fees (buy + sell): ~0.0004 SOL
  Priority fees (buy + sell): ~0.0002 SOL

  TOTAL COSTS: ~0.0007 SOL (7% of position)
```

### Conservative Estimate (targeting 20% GROSS gains)
- Fixed position size: 0.01 SOL
- Trades per token: 3
- Win rate: 60%
- Average GROSS profit per winning trade: 20%
- Average NET profit per winning trade: 20% - 7% fees = **13%**
- Average loss per losing trade: -8% - 7% fees = **-15%**

Expected value per trade:
```
EV = (0.60 * 0.13) + (0.40 * -0.15) = 0.078 - 0.06 = 0.018 = 1.8%

Per token (3 trades @ 0.01 SOL each):
  Initial: 0.03 SOL
  Expected profit: 0.03 * 0.018 = 0.00054 SOL (1.8% net return per token)

Per day (assuming 50 tokens):
  Total profit: 0.00054 * 50 = 0.027 SOL
```

### Adjusted Strategy (targeting 30% GROSS gains) - RECOMMENDED
- Win rate: 55%
- Average NET profit per winning trade: 30% - 7% = **23%**
- Average loss per losing trade: -8% - 7% = **-15%**

```
EV = (0.55 * 0.23) + (0.45 * -0.15) = 0.1265 - 0.0675 = 0.059 = 5.9%

Per token (3 trades @ 0.01 SOL each):
  Initial: 0.03 SOL
  Expected profit: 0.03 * 0.059 = 0.00177 SOL (5.9% net return per token)

Per day (assuming 50 tokens):
  Total profit: 0.00177 * 50 = 0.0885 SOL (~$17 @ $200/SOL)
```

### Aggressive Estimate (targeting 40% GROSS gains)
- Win rate: 50%
- Average NET profit per winning trade: 40% - 7% = **33%**
- Average loss per losing trade: **-15%**

```
EV = (0.50 * 0.33) + (0.50 * -0.15) = 0.165 - 0.075 = 0.09 = 9.0%

Per token (4 trades @ 0.01 SOL each):
  Initial: 0.04 SOL
  Expected profit: 0.04 * 0.09 = 0.0036 SOL (9.0% net return per token)

Per day (assuming 50 tokens):
  Total profit: 0.0036 * 50 = 0.18 SOL (~$36 @ $200/SOL)
```

## Performance Comparison Table (Jito Infrastructure)

| Strategy | Win Rate | Gross Target | Net Return/Token | Daily Profit (50 tokens) |
|----------|----------|--------------|------------------|--------------------------|
| Conservative | 60% | 20% | 1.8% | 0.027 SOL |
| **Adjusted (Recommended)** | **55%** | **30%** | **5.9%** | **0.0885 SOL** |
| Aggressive | 50% | 40% | 9.0% | 0.18 SOL |

**Key Insight**: Jito infrastructure reduces fees from ~10% to ~7%, making even conservative strategies profitable. The recommended 30% GROSS target strategy offers optimal risk/reward with ~6% net return per token.

## Common Pitfalls to Avoid

1. **Overfitting**: Model works great on backtest, fails live
   - Solution: Use walk-forward validation, cross-validation

2. **Execution delay**: Signals generated too late
   - Solution: Optimize code, use faster WebSocket library

3. **Ignoring transaction costs**: Profitable in theory, loses money in practice
   - Solution: Always include realistic cost model in backtests

4. **No risk management**: One bad trade wipes out 10 good ones
   - Solution: Strict stop losses, position limits

5. **Overtrading**: Too many trades, high costs
   - Solution: Increase confidence threshold, reduce trade frequency

## Key Success Factors

1. **Variable-Length Sequences**: Model sees full price history from discovery, not just fixed 30s window
2. **Jito Infrastructure**: Reduces fees from ~10% to ~7% and confirmation time to <1 second
3. **Higher Profit Targets**: Target 30-40% gains for optimal risk/reward after fees
4. **Model Self-Improvement**: Continuously reevaluate and adjust parameters after each token
5. **Selective Trading**: Only enter high-confidence setups (>70% confidence)
6. **Fast Exits on Losers**: Stop loss at -8% to minimize fee impact on losses
7. **Delay Simulation**: Train model with 1-second worst-case execution prices (Jito-optimized)
8. **Fee-Aware Labeling**: Training labels must account for net profit, not gross
9. **Dynamic Parameter Adjustment**: Adapt take-profit/stop-loss based on real-time performance
10. **Real Market Simulation**: Training mimics exact conditions of watching live charts

## Next Steps

1. ✅ Data analysis completed
2. ✅ Strategy documented
3. ✅ Variable-length training approach designed
4. **TODO:** Build data preprocessing pipeline with full historical context
5. **TODO:** Implement LSTM/Transformer model for variable-length sequences
6. **TODO:** Train model with realistic Jito fees and execution delays
7. **TODO:** Backtest strategy on test dataset
8. **TODO:** Paper trade for validation
9. **TODO:** Scale to live trading

## Future Improvements

1. **Advanced Features**
   - Order book depth analysis
   - Whale wallet tracking
   - Social sentiment analysis
   - Cross-token correlation patterns

2. **Model Enhancements**
   - Ensemble methods (combine multiple models)
   - Transfer learning from new tokens
   - Online learning (update model in real-time)

3. **Strategy Optimization**
   - Dynamic position sizing based on confidence
   - Correlation-based portfolio allocation
   - Market regime detection (high/low volatility modes)

4. **Infrastructure Improvements**
   - Co-locate trading server near Jito block engines
   - Implement redundant submission paths
   - Real-time tip monitoring and adjustment
   - Custom RPC node for lowest latency reads

---

## Conclusion

The historical data shows exceptional profit potential with:
- 99.4% of tokens showing profitable opportunities
- Average 174% maximum gain per token
- Multiple entry/exit points per token

### Realistic Profitability Assessment (Jito Infrastructure)

When accounting for **realistic trading conditions with Jito**:

| Factor | Impact |
|--------|--------|
| Fixed position size | 0.01 SOL per trade |
| Transaction delay (Jito) | 0.3-0.8 sec (~1% slippage) |
| Jito tips + gas + priority fees | ~0.0007 SOL per round trip |
| **Total cost per trade** | **~7% of position value** |

A well-designed AI model with **Jito infrastructure** and **continuous self-improvement** can realistically achieve:
- **30-40%** GROSS profit target per successful trade (dynamically adjusted)
- **23-33%** NET profit after all fees (improved from 20-30% with GMGN)
- 50-55% win rate (higher gains compensate for lower win rate)
- **5.9-9.0%** net return per token (improved from 3-6% with GMGN)

### Critical Success Factors

1. **Variable-Length Sequence Training** (Revolutionary Approach):
   - Model sees FULL price history from token discovery to current time
   - Exactly simulates watching a live trading chart evolve in real-time
   - No artificial fixed-window constraints
   - Enables true lifecycle awareness and context understanding

2. **Jito Infrastructure Benefits**:
   - 3% fee savings vs platform fees (7% vs 10%)
   - Sub-second confirmations (0.3-0.8s vs 1-2s)
   - MEV protection prevents sandwich attacks

3. **Model Self-Improvement**:
   - Reevaluates performance after each token
   - Dynamically adjusts take-profit and stop-loss thresholds
   - Learns from mistakes and adapts to market conditions
   - Maintains rolling performance window for trend detection

4. **Training Methodology**:
   - Trained on **historical data only** (no future peeking)
   - Simulates 1-second transaction delays (Jito-optimized)
   - All profitability calculations include realistic fee structure
   - Ensures deployed model performs as predicted during backtesting

**Expected Daily Performance** (50 tokens/day, 30% GROSS target strategy):
- Daily profit: ~0.0885 SOL (~$17 @ $200/SOL)
- Monthly profit: ~2.655 SOL (~$530 @ $200/SOL)
- Risk-adjusted returns with continuous parameter optimization

---

*Document Version: 3.0 - Variable-Length Sequences & Real Market Simulation*
*Last Updated: December 21, 2025*
*Key Changes:*
- *Revolutionary variable-length sequence approach: model sees full price history from discovery*
- *Replaced fixed 30-second windows with realistic live chart simulation*
- *Added LSTM and Transformer implementations for variable-length sequences*
- *Enhanced training methodology to exactly mimic real trading conditions*
- *Previous: Jito Infrastructure integration, Model Self-Improvement framework*
