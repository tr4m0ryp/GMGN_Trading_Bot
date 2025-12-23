# AI Trading Model Design Specification

## Core Philosophy: Variable-Length Sequence Learning

**Revolutionary Approach**: Unlike traditional fixed-window models, this system uses **variable-length sequences** where the model sees ALL historical data from token discovery to the current timestamp, exactly simulating how a real trader watches a live chart evolve.

**Key Innovation:**
- At T=31s: Model sees 31 candles [0-30] - knows token is brand new
- At T=100s: Model sees 100 candles [0-99] - sees if it already pumped 300%
- At T=300s: Model sees 300 candles [0-299] - full lifecycle visible

**Why This is Superior:**
1. **Real Market Simulation**: Exactly mimics watching a live trading chart
2. **Full Context**: Model knows if token already pumped 500% or just starting
3. **Lifecycle Awareness**: Sequence length itself indicates token maturity
4. **Pattern Recognition**: Can identify pump cycles, support/resistance across full history
5. **No Artificial Constraints**: Removes the arbitrary limitation of fixed windows

---

## Model Architecture: LSTM


**Architecture:**
```
Input: Variable-length sequences (batch, seq_len, 15 features)
  - seq_len ranges from 12 to 500+ candles

LSTM Layers:
  - 2-3 layers with 128-256 hidden units
  - Batch-first processing
  - Dropout: 0.2-0.3 for regularization
  - Handles variable lengths with pack_padded_sequence

Dense Layers:
  - FC1: hidden_size -> 64
  - ReLU activation
  - Dropout: 0.3
  - FC2: 64 -> 3 (0=HOLD, 1=BUY, 2=SELL)
  - Softmax output

Output:
  - Predictions: (batch, 3) - class probabilities
  - Confidence: (batch,) - max probability
```

**Advantages:**
- Handles variable-length sequences natively
- Good for temporal dependencies
- Proven architecture for sequence modeling
- Efficient training on GPU


---

## Input Features

Each training sample consists of ALL historical data from token discovery to current timestamp.

### Feature Set (15 total)
- **log_close**
- **ret_1s**, **ret_3s**, **ret_5s** (log returns)
- **range_ratio**: (high-low)/close
- **volume_log**: log1p(volume)
- **rsi_norm**
- **macd_norm**: macd/close
- **bb_upper_dev**, **bb_lower_dev**: (band-close)/close
- **vwap_dev**: (vwap-close)/close
- **momentum_10**
- **indicator_ready_short/long**: masks for early-window reliability
- **in_position_flag**: 0 for entry context, 1 for exit context

### Input Shape Specification

**Shape:** `(batch, variable_seq_len, 15)` where:
- `variable_seq_len` = current_timestamp - discovery_timestamp (in seconds)
- Minimum: 12 candles (act earlier on fresh launches)
- Maximum: 500+ candles for long-lived tokens

**Example Shapes:**
```
Token at 31s:  (batch, 31, 11)   - Early stage, high volatility expected
Token at 60s:  (batch, 60, 11)   - First minute complete, patterns emerging
Token at 200s: (batch, 200, 11)  - Mature token, model sees full pump history
```

**Critical Advantage**: The sequence length itself is a powerful signal:
- Short sequences (12-60s): Token is fresh, aggressive entry viable
- Medium sequences (60-180s): Prime trading window, multiple cycles visible
- Long sequences (180s+): Token maturing, model sees if it's dying or continuing

---

## Model Output

### Classification
Binary classification for each input:
- **HOLD (0)**: No action - wait for better opportunity
- **BUY (1)**: Enter position (if no position) or add to position
- **SELL (2)**: Exit position (if position exists)

### Confidence Score
- Range: [0.0, 1.0]
- Represents maximum class probability
- **Only execute trades when confidence > 0.7**

### Output Format
```python
predictions: torch.Tensor  # Shape: (batch, 3) - softmax probabilities
confidence: torch.Tensor   # Shape: (batch,) - max(predictions, dim=1)
```

---

## Training Data Preparation

### Constants (Jito-Optimized)
```python
FIXED_POSITION_SIZE = 0.01  # SOL per trade
DELAY_SECONDS = 1           # Jito transaction confirmation delay
JITO_TIP_AVG = 0.00005      # Average Jito tip (50,000 lamports)
GAS_FEE_FIXED = 0.0002      # Solana gas fee
PRIORITY_FEE = 0.0001       # Priority fee for compute units
TOTAL_FEE_PER_TX = 0.00035  # Total fee per transaction
MIN_HISTORY_LENGTH = 12     # Minimum candles required (act early)
```

### Realistic Execution Simulation

**Buy Orders:**
```python
def get_execution_price(candles, start_idx, is_buy=True):
    """Simulate 1-second delay with worst-case pricing."""
    delay_window = candles[start_idx:start_idx + DELAY_SECONDS + 1]

    if is_buy:
        return max(c['high'] for c in delay_window)  # Worst case slippage
    else:
        return min(c['low'] for c in delay_window)   # Worst case slippage
```

**Profit Calculation:**
```python
def calculate_net_profit(buy_price, sell_price):
    """Calculate profit after all Jito fees."""
    tokens = FIXED_POSITION_SIZE / buy_price
    sell_value = tokens * sell_price
    net_value = sell_value - (2 * TOTAL_FEE_PER_TX)  # Buy + Sell fees
    return net_value - FIXED_POSITION_SIZE
```

### Sample Generation (Full Historical Context)

```python
def prepare_realistic_training_data(token_candles):
    """
    Generate training samples with full historical context.

    Each sample contains ALL price history from token discovery
    to current timestamp, simulating real market observation.
    """
    samples = []

for current_time in range(MIN_HISTORY_LENGTH, len(token_candles)):
    historical_data = token_candles[0:current_time]

    # Two contexts: flat (entry) and in-position (exit)
    features_flat = extract_features(historical_data, in_position=False)  # (current_time, 15)
    features_in_pos = extract_features(historical_data, in_position=True) # (current_time, 15)

    buy_price = get_execution_price(token_candles, current_time, is_buy=True)
    future_candles = token_candles[current_time + DELAY_SECONDS:current_time + LOOKAHEAD_SECONDS]
    max_future_high = max(c['high'] for c in future_candles)
    min_future_low = min(c['low'] for c in future_candles)
    end_close = future_candles[-1]['close']

    profit_pct = calculate_net_profit(buy_price, max_future_high) / FIXED_POSITION_SIZE
    drawdown_pct = (min_future_low - buy_price) / buy_price
    peak_gain_pct = (max_future_high - buy_price) / buy_price
    rolled_over = peak_gain_pct >= TAKE_PROFIT_PCT and end_close <= max_future_high * (1 - TRAIL_BACKOFF_PCT)

    # Flat context: BUY only if upside > TP and drawdown above SL
    label_flat = 1 if profit_pct >= TAKE_PROFIT_PCT and drawdown_pct >= STOP_LOSS_PCT else 0

    # In-position context: SELL on stop or rollover
    label_in_pos = 2 if (drawdown_pct <= STOP_LOSS_PCT or rolled_over) else 0

    samples.append({'features': features_flat, 'label': label_flat, 'seq_length': current_time, 'timestamp': current_time})
    samples.append({'features': features_in_pos, 'label': label_in_pos, 'seq_length': current_time, 'timestamp': current_time})

    return samples
```

---

## Key Training Principles

### 1. Full Historical Context
Model sees ALL candles from discovery to current time:
- At 31s: sees 31 candles [0-30]
- At 100s: sees 100 candles [0-99]
- At 300s: sees 300 candles [0-299]

### 2. Real Market Simulation
Exactly mimics watching a live chart evolve:
- No artificial fixed-window constraints
- Model knows token age implicitly from sequence length
- Can identify if token already pumped 500% vs just starting

### 3. Realistic Execution
Simulates 1-second delay with worst-case pricing:
- Buy orders: highest price in delay window (worst slippage)
- Sell orders: lowest price in delay window (worst slippage)

### 4. Fee-Aware Labels
All labels based on NET profit potential:
- Accounts for Jito tips, gas fees, priority fees (~7% total)
- BUY only when >10% net profit possible
- Ensures trained model is profitable in real trading

### 5. Variable-Length Sequences
Each sample has different length:
- Requires padding/masking for batching
- LSTM handles natively with pack_padded_sequence

---

## Data Batching for Variable Lengths

### Collate Function
```python
from torch.nn.utils.rnn import pad_sequence

def collate_variable_length(batch):
    """Collate function for DataLoader."""
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
```

### DataLoader Setup
```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_variable_length,
    num_workers=4,
)
```

---

## Model Evaluation Metrics

### Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| Win rate | >80% | Conservative, achievable with 30-40% targets |
| Avg profit/trade | >18% NET | Well above transaction costs (~7%) |
| Sharpe ratio | >1.5 | Strong risk-adjusted returns |
| Max drawdown | <15% | Acceptable risk level |
| Confidence accuracy | >70% | High-confidence predictions are reliable |

### Backtest Requirements

1. **Realistic Fee Simulation**: Include all Jito fees (~7% per round trip)
2. **Execution Delay**: Simulate 1-second confirmation delay
3. **Worst-Case Slippage**: Use highest/lowest prices in delay window
4. **Walk-Forward Testing**: Test on future data, not training data
5. **Multiple Tokens**: Evaluate across diverse token behaviors

---




---

## Model Self-Improvement System

### Core Philosophy: Continuous Learning

The AI model MUST NOT operate with static parameters. After each token trading session, the model enters a **self-evaluation phase** to analyze performance and adjust strategy parameters dynamically.

### Post-Trade Evaluation Workflow

**After completing all trades on a token**, execute this workflow:

#### 1. Performance Analysis

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

#### 2. Parameter Adjustment Strategy

Based on performance metrics, dynamically adjust take-profit and stop-loss thresholds:

| Performance Pattern | Adjustment Action |
|---------------------|-------------------|
| **High win rate (>70%) but low avg profit** | Increase take-profit target by 5-10% |
| **Low win rate (<50%) with high losses** | Tighten stop-loss by 1-2% |
| **Many premature exits** | Increase take-profit, implement trailing stop |
| **Many late entries** | Adjust entry signal sensitivity |
| **Large missed opportunities** | Analyze exit timing, consider multi-stage exits |

#### 3. Learning from Mistakes

**Track and categorize trade failures:**

| Failure Type | Cause | Recommended Fix |
|--------------|-------|-----------------|
| **Early Exit** | Hit take-profit too soon, price continued up | Increase take-profit OR use staged exits |
| **Late Entry** | Bought near peak, immediate drawdown | Improve entry signal OR wait for deeper pullback |
| **Whipsaw** | Stopped out then price recovered | Widen stop-loss OR reduce position size |
| **Missed Reversal** | Didn't exit before major dump | Add reversal detection OR tighter trailing stop |

#### 4. Rolling Performance Window

Maintain a rolling window of the last 20-50 tokens to track long-term trends. If performance degrades over 10+ tokens (e.g., recent PnL < 80% of older PnL), trigger a parameter review or model retraining.

#### 5. Parameter Bounds & Safety Limits

**Never allow parameters to drift outside safe ranges:**

```python
PARAMETER_BOUNDS = {
    'take_profit': (0.10, 0.80),      # Min 10%, Max 80%
    'stop_loss': (-0.15, -0.03),      # Min -15%, Max -3%
    'position_size': (0.005, 0.02),   # Min 0.005 SOL, Max 0.02 SOL
    'max_trades_per_token': (2, 8),   # Min 2, Max 8
    'confidence_threshold': (0.60, 0.90),  # Min 60%, Max 90%
}
```

---

## Implementation Checklist

- [ ] Implement variable-length data preparation
- [ ] Extract features with technical indicators
- [ ] Create LSTM model with pack_padded_sequence
- [ ] Implement custom collate function
- [ ] Set up DataLoader with proper batching
- [ ] Implement training loop with validation
- [ ] Add early stopping and checkpointing
- [ ] Create backtesting engine with realistic fees
- [ ] Evaluate on test set
- [ ] Calculate Sharpe ratio and max drawdown

---

*Model Design Version: 1.0*
*Last Updated: December 21, 2025*
