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

## Model Architecture Options

### Option A: LSTM (Recommended for Start)

**Architecture:**
```
Input: Variable-length sequences (batch, seq_len, 11 features)
  - seq_len ranges from 30 to 500+ candles

LSTM Layers:
  - 2-3 layers with 128-256 hidden units
  - Batch-first processing
  - Dropout: 0.2-0.3 for regularization
  - Handles variable lengths with pack_padded_sequence

Dense Layers:
  - FC1: hidden_size -> 64
  - ReLU activation
  - Dropout: 0.3
  - FC2: 64 -> 3 (BUY/HOLD/SELL)
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

### Option B: Transformer (For Advanced Users)

**Architecture:**
```
Input: Variable-length sequences (batch, seq_len, 11 features)

Input Projection:
  - Linear: 11 features -> d_model (128)

Positional Encoding:
  - Sinusoidal position embeddings
  - Max length: 1000 timesteps

Transformer Encoder:
  - 4-6 layers
  - 8 attention heads per layer
  - d_model: 128
  - Feedforward dim: 512
  - Dropout: 0.3
  - Attention masks for padding

Classification Head:
  - FC1: d_model -> 64
  - ReLU activation
  - Dropout: 0.3
  - FC2: 64 -> 3 (BUY/HOLD/SELL)
  - Softmax output

Output:
  - Predictions: (batch, 3) - class probabilities
  - Confidence: (batch,) - max probability
```

**Advantages:**
- Better for long-range dependencies
- Parallel processing (faster than LSTM)
- State-of-the-art for sequence tasks
- Attention mechanism reveals important timesteps

---

## Input Features

Each training sample consists of ALL historical data from token discovery to current timestamp.

### Raw OHLCV Data (5 features)
- **Open**: Opening price for the second
- **High**: Highest price in the second
- **Low**: Lowest price in the second
- **Close**: Closing price for the second
- **Volume**: Trading volume in the second

### Technical Indicators (6 features)
- **RSI (14)**: Relative Strength Index (14-period, adjusted for sequence length)
- **MACD**: Moving Average Convergence Divergence (12, 26, 9)
- **Bollinger Upper**: Upper Bollinger Band (20-period)
- **Bollinger Lower**: Lower Bollinger Band (20-period)
- **VWAP**: Volume-Weighted Average Price
- **Momentum**: Rate of Change (10-period)

### Input Shape Specification

**Shape:** `(batch, variable_seq_len, 11)` where:
- `variable_seq_len` = current_timestamp - discovery_timestamp (in seconds)
- Minimum: 30 candles (30 seconds minimum history)
- Maximum: 500+ candles for long-lived tokens

**Example Shapes:**
```
Token at 31s:  (batch, 31, 11)   - Early stage, high volatility expected
Token at 60s:  (batch, 60, 11)   - First minute complete, patterns emerging
Token at 200s: (batch, 200, 11)  - Mature token, model sees full pump history
```

**Critical Advantage**: The sequence length itself is a powerful signal:
- Short sequences (30-60s): Token is fresh, aggressive entry viable
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
MIN_HISTORY_LENGTH = 30     # Minimum candles required
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
        # CRITICAL: Model sees ALL history from discovery to current_time
        historical_data = token_candles[0:current_time]  # Variable length!

        # Extract features from all historical candles
        features = extract_features(historical_data)  # (current_time, 11)

        # Simulate buy execution with realistic delay
        buy_price = get_execution_price(token_candles, current_time, is_buy=True)

        # Look ahead to find optimal sell point (for labeling ONLY)
        future_candles = token_candles[current_time + DELAY_SECONDS:current_time + 20]

        # Find best sell price in next 20 seconds
        best_sell_price = max(c['high'] for c in future_candles)

        # Calculate potential NET profit (after fees)
        net_profit = calculate_net_profit(buy_price, best_sell_price)
        profit_pct = net_profit / FIXED_POSITION_SIZE

        # Determine label based on net profit potential
        if profit_pct > 0.10:      # >10% net profit possible
            label = 1  # BUY
        elif profit_pct < 0.03:    # <3% profit, not worth risk
            label = 0  # HOLD
        else:
            label = 2  # SELL

        samples.append({
            'features': features,           # Shape: (current_time, 11)
            'label': label,                 # 0/1/2
            'seq_length': current_time,     # Actual sequence length
            'timestamp': current_time,
        })

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
- Transformers use attention masks

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
| Win rate | >60% | Conservative, achievable with 30-40% targets |
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

## Comparison: Fixed Window vs Variable-Length

| Aspect | Fixed 30s Window | Variable-Length (This Model) |
|--------|------------------|------------------------------|
| **Realism** | Partial view only | Full market view like real traders |
| **Context** | No lifecycle awareness | Sees if token already pumped 500% |
| **Time awareness** | Needs explicit feature | Implicit from sequence length |
| **Pattern recognition** | Local patterns only | Global trends + local patterns |
| **Training alignment** | Artificial constraint | Matches real trading exactly |
| **Information** | Last 30 seconds | Entire history from discovery |

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
