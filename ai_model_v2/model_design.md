# Solana Memecoin Trading - Multi-Model Architecture

## 1. Project Overview

This project develops a machine learning system for trading Solana memecoins. We use the `logger_c` component to track filtered tokens with:
- Market cap between $5.5k-$20k
- Detected within 10 minutes of launch
- At least 1 KOL (Key Opinion Leader) involvement

Our training data is sourced from `ai_data/combined.csv`, containing chart data for filtered memecoins until inactivity (low liquidity, volume death, rug pull, etc.).

---

## 2. Data Analysis Summary

### 2.1 Token Lifecycle Statistics

Our dataset contains ~956 Solana memecoins with extremely short lifecycles:

| Metric | Value |
|--------|-------|
| Average lifespan | ~220 seconds (3.7 min) |
| Median lifespan | ~172 seconds |
| Median time to peak | ~76 seconds (1.3 min) |
| 90th percentile time to peak | ~4.4 min |

### 2.2 Price Performance Distribution

| Metric | Peak/Entry Ratio |
|--------|------------------|
| Median | 2.35x |
| 75th percentile | 3.5x |
| 90th percentile | 5.0x |
| Top 1% outliers | >12x |
| Maximum observed | ~30x |

### 2.3 Success Rates by Entry Market Cap

| Entry MC Range | Tokens that 2x | Tokens that 4x |
|----------------|----------------|----------------|
| < $4k | 82% | 30% |
| $5k-$10k | ~70% | ~20% |
| > $10k | 31% | 10% |

**Key Insight**: Lower entry MC correlates with higher success rates, but also higher risk.

### 2.4 Death Flag Analysis

| Death Reason | Count | 4x+ Rate |
|--------------|-------|----------|
| price_stable | 721 | 21% |
| volume_low | 178 | 9% |
| candle_gap | 34 | varies |

Tokens dying from "price_stable" (natural fade-out) achieved higher peaks than those dying from "volume_low" (liquidity crash).

---

## 3. Winner vs Loser Patterns

### 3.1 Price Action Signatures

**Winners (top ~18% by peak):**
- Continuous parabolic acceleration after launch
- Price reaches ~2.3x by 30s, ~3.0x by 60s, ~3.3x by 2 min
- Modest pullbacks with higher lows
- Sustained momentum

**Losers:**
- Early spike followed by stall
- Peak at ~1.14x by 7s, then flatten/decline
- Reach only ~1.25x at 60s
- Never recover above early peaks

### 3.2 Volume Signatures

**Winners:**
- First-minute volume: ~16.9k units (average)
- Volume spread evenly (~35% in first minute)
- Sustained trading beyond 60s
- Gradual volume bursts (FOMO-driven)

**Losers:**
- First-minute volume: ~8.8k units (average)
- >65% of total volume in first minute
- Rapid spike then dry-up
- High first-second volume often signals snipe/dump

### 3.3 Failure Indicators

- Initial sniper buy spikes price, then immediate volume evaporation
- Price drops >10% within seconds after peak
- Zero volume bars within first minute
- Sharp reversals (>30% drop in few seconds)
- Never surpassing ~1.2x entry

---

## 4. Multi-Model Trading Architecture

We propose a **three-model ensemble** where each model specializes in a distinct decision:

```
                    +------------------+
                    |   Token Stream   |
                    +--------+---------+
                             |
                             v
                 +-----------+-----------+
                 |   MODEL 1: SCREENER   |
                 |   "Should we watch?"  |
                 +-----------+-----------+
                             |
                    [PASS]   |   [REJECT] --> Skip token
                             v
                 +-----------+-----------+
                 |   MODEL 2: ENTRY      |
                 |   "When to enter?"    |
                 +-----------+-----------+
                             |
                    [ENTER]  |   [WAIT] --> Continue monitoring
                             v
                 +-----------+-----------+
                 |   MODEL 3: EXIT       |
                 |   "When to exit?"     |
                 +-----------+-----------+
                             |
                             v
                    +--------+---------+
                    |   Execute Trade  |
                    +------------------+
```

---

## 5. Model 1: Entry Worthiness Screener

### 5.1 Objective

Binary classification: Determine if a token is **worth monitoring** for potential entry.

**Labels:**
- `WORTHY` (1): Token has potential to reach >2x from current state
- `AVOID` (0): Token is likely a rug, scam, or dead-on-arrival

### 5.2 Input Features (Static + Early Dynamic)

**Static Features (available at detection):**
- Initial market cap bin: [<5k, 5-10k, 10-15k, 15-20k, >20k]
- KOL count and influence score
- Token age at detection (seconds since launch)
- Liquidity pool size (if available)
- Holder count at detection
- Top 10 wallet concentration (Gini coefficient)
- DEX platform (pump.fun, Raydium, etc.)

**Early Dynamic Features (first 10-30s):**
- Price return: 5s, 10s, 15s, 30s
- Volume in first 10s, 20s, 30s
- Transaction count in first 30s
- Buy/sell ratio in first 30s
- Largest single transaction size
- Volume acceleration rate

### 5.3 Architecture Options

| Model Type | Pros | Cons |
|------------|------|------|
| XGBoost/LightGBM | Fast inference, interpretable, robust on small data | May miss temporal patterns |
| Random Forest | Good generalization, feature importance | Slower inference |
| Logistic Regression | Very fast, baseline | Limited pattern capture |
| LSTM/GRU | Captures temporal dependencies | Needs >10k samples, slower |
| Transformer | Attention mechanisms | Overkill for short sequences |

### 5.4 Architecture Selection Analysis

Based on actual dataset characteristics:

```
Dataset size:         950 tokens (SMALL - favors tree models)
Median sequence:      104 datapoints (SHORT - limited temporal patterns)
Class distribution:   81% WORTHY, 19% AVOID (inverse imbalance)
Time to peak:         76s median (FAST - need quick inference)
```

**Why XGBoost/LightGBM is optimal:**

1. **Sample size constraint**: 950 samples is insufficient for neural networks
   which typically need 10,000+ samples for proper generalization. Tree-based
   models with regularization handle small datasets effectively.

2. **Sequence length**: With 104 datapoints over ~3.7 min average lifespan,
   the temporal patterns are limited. Engineered features (momentum, volume
   acceleration) capture this better than raw sequence models.

3. **Inverse class imbalance**: AVOID is the minority class (19%). XGBoost
   handles this via `scale_pos_weight` parameter more reliably than neural
   networks which tend to collapse to majority class.

4. **Inference latency**: Median time to peak is 76s. Model must run in <100ms.
   - XGBoost: ~0.1-1ms per inference
   - LSTM: ~10-50ms per inference
   - Transformer: ~20-100ms per inference

5. **Debugging**: Tree models provide feature importance, making it easy to
   understand why a token was classified as WORTHY/AVOID.

**Recommended**: XGBoost with:
- `scale_pos_weight = 4.0` (weight AVOID class higher since it's minority)
- `max_depth = 6` (prevent overfitting on small data)
- `early_stopping_rounds = 50`
- `learning_rate = 0.05`

### 5.5 Training Labels

```
WORTHY = 1 if max(price_ratio) >= 2.0 within 5 minutes
AVOID  = 0 otherwise
```

### 5.6 Evaluation Metrics

- Primary: Precision @ high recall (we want to catch most winners)
- Secondary: ROC-AUC, F1-score
- Business metric: % of passed tokens that achieve 2x+

---

## 6. Model 2: Entry Timing Optimizer

### 6.1 Objective

Given a token passed by Model 1, determine the **optimal moment to enter**.

**Output Options:**
- **Classification**: `ENTER_NOW`, `WAIT`, `ABORT`
- **Regression**: Predicted price at t+30s, t+60s (enter if positive)

### 6.2 Input Features (Continuous Monitoring)

**Time-Series Features (rolling window):**
- Price sequence: last 30-60 seconds
- Volume sequence: last 30-60 seconds
- Order flow imbalance (buy volume - sell volume)
- Momentum indicators: 10s/30s price ROC
- Short-term RSI (14-period on 1s candles)
- Volume-weighted price velocity

**Pattern Features:**
- Current drawdown from recent high
- Consolidation detection (price range compression)
- Higher-low formation (bullish structure)
- Volume surge detection (>2x average)

**Context Features:**
- Time since launch
- Current price vs entry MC
- Cumulative volume
- Model 1 confidence score (pass-through)

### 6.3 Architecture Options

| Model Type | Pros | Cons |
|------------|------|------|
| LSTM/GRU | Captures temporal dependencies | Slower training, needs more data |
| Transformer | Attention on key moments | Complex, may overfit |
| 1D-CNN | Fast, captures local patterns | May miss long-range dependencies |
| XGBoost + Features | Robust, interpretable | Requires manual feature engineering |

**Recommended**: Hybrid approach
1. LSTM/1D-CNN for sequence encoding
2. Concatenate with static features
3. Dense layers for final decision

### 6.4 Training Labels

```
ENTER_NOW = 1 if:
    - price increases >20% in next 60 seconds
    - AND max drawdown in next 60s < 15%

WAIT = 1 if:
    - current price > 90% of local peak (avoid buying tops)
    - OR volume declining (wait for confirmation)

ABORT = 1 if:
    - price dropped >20% from peak already
    - OR volume collapsed (>80% drop from peak volume)
```

### 6.5 Inference Strategy

- Run inference every 5 seconds after Model 1 passes token
- Confidence threshold for ENTER_NOW: >0.7
- Maximum wait time: 3 minutes (abort if no entry signal)

---

## 7. Model 3: Exit Point Optimizer

### 7.1 Objective

Once entered, determine the **optimal exit point** to maximize profit while limiting downside.

**Output Options:**
- **Classification**: `EXIT_NOW`, `HOLD`, `PARTIAL_EXIT`
- **Regression**: Predicted max price in next N seconds

### 7.2 Input Features (Post-Entry Monitoring)

**Position Context:**
- Entry price and current price
- Current unrealized P&L %
- Time since entry
- Entry conditions (Model 2 confidence)

**Time-Series Features (from entry):**
- Price trajectory since entry
- Volume trajectory since entry
- Momentum divergence (price up, volume down = bearish)
- Higher-high / lower-low sequence

**Exhaustion Indicators:**
- RSI extreme readings (>80 = overbought)
- Volume exhaustion (declining on price increases)
- Parabolic extension (price >> moving average)
- Time since last higher-high

**Risk Indicators:**
- Sudden volume spike (potential whale exit)
- Price velocity deceleration
- Bid/ask spread widening (if available)
- Large sell orders appearing

### 7.3 Exit Strategy Labels

```
EXIT_NOW = 1 if:
    - Price drops >10% from position high
    - OR volume collapses (likely top is in)
    - OR unrealized gain >50% AND momentum slowing

PARTIAL_EXIT = 1 if:
    - Unrealized gain >30% AND momentum still positive
    - Take 50% profit, let rest ride

HOLD = 1 if:
    - Still making higher highs
    - Volume sustained
    - Unrealized gain < target threshold
```

### 7.4 Architecture Options

| Model Type | Pros | Cons |
|------------|------|------|
| XGBoost | Fast decisions, interpretable | May miss subtle patterns |
| LSTM | Good for sequence-to-decision | Latency concerns |
| Rule-based hybrid | Guaranteed risk limits | May miss optimal exits |

**Recommended**: Ensemble approach
1. ML model predicts optimal exit probability
2. Hard rules for risk management (stop-loss, trailing stop)
3. Combine: ML suggests, rules enforce limits

### 7.5 Risk Management Rules (Hard-Coded)

These override ML predictions:

```
STOP_LOSS:      Exit if unrealized loss > 25%
TRAILING_STOP:  Exit if price drops > 15% from position high
TIME_STOP:      Exit after 5 minutes regardless (unless >100% gain)
PROFIT_TARGET:  Exit at 200% gain (take profits)
```

---

## 8. Feature Engineering Specifications

### 8.1 Price-Based Features

```python
# Returns at various windows
return_5s  = (price_now - price_5s_ago) / price_5s_ago
return_10s = (price_now - price_10s_ago) / price_10s_ago
return_30s = (price_now - price_30s_ago) / price_30s_ago
return_60s = (price_now - price_60s_ago) / price_60s_ago

# Momentum
momentum_10s = return_10s - return_10s_previous
price_acceleration = (return_10s - return_20s) / 10

# Volatility
volatility_30s = std(returns_1s[-30:])
volatility_60s = std(returns_1s[-60:])

# Trend
ema_10s = EMA(prices, span=10)
ema_30s = EMA(prices, span=30)
trend_strength = (ema_10s - ema_30s) / ema_30s

# Drawdown
rolling_high = max(prices[-30:])
drawdown = (price_now - rolling_high) / rolling_high
```

### 8.2 Volume-Based Features

```python
# Raw volume
volume_10s = sum(volumes[-10:])
volume_30s = sum(volumes[-30:])
volume_60s = sum(volumes[-60:])

# Volume rate
volume_rate = volume_30s / 30  # per second
volume_acceleration = (volume_rate_now - volume_rate_10s_ago) / 10

# Volume distribution
volume_first_minute_pct = volume_60s / total_volume

# Buy/Sell pressure (if available)
buy_volume = sum(buy_volumes[-30:])
sell_volume = sum(sell_volumes[-30:])
order_imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)
```

### 8.3 Market Cap Features

```python
# MC estimation
estimated_mc = current_price * total_supply

# MC bins (one-hot)
mc_bin_sub_5k   = 1 if estimated_mc < 5000 else 0
mc_bin_5k_10k   = 1 if 5000 <= estimated_mc < 10000 else 0
mc_bin_10k_15k  = 1 if 10000 <= estimated_mc < 15000 else 0
mc_bin_15k_20k  = 1 if 15000 <= estimated_mc < 20000 else 0
mc_bin_above_20k = 1 if estimated_mc >= 20000 else 0

# MC velocity
mc_growth_rate = (estimated_mc - mc_30s_ago) / mc_30s_ago
```

### 8.4 Technical Indicators

```python
# RSI (short window)
rsi_14 = calculate_rsi(prices, period=14)

# Stochastic
stoch_k, stoch_d = calculate_stochastic(prices, period=14)

# Bollinger position
bb_upper, bb_middle, bb_lower = calculate_bollinger(prices, period=20)
bb_position = (price_now - bb_lower) / (bb_upper - bb_lower)
```

---

## 9. Training Pipeline

### 9.1 Data Preparation

```
1. Load combined.csv
2. Parse token timeseries (price, volume per second)
3. Compute all features at each timestamp
4. Generate labels based on future outcomes
5. Split by time (chronological, no shuffle)
   - Train: first 70% of tokens
   - Validation: next 15%
   - Test: final 15%
```

### 9.2 Training Strategy

**Model 1 (Screener):**
- Train on features at t=30s after detection
- Label based on max price in next 5 min
- Class weights: {AVOID: 1, WORTHY: 4} (handle imbalance)

**Model 2 (Entry):**
- Train only on WORTHY tokens (filtered by Model 1)
- Generate training samples every 5s
- Label based on price outcome in next 60s

**Model 3 (Exit):**
- Train only on entered positions (simulated)
- Generate samples every 5s after simulated entry
- Label based on optimal exit point (hindsight)

### 9.3 Cross-Validation

Use **time-series cross-validation** (walk-forward):

```
Fold 1: Train [0-50%],  Validate [50-60%]
Fold 2: Train [0-60%],  Validate [60-70%]
Fold 3: Train [0-70%],  Validate [70-80%]
Fold 4: Train [0-80%],  Validate [80-90%]
Fold 5: Train [0-90%],  Validate [90-100%]
```

---

## 10. Evaluation Framework

### 10.1 Model-Level Metrics

| Model | Primary Metric | Secondary Metrics |
|-------|----------------|-------------------|
| Screener | Recall@Precision>0.6 | ROC-AUC, F1 |
| Entry | Precision | Win rate, avg gain on entry |
| Exit | P&L vs buy-and-hold | Max drawdown, Sharpe |

### 10.2 System-Level Backtest

Simulate full trading pipeline:

```
For each test token:
    1. Run Screener at t=30s
    2. If WORTHY, monitor for Entry signal (max 3 min)
    3. If ENTER, monitor for Exit signal
    4. Record: entry_price, exit_price, duration, P&L
```

**Business Metrics:**
- Win rate (% of trades with positive P&L)
- Average win / average loss ratio
- Total return over test period
- Maximum drawdown
- Sharpe ratio
- Number of trades per day

**Target Performance:**
- Win rate: >60%
- Avg win/loss ratio: >2.0
- Max drawdown: <30%

---

## 11. Implementation Roadmap

### Phase 1: Data Infrastructure
- [ ] Parse and validate combined.csv
- [ ] Build feature computation pipeline
- [ ] Create training/validation/test splits
- [ ] Implement label generation functions

### Phase 2: Model 1 Development
- [ ] Train baseline XGBoost screener
- [ ] Tune hyperparameters
- [ ] Evaluate on test set
- [ ] Set confidence thresholds

### Phase 3: Model 2 Development
- [ ] Build LSTM/hybrid entry model
- [ ] Train on screener-passed tokens
- [ ] Optimize entry timing accuracy
- [ ] Integrate with Model 1

### Phase 4: Model 3 Development
- [ ] Develop exit prediction model
- [ ] Implement risk management rules
- [ ] Combine ML predictions with hard stops
- [ ] Full pipeline integration

### Phase 5: Backtesting and Optimization
- [ ] Run full system backtest
- [ ] Analyze failure modes
- [ ] Tune model thresholds
- [ ] Stress test on edge cases

### Phase 6: Production Deployment
- [ ] Real-time feature computation
- [ ] Model serving infrastructure
- [ ] Integration with trading execution
- [ ] Monitoring and logging

---

## 12. Risk Considerations

### 12.1 Model Risks

- **Overfitting**: Small dataset (~956 tokens) risks overfit; use regularization
- **Regime change**: Memecoin market dynamics may shift; retrain periodically
- **Latency**: Model inference must complete in <100ms for real-time trading
- **Class imbalance**: Only ~18% of tokens are "winners"; use appropriate techniques

### 12.2 Trading Risks

- **Slippage**: Low liquidity means actual execution may differ from model assumptions
- **Front-running**: Bots may detect and front-run our trades
- **Rug pulls**: Some losses are unavoidable (dev rugs instantly)
- **Gas/fees**: Transaction costs eat into profits on small trades

### 12.3 Mitigation Strategies

- Position sizing: Never risk >5% of portfolio on single trade
- Diversification: Trade multiple tokens simultaneously
- Hard stops: Always enforce stop-loss regardless of model output
- Continuous monitoring: Track model performance, pause if degrading

---

## 13. Appendix: Key Statistics Reference

| Metric | Winners | Losers |
|--------|---------|--------|
| First-minute volume | 16.9k | 8.8k |
| First-minute volume % of total | 35% | 65% |
| Price at 30s | 2.3x | 1.14x |
| Price at 60s | 3.0x | 1.25x |
| Median peak ratio | >4x | <2x |
| Typical time to peak | 60-120s | <30s |


## 14. Extra details
we are using google colab to train the model on. therbey we nee some manage notebooks. you should follow the strucutre being used in  @ai_model/notebooks. This allows easily acces to every source.
Google colab allows us acces to high end gpu with alot of vram.

