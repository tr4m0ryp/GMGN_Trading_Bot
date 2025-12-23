# Session Summary: Model Improvements & Diagnostics

**Date:** 2025-12-22
**Focus:** Fix supervised learning approach with threshold adjustments and class balancing

---

## What We Accomplished

### 1. Comprehensive Improvement Plan ✓

**File:** [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md)

Identified root causes of training failure:
- Flawed label logic (3-10% profit → SELL was confusing)
- Severe class imbalance (BUY only 6.4%)
- Feature normalization issues
- Need for diagnostics

### 2. Fixed Feature Normalization ✓

**File:** [preparation.py:286-288](../src/data/preparation.py#L286-L288)

**Before:**
```python
log_close = np.log(np.clip(closes, 1e-8, None))  # Varies by token price
```

**After:**
```python
log_close = np.log(np.clip(closes, 1e-8, None)) - np.log(np.clip(closes[0], 1e-8, None))
# Relative to first candle - prevents scale variation
```

### 3. Added Diagnostic Utilities ✓

**File:** [train.py:35-187](../src/training/train.py#L35-L187)

Three diagnostic functions:
1. `sanity_check_overfit_batch()` - Test if model can learn (target: >80% on 1 batch)
2. `analyze_predictions()` - Detect class collapse
3. `diagnose_training_setup()` - Full pre-training diagnostics

### 4. Created Diagnostic Script ✓

**File:** [diagnose.py](../src/diagnose.py)

```bash
cd ai_model/src
python diagnose.py
```

Runs comprehensive checks before full training.

### 5. Implemented Attention Mechanism ✓

**File:** [attention_lstm.py](../src/models/attention_lstm.py)

New `AttentionLSTMTrader` model with self-attention:
- Learns which timesteps matter most
- Can visualize attention weights
- Same interface as baseline LSTM

### 6. Adjusted Profit Thresholds ✓

**File:** [config/__init__.py](../src/config/__init__.py)

| Threshold | Old | New | Reason |
|-----------|-----|-----|--------|
| TAKE_PROFIT_PCT | 8% | **5%** | Increase BUY samples |
| STOP_LOSS_PCT | -4% | **-6%** | More tolerant |
| TRAIL_BACKOFF_PCT | 2% | **3%** | More rollover tolerance |

**Result:**
- BUY class: **6.4% → 9.9%** (54% increase!)
- Mean BUY profit: 23.37% (still good quality)

### 7. Implemented Weighted Sampler ✓

**File:** [train.py:381-416](../src/training/train.py#L381-L416)

```python
def create_weighted_sampler(dataset, num_classes=3):
    """Oversample minority classes for balanced batches."""
```

**Config:** `'use_weighted_sampler': True`

**Effect:**
- Batch distribution: 31% HOLD, 38% BUY, 31% SELL (balanced!)
- Was: 53% HOLD, 10% BUY, 37% SELL (imbalanced)

### 8. Re-preprocessed Data ✓

**New statistics:**
```
Total samples: 74,086
- Train: 59,628 (260 tokens)
- Val:   7,112 (32 tokens)
- Test:  7,346 (34 tokens)

Class distribution (train):
- HOLD: 53.5%
- BUY:  9.9%  ← UP from 6.4%
- SELL: 36.6%

BUY signal quality:
- Mean profit: 23.37%
- Min profit:  5.00%
- Max profit:  322.09%
```

### 9. Verified Model Architecture Works ✓

**File:** [test_simple.py](../src/test_simple.py)

Tested both basic LSTM and our architecture on simple patterns:
```
Basic LSTM:        PASS (100% acc)
Our architecture:  PASS (100% acc)
```

**Conclusion:** Model is fine. Issue is in trading data setup.

---

## Current Status

### What Works ✅

1. **Data preprocessing** - Clean, no NaN/Inf
2. **Feature engineering** - 15 features, normalized correctly
3. **Model architecture** - Can learn simple patterns perfectly
4. **Weighted sampler** - Balanced batches
5. **Class distribution** - Improved from 6.4% to 9.9% BUY
6. **Focal loss** - Configured and ready
7. **Diagnostics** - Comprehensive testing suite

### What's Still Broken ❌

**Diagnostic results:**
```
[Labels in batch with weighted sampler]
  HOLD (0): 31.2%  ✓ Balanced
  BUY (1): 37.5%   ✓ Balanced
  SELL (2): 31.2%  ✓ Balanced

[Sanity Check]
  Step 0: loss=1.1061, acc=0.3906
  Step 40: loss=0.6116, acc=0.5469
  WARNING: Model cannot overfit single batch (acc=0.70)  ✗ FAIL

[Initial Predictions]
  HOLD (0): pred=0%, actual=53%    ✗ Never predicted
  BUY (1):  pred=100%, actual=9%   ✗ Always predicted
  SELL (2): pred=0%, actual=38%    ✗ Never predicted
```

**Problem:** Model collapses to always predicting BUY (100%).

---

## Root Cause Analysis

### Hypothesis: `in_position` Flag is Too Strong

The `in_position` feature (feature 14) has near-perfect correlation with labels:

```python
# From debug_data.py output:
Label distribution when FLAT (in_position=0):
  HOLD: 90.0%
  BUY:  10.0%
  SELL: 0.0%  ← Never SELL when flat

Label distribution when IN-POSITION (in_position=1):
  HOLD: 7.2%
  BUY:  0.0%   ← Never BUY when in position
  SELL: 92.8%
```

**The issue:**
1. Model learns to focus on `in_position` flag (easiest signal)
2. But the correlation isn't perfect (10% BUY vs 90% HOLD when flat)
3. Model gets confused and collapses to majority class

**Why sanity check fails:**
- Weighted sampler creates balanced batches (33% each class)
- But `in_position` flag still says "50% should be SELL, 50% should be BUY/HOLD"
- Model can't reconcile this contradiction with only 50 optimization steps
- Needs more complex learning to use other features

---

## Possible Solutions

### Option A: Remove `in_position` Flag (Simplest)

**Rationale:** Force model to learn from price/volume patterns, not context flag.

```python
# In preparation.py extract_features()
# Remove feature 14
features = np.column_stack([
    log_close, ret_1, ret_3, ret_5, range_ratio, volume_log,
    rsi / 100.0, macd_norm, bb_upper_dev, bb_lower_dev,
    vwap_dev, momentum_10,
    indicator_ready_short, indicator_ready_long,
    # position_flag,  ← REMOVE THIS
])
```

**Pros:**
- Model must learn actual trading patterns
- More generalizable
- Simpler feature space

**Cons:**
- Harder to learn (no easy signal)
- May need deeper network

### Option B: Use Binary Classification Per Context

Instead of 3-class with dual samples, split into two separate models:

**Model 1 (Entry):** When flat, predict BUY vs HOLD
```python
# Only train on in_position=0 samples
# Classes: HOLD (0) vs BUY (1)
```

**Model 2 (Exit):** When in-position, predict SELL vs HOLD
```python
# Only train on in_position=1 samples
# Classes: HOLD (0) vs SELL (2)
```

**Pros:**
- Each model has simpler task
- No contradiction in signals
- Can tune each independently

**Cons:**
- Need to train/deploy two models
- More complex pipeline

### Option C: Increase Sanity Check Steps

Maybe model just needs more steps to learn:

```python
# In diagnose.py and train.py
sanity_check_overfit_batch(model, train_loader, device, num_steps=200)  # Was 50
```

**Pros:**
- Quick to test
- May resolve issue if just slow convergence

**Cons:**
- Doesn't fix fundamental contradiction
- Longer diagnostic time

### Option D: Simplify to Binary (Recommended for Quick Win)

**Go back to binary classification:** BUY vs NO_BUY

```python
# In preparation.py
if profit_pct >= TAKE_PROFIT_PCT and drawdown_pct >= STOP_LOSS_PCT:
    label = 1  # BUY
else:
    label = 0  # NO_BUY (combines HOLD and SELL)
```

**Pros:**
- Simplest problem for model
- No position context needed
- Should work immediately

**Cons:**
- Doesn't learn when to exit
- Need separate exit logic

---

## Recommended Next Steps

### Immediate (Today):

**1. Try Option A (Remove `in_position` flag)**

Quickest test - modify one line:
```bash
cd ai_model/src/data
# Edit preparation.py line 302: comment out position_flag
# Edit preparation.py line 319: remove position_flag from column_stack
python preprocess.py --csv-path ../../data/raw/rawdata.csv --output-dir ../../data/processed
cd ../
python diagnose.py
```

If sanity check passes (>80% acc), proceed to full training.

**2. If still fails, try Option D (Binary classification)**

Modify labeling logic, reprocess, test.

### Short-term (This Week):

If diagnostics pass:
1. Train baseline LSTM for 10 epochs
2. Check if validation loss decreases
3. If loss decreases, train to completion
4. Backtest on historical data

### Medium-term (Next Week):

1. Compare baseline LSTM vs Attention LSTM
2. Analyze attention weights to see what model focuses on
3. Tune hyperparameters if needed
4. Paper trade with best model

---

## Files Modified/Created

| File | Status | Description |
|------|--------|-------------|
| `config/__init__.py` | Modified | Lowered thresholds, added sampler config |
| `preparation.py` | Modified | Fixed log_close normalization |
| `train.py` | Modified | Added diagnostics + weighted sampler |
| `attention_lstm.py` | Created | Attention-enhanced LSTM |
| `diagnose.py` | Created | Diagnostic script |
| `test_simple.py` | Created | Architecture verification |
| `debug_data.py` | Created | Data quality analysis |
| `models/__init__.py` | Modified | Export attention model |
| `IMPROVEMENT_PLAN.md` | Created | Detailed improvement roadmap |
| `IMPLEMENTATION_SUMMARY.md` | Created | Implementation details |
| `SESSION_SUMMARY.md` | Created | This file |

---

## Key Takeaways

1. **Supervised learning is viable** - just need right data setup
2. **Class imbalance addressed** - BUY went from 6.4% → 9.9%
3. **Model architecture works** - verified with simple tests
4. **Issue is data, not model** - specifically the `in_position` flag
5. **Two paths forward:**
   - **Quick win:** Remove `in_position` flag or go binary
   - **Better long-term:** Two separate models (entry/exit)

6. **Not ready for RL yet** - need supervised baseline working first

---

## Decision Point

**What would you like to do next?**

**Option A:** Remove `in_position` flag and retest (15 min)
**Option D:** Simplify to binary BUY/NO_BUY (30 min)
**Option B:** Split into two models (2-3 hours)
**Other:** Discuss alternative approaches

Let me know and I'll implement immediately!

---

*Created: 2025-12-22*
*Author: Trading Team*
*Status: Awaiting decision on next approach*
