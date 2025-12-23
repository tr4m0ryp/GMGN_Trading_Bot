# AI Model Improvement Plan

## Current Status (Updated)

### Already Implemented

| Component | Change | Status |
|-----------|--------|--------|
| **Label Logic** | Position-aware: BUY/HOLD when flat, SELL/HOLD when in-position | Done |
| **MIN_HISTORY_LENGTH** | Reduced to 12 (from 30) for fast-acting | Done |
| **Features** | 15 features including `in_position` flag, indicator readiness | Done |
| **Focal Loss** | Implemented with class weights | Done |
| **Thresholds** | 8% TP, -4% SL, 2% trail backoff | Done |
| **Dual Samples** | Two samples per timestamp (flat + in-position context) | Done |

### Training Results (Original - Before Changes)

```
Loss: ~1.07 (near random)
Accuracy: ~62% oscillating
Class weights: [0.52, 0.97, 1.51] - imbalanced
```

---

## Remaining Issues & Solutions

### Issue 1: LSTM Architecture May Not Capture Patterns

**Problem**: Using only the last hidden state of LSTM discards temporal attention.

**Current architecture**:
```
LSTM(15, 128, 2 layers) -> last_hidden -> FC(128->64) -> FC(64->3)
```

**Proposed**: Add attention over LSTM outputs.

```python
class AttentionLSTMTrader(nn.Module):
    def __init__(self, input_size=15, hidden_size=128, num_layers=2, num_classes=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

        # Attention weights
        attn_scores = self.attention(lstm_out).squeeze(-1)  # (batch, seq_len)

        # Mask padding positions
        mask = torch.arange(lstm_out.size(1), device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)

        logits = self.fc(context)
        probs = torch.softmax(logits, dim=1)
        confidence = torch.max(probs, dim=1)[0]

        return logits, confidence
```

### Issue 2: Need Sanity Check Before Full Training

**Problem**: Without verifying the model can learn at all, we waste time on full training runs.

**Solution**: Add sanity check function.

```python
def sanity_check_overfit_batch(model, train_loader, device, num_steps=100):
    """Verify model can overfit a single batch (proves learning is possible)."""
    model.train()
    batch = next(iter(train_loader))
    features = batch['features'].to(device)
    labels = batch['labels'].to(device)
    seq_lengths = batch['seq_lengths']

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    print("Sanity check: overfitting single batch...")
    for step in range(num_steps):
        optimizer.zero_grad()
        logits, _ = model(features, seq_lengths)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            acc = (logits.argmax(dim=1) == labels).float().mean()
            print(f"  Step {step}: loss={loss.item():.4f}, acc={acc.item():.4f}")

    final_acc = (logits.argmax(dim=1) == labels).float().mean()
    if final_acc < 0.8:
        print("WARNING: Model cannot overfit single batch - check architecture/data")
        return False
    print("Sanity check PASSED: model can learn")
    return True
```

### Issue 3: Need Prediction Distribution Monitoring

**Problem**: Model might collapse to always predicting one class.

**Solution**: Log class distribution during validation.

```python
def analyze_predictions(model, val_loader, device):
    """Check if model predicts diverse classes."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            seq_lengths = batch['seq_lengths']

            logits, _ = model(features, seq_lengths)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    print("\nPrediction Distribution:")
    for cls in [0, 1, 2]:
        pred_count = sum(1 for p in all_preds if p == cls)
        label_count = sum(1 for l in all_labels if l == cls)
        print(f"  Class {cls}: pred={pred_count} ({pred_count/len(all_preds)*100:.1f}%), "
              f"actual={label_count} ({label_count/len(all_labels)*100:.1f}%)")

    return all_preds, all_labels
```

### Issue 4: Class Distribution After New Labeling

**Action Required**: Re-run preprocessing and check new label distribution.

The new dual-sample approach (flat + in-position) should produce:
- More balanced BUY vs HOLD (flat context)
- More balanced SELL vs HOLD (in-position context)

But we need to verify this empirically.

### Issue 5: Learning Rate May Still Be Too High

**Current**: 0.0005 with ReduceLROnPlateau

**Alternative**: Warmup + Cosine Annealing

```python
def get_cosine_scheduler(optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + 0.5 * (1 - min_lr) * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

### Issue 6: Feature Scaling Stability

**Current**: Per-sample normalization with log returns.

**Potential Issue**: `log_close` varies across samples (absolute price level).

**Solution**: Make `log_close` relative (log return from first candle).

```python
# Current (may cause issues)
log_close = np.log(np.clip(closes, 1e-8, None))

# Better (relative to first price)
log_close = np.log(np.clip(closes, 1e-8, None)) - np.log(np.clip(closes[0], 1e-8, None))
```

---

## Recommended Action Order

| Priority | Action | Why |
|----------|--------|-----|
| **1** | Re-preprocess data with new labeling | Get fresh class distribution |
| **2** | Run sanity check (overfit 1 batch) | Verify model can learn |
| **3** | Check prediction distribution | Detect class collapse |
| **4** | Fix `log_close` to be relative | Normalize feature scale |
| **5** | Add attention mechanism | Better temporal modeling |
| **6** | Try warmup LR schedule | Stabilize early training |

---

## Quick Diagnostic Script

```python
# Run this before full training
def diagnose_training_setup(model, train_loader, val_loader, device):
    print("=" * 50)
    print("TRAINING DIAGNOSTICS")
    print("=" * 50)

    # 1. Check data
    batch = next(iter(train_loader))
    print(f"\n[Data] Batch size: {batch['features'].shape[0]}")
    print(f"[Data] Sequence lengths: min={batch['seq_lengths'].min()}, max={batch['seq_lengths'].max()}")
    print(f"[Data] Feature shape: {batch['features'].shape}")

    # 2. Check label distribution
    labels = batch['labels'].numpy()
    for cls in [0, 1, 2]:
        pct = (labels == cls).sum() / len(labels) * 100
        print(f"[Labels] Class {cls}: {pct:.1f}%")

    # 3. Check feature stats
    feats = batch['features'].numpy()
    print(f"\n[Features] Mean: {feats.mean():.4f}, Std: {feats.std():.4f}")
    print(f"[Features] Min: {feats.min():.4f}, Max: {feats.max():.4f}")
    print(f"[Features] NaN count: {np.isnan(feats).sum()}")
    print(f"[Features] Inf count: {np.isinf(feats).sum()}")

    # 4. Sanity check
    print("\n[Sanity] Testing if model can overfit single batch...")
    passed = sanity_check_overfit_batch(model, train_loader, device, num_steps=50)

    # 5. Initial predictions
    print("\n[Predictions] Initial model output distribution...")
    analyze_predictions(model, val_loader, device)

    return passed
```

---

## Expected Results After Fixes

| Metric | Before | Target |
|--------|--------|--------|
| Training loss | ~1.07 | < 0.7 (epoch 1), < 0.3 (epoch 10) |
| Validation accuracy | ~62% | > 70% |
| Prediction distribution | Unknown/collapsed | ~30-40% per class |
| Sanity check | N/A | PASS (>80% on 1 batch) |

---

## Code Changes Summary

Files to modify:
1. `preparation.py:285` - Make `log_close` relative
2. `lstm.py` - Add attention variant (optional)
3. `train.py` - Add sanity check and prediction monitoring
4. `preprocess.py` - Re-run after data changes

---

*Updated: 2025-12-22*
*Status: Partially implemented, diagnostics pending*
