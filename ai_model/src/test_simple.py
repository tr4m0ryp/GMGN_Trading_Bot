"""
Test if model can learn a trivial pattern (no packed sequences).

This tests if the LSTM itself is broken or if it's an issue with our data/training setup.
"""

import torch
import torch.nn as nn
import numpy as np

# Simple test: can LSTM learn XOR-like pattern?
def test_lstm_basic():
    """Test if LSTM can learn a simple pattern."""
    print("Testing basic LSTM learning capability...")

    # Create simple model
    model = nn.LSTM(input_size=2, hidden_size=16, num_layers=1, batch_first=True)
    fc = nn.Linear(16, 2)

    # Simple dataset: classify last element
    # If seq = [0, 0, 0, 1], label = 1
    # If seq = [1, 1, 1, 0], label = 0
    X_train = []
    y_train = []

    for _ in range(100):
        seq_len = np.random.randint(5, 20)
        seq = np.random.randint(0, 2, (seq_len, 2)).astype(np.float32)
        label = seq[-1, 0]  # Label is last element's first feature
        X_train.append(torch.FloatTensor(seq))
        y_train.append(int(label))

    # Pad sequences
    X_padded = nn.utils.rnn.pad_sequence(X_train, batch_first=True)
    y_tensor = torch.LongTensor(y_train)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(fc.parameters()), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    print("\nTraining on simple pattern...")
    for step in range(200):
        optimizer.zero_grad()

        lstm_out, _ = model(X_padded)
        # Get last output for each sequence
        last_outputs = lstm_out[range(len(X_train)), [len(x)-1 for x in X_train], :]
        logits = fc(last_outputs)

        loss = criterion(logits, y_tensor)
        loss.backward()
        optimizer.step()

        if step % 40 == 0:
            acc = (logits.argmax(dim=1) == y_tensor).float().mean()
            print(f"  Step {step}: loss={loss.item():.4f}, acc={acc.item():.4f}")

    final_acc = (logits.argmax(dim=1) == y_tensor).float().mean().item()

    if final_acc > 0.9:
        print(f"\n✓ PASS: LSTM can learn (acc={final_acc:.2f})")
        return True
    else:
        print(f"\n✗ FAIL: LSTM cannot learn simple pattern (acc={final_acc:.2f})")
        return False


# Test our exact model architecture
def test_our_model():
    """Test if our model architecture can learn a simple pattern."""
    print("\n" + "="*60)
    print("Testing our VariableLengthLSTMTrader architecture...")
    print("="*60)

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))

    from models.lstm import VariableLengthLSTMTrader

    model = VariableLengthLSTMTrader(input_size=15, hidden_size=64, num_layers=2, dropout=0.3)

    # Create simple dataset with our exact feature format
    X_train = []
    y_train = []
    lengths = []

    for _ in range(100):
        seq_len = np.random.randint(12, 50)
        # 15 features per timestep
        seq = np.random.randn(seq_len, 15).astype(np.float32)

        # Simple rule: if mean of feature 0 > 0, label = 1, else label = 0
        label = 1 if seq[:, 0].mean() > 0 else 0

        X_train.append(torch.FloatTensor(seq))
        y_train.append(label)
        lengths.append(seq_len)

    # Pad sequences
    X_padded = nn.utils.rnn.pad_sequence(X_train, batch_first=True, padding_value=0.0)
    y_tensor = torch.LongTensor(y_train)
    lengths_tensor = torch.LongTensor(lengths)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    print("\nTraining on simple mean > 0 pattern...")
    for step in range(200):
        optimizer.zero_grad()

        logits, _ = model(X_padded, lengths_tensor)
        loss = criterion(logits[:, :2], y_tensor)  # Only use first 2 classes
        loss.backward()
        optimizer.step()

        if step % 40 == 0:
            preds = logits[:, :2].argmax(dim=1)
            acc = (preds == y_tensor).float().mean()
            print(f"  Step {step}: loss={loss.item():.4f}, acc={acc.item():.4f}")

    preds = logits[:, :2].argmax(dim=1)
    final_acc = (preds == y_tensor).float().mean().item()

    if final_acc > 0.8:
        print(f"\n✓ PASS: Our model can learn (acc={final_acc:.2f})")
        return True
    else:
        print(f"\n✗ FAIL: Our model cannot learn simple pattern (acc={final_acc:.2f})")
        print("This suggests an issue with the model architecture.")
        return False


if __name__ == '__main__':
    print("="*60)
    print("LSTM CAPABILITY TESTS")
    print("="*60)

    test1 = test_lstm_basic()
    test2 = test_our_model()

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Basic LSTM:        {'PASS' if test1 else 'FAIL'}")
    print(f"Our architecture:  {'PASS' if test2 else 'FAIL'}")

    if test1 and not test2:
        print("\n⚠ Our model architecture has issues - investigate dropout/layers")
    elif not test1:
        print("\n⚠ PyTorch LSTM itself is failing - check installation")
    elif test1 and test2:
        print("\n✓ Models work - issue is likely in the data or training setup")
