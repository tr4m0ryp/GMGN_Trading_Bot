# Notebooks Directory

## CRITICAL: Notebooks are ONLY for GPU Access

**DO NOT define model code in notebooks!**

Notebooks should contain MINIMAL code and only be used to:
1. Import model classes from `src/`
2. Load data
3. Call training functions
4. Access GPU resources

## Purpose

Jupyter notebooks run on systems with GPU access (Google Colab, local GPU machine, etc.). They provide GPU acceleration for training, but ALL model logic must be in Python files in `src/`.

## File Structure

### train_gpu.ipynb
Main training notebook:
```python
# Cell 1: Imports
from src.model_lstm import VariableLengthLSTMTrader
from src.train import train_model
from src.data_preparation import load_training_data

# Cell 2: Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Cell 3: Load Data
train_data, val_data = load_training_data('../data/processed/')

# Cell 4: Initialize Model
model = VariableLengthLSTMTrader(
    input_size=11,
    hidden_size=128,
    num_layers=2
).to(device)

# Cell 5: Train
train_model(
    model=model,
    train_loader=train_data,
    val_loader=val_data,
    device=device,
    epochs=100
)
```

## What NOT to Do

**WRONG - Defining model in notebook:**
```python
# ❌ DO NOT DO THIS
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(11, 128)
        # ... more model code
```

**CORRECT - Import from src:**
```python
# ✅ DO THIS INSTEAD
from src.model_lstm import VariableLengthLSTMTrader

model = VariableLengthLSTMTrader()
```

## Why This Matters

1. **Version Control**: Model code in `.py` files can be properly tracked in git
2. **Testing**: Python files can be unit tested, notebooks cannot
3. **Reusability**: Model code can be imported in multiple places
4. **Debugging**: Python files are easier to debug than notebooks
5. **Collaboration**: Easier for multiple developers to work on Python files

## Running on Google Colab

1. Upload entire `ai_model/` directory to Google Drive
2. Mount Drive in Colab: `from google.colab import drive; drive.mount('/content/drive')`
3. Navigate to directory: `cd /content/drive/MyDrive/ai_model`
4. Open `notebooks/train_gpu.ipynb`
5. Run all cells

## Running Locally with GPU

```bash
cd ai_model/notebooks
jupyter notebook train_gpu.ipynb
```

Ensure you have:
- CUDA installed
- PyTorch with CUDA support
- GPU drivers updated
