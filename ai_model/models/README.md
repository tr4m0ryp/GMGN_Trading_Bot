# Models Directory

## Structure

- `checkpoints/` - Training checkpoints saved during training
- `best_model.pth` - Best performing model on validation set

## Model Files

### Checkpoint Format
Checkpoints are saved with format: `checkpoint_epoch_{N}.pth`

Each checkpoint contains:
```python
{
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': OrderedDict,
    'loss': float,
    'val_accuracy': float,
}
```

### Best Model
The `best_model.pth` file contains the model with best validation performance.

## Loading Models

```python
import torch
from src.model_lstm import VariableLengthLSTMTrader

model = VariableLengthLSTMTrader()
checkpoint = torch.load('models/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```
