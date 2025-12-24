# AI Trader

Automated trading bot for Solana memecoins using trained AI models.

## Quick Start

1. **Copy your trained model**:
   ```bash
   cp /path/to/your/model.zip models/
   ```

2. **Configure settings** (optional):
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

3. **Run the trader**:
   ```bash
   ./build/ai_trader --demo    # Demo mode with test token
   # OR
   ./build/ai_trader -t TOKEN_ADDRESS:SYMBOL  # Add specific tokens
   ```

## Directory Structure

```
ai_trader/
├── build/ai_trader      # Executable
├── models/
│   └── model.zip        # Your trained stable-baselines3 PPO model
├── logs/
│   └── YYYY-MM-DD/      # Daily log files
│       ├── trader.log   # General logs
│       ├── trades.csv   # Trade records
│       └── errors.log   # Errors only
├── scripts/
│   └── inference.py     # Python inference server
├── .env                  # Configuration
└── .env.example          # Config template
```

## Configuration (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| `PAPERTRADE` | `true` | Paper trade mode (set `false` for live) |
| `TRADE_AMOUNT_SOL` | `0.01` | SOL per trade |
| `PAPER_WALLET_BALANCE` | `1.0` | Starting paper balance |
| `TAKE_PROFIT_PCT` | `0.30` | Take profit at 30% |
| `STOP_LOSS_PCT` | `0.08` | Stop loss at 8% |

## Model Requirements

The model should be a trained **stable-baselines3 PPO** model saved with:

```python
model.save("model.zip")
```

**Expected Input**: 19 features (14 price + 5 position state)
**Expected Output**: 3 actions (HOLD, BUY, SELL)

## Commands

```bash
# Build
make clean && make

# Run with demo token
./build/ai_trader --demo

# Run with specific tokens
./build/ai_trader -t So11111...112:SOL -t PEPE123:PEPE

# Custom config and model path
./build/ai_trader -c /path/to/.env -m /path/to/models
```

## Features

- **Paper Trading**: Simulated trading with 7% round-trip fee
- **Take Profit / Stop Loss**: Automatic position management
- **Structured Logging**: CSV trade records for analysis
- **Model Inference**: Uses your trained RL model
- **Real-time Charts**: 1-second polling from GMGN API
