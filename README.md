# GMGN Trading Bot

Automated cryptocurrency trading bot that learns optimal buy/sell strategies using reinforcement learning. The bot connects to GMGN.ai to monitor new token launches and makes trading decisions based on a PPO-trained agent.

## Project Overview

This project consists of three main components:

1. **logger_c/** - Real-time C-based token logger that connects to GMGN.ai WebSocket API
2. **ai_model/** - Reinforcement learning trading agent with curriculum learning
3. **trading_algorithm/** (Future) - Production trading system integration

## Quick Start

### Training the RL Agent

The AI model uses reinforcement learning (PPO) with curriculum learning to discover profitable trading patterns.

```bash
cd ai_model
python src/data/preprocess.py  # Prepare training data
# Then use train_rl_agent.ipynb in Google Colab for GPU training
```

See [ai_model/README.md](ai_model/README.md) for detailed instructions.

### Running the Token Logger

The C-based logger connects to GMGN.ai and streams new token data:

```bash
cd logger_c
make
./build/gmgn_logger
```

See [logger_c/README.md](logger_c/README.md) for configuration options.

## AI Model Approach

### Why Reinforcement Learning?

Initial experiments with classification (predicting BUY/SELL/HOLD labels) failed due to:
- Unrealistic label generation (64% of data labeled as SELL)
- Features couldn't predict 20-second future price movements
- Model accuracy was worse than random (14-34% vs 33% baseline)

**Solution**: Reinforcement learning lets the agent **discover** profitable patterns through trial and error, rather than fitting to hand-crafted labels.

### Key Features

- **Curriculum Learning**: Gradually increases trading fees from 0% to 100% during training
- **Hindsight Rewards**: Penalizes agent for missing profitable opportunities
- **Asymmetric Rewards**: Bigger bonus for winning trades than penalty for losses
- **PPO Algorithm**: Proximal Policy Optimization with high entropy (0.05) for exploration

### Training Results

The agent learns to:
- Make active trades (prevents "do nothing" policy collapse)
- Identify profitable entry/exit points
- Manage risk with proper position sizing

Training takes ~1M steps (~2-3 hours on T4 GPU).

## Project Structure

```
gmgn_trading/
├── logger_c/              # C-based WebSocket token logger
│   ├── src/               # Source files
│   ├── include/           # Headers
│   └── Makefile           # Build configuration
├── ai_model/              # RL trading agent
│   ├── data/              # Training data
│   ├── notebooks/         # Jupyter training notebooks
│   ├── src/
│   │   ├── config/        # Configuration
│   │   ├── data/          # Data loading and preprocessing
│   │   ├── rl/            # RL environment and training
│   │   └── utils/         # Helper functions
│   └── models/            # Saved models
├── trading_algorithm/     # (Future) Production trading
└── logger/                # Legacy JS logger
```

## Dependencies

### AI Model
- Python 3.10+
- PyTorch 2.0+
- Stable-Baselines3
- Gymnasium
- NumPy, Pandas

### Logger
- C compiler (gcc/clang)
- libwebsockets 4.x
- JSON-C

## Development Guidelines

This project follows strict C coding standards documented in [CLAUDE.md](CLAUDE.md):
- Snake_case naming for functions/variables
- Comprehensive documentation for all functions
- Memory safety and leak prevention
- Professional code quality

## Known Issues & Lessons Learned

### GMGN WebSocket API

Critical discoveries documented in [CLAUDE.md](CLAUDE.md):
- Correct subscription format requires `action`, `f`, and `id` fields
- Connection URL: `wss://ws.gmgn.ai/quotation`
- Use `new_pool_info` channel for new token launches

### Training Issues Solved

1. **Policy Collapse**: Agent learned to do nothing → Fixed with curriculum learning
2. **Poor Labels**: Classification failed → Switched to RL
3. **Low Exploration**: Added higher entropy coefficient (0.05)

## Future Work

- [ ] Production trading algorithm integration
- [ ] Real-time inference pipeline
- [ ] Risk management system
- [ ] Multi-DEX support beyond Pump.fun
- [ ] Advanced reward shaping with Sharpe ratio optimization

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please follow the coding standards in CLAUDE.md.

## Disclaimer

This is experimental software for educational purposes. Cryptocurrency trading carries significant risk. Use at your own risk.
