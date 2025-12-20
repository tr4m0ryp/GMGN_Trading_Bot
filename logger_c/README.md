# GMGN Token Logger (C)

Real-time Solana token logger that connects to GMGN.ai WebSocket API.

## Features

- Real-time connection to GMGN.ai WebSocket API
- Configurable token filters (market cap, KOL count, age, etc.)
- Terminal-based output with color support
- Automatic reconnection with exponential backoff
- Low latency C implementation

## Default Filters

- Market Cap: $5.5K - $10K
- Minimum KOL: 1
- Maximum Age: 10 minutes

## Dependencies

```bash
# Fedora/RHEL
sudo dnf install libwebsockets-devel cjson-devel openssl-devel

# Ubuntu/Debian
sudo apt install libwebsockets-dev libcjson-dev libssl-dev
```

## Build

```bash
cd logger_c
make clean && make
```

## Run

```bash
./build/gmgn_logger [options]

Options:
  -c, --chain CHAIN      Target chain (default: sol)
  -v, --verbose          Increase verbosity
  -q, --quiet            Minimal output
  --min-mc VALUE         Minimum market cap in dollars (e.g., 5.5)
  --max-mc VALUE         Maximum market cap in dollars (e.g., 10)
  --min-kol VALUE        Minimum KOL count
  --max-age VALUE        Maximum age in minutes
  -h, --help             Show help
```

## Project Structure

```
logger_c/
├── src/
│   ├── gmgn_logger.c      # Main application
│   ├── websocket_client.c # WebSocket connection handler
│   ├── json_parser.c      # GMGN JSON message parser
│   ├── filter.c           # Token filtering logic
│   └── output.c           # Terminal output formatting
├── include/
│   ├── gmgn_types.h       # Data structures
│   ├── websocket_client.h # WebSocket client interface
│   ├── json_parser.h      # JSON parser interface
│   ├── filter.h           # Filter interface
│   └── output.h           # Output interface
├── build/                 # Compiled binaries
└── Makefile               # Build configuration
```
