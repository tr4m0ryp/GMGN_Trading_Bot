# gmgn Network Logger

Puppeteer-based tool for capturing network traffic from gmgn.ai platform to analyze API endpoints and data structures.

## Purpose

This tool captures all network requests/responses from gmgn.ai while you set up filters and monitor coin listings. The captured data will be used to:
1. Identify API endpoints
2. Understand request/response formats
3. Build a C-based listener that directly interfaces with the API

## Installation

```bash
cd logger
npm install
```

## Usage

```bash
# Run with default 5-minute capture
npm start

# Run with custom duration (in minutes)
node capture.js 10
```

## How It Works

1. Opens a Chrome browser window (non-headless)
2. Navigates to gmgn.ai
3. **You manually set up your filters** in the browser
4. Captures all network traffic in the background
5. After the specified duration, saves logs to `./logs/` directory

## Output Files

- `network_capture_TIMESTAMP.json` - Full network log with requests/responses
- `endpoints_summary_TIMESTAMP.txt` - List of unique API endpoints discovered

## Next Steps

After capturing network logs:
1. Analyze the JSON files to identify relevant API endpoints
2. Understand the request parameters and response formats
3. Implement a C-based HTTP client to replicate these API calls
4. Build the real-time listener in C for production use
