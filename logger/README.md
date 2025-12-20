# gmgn Network Logger

Playwright-based tool for capturing network traffic from gmgn.ai platform to analyze API endpoints and data structures. Uses Firefox browser for better stability.

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

1. Opens a Firefox browser window (non-headless)
2. Navigates to gmgn.ai
3. **WAITS 30 SECONDS** - Complete Cloudflare verification during this time
4. Network logging starts automatically after 30 seconds
5. Navigate to trenches and **manually set up your filters**
6. Captures all network traffic in the background
7. After the specified duration, saves logs to `./logs/` directory

## Output Files

- `network_capture_TIMESTAMP.json` - Full network log with requests/responses
- `endpoints_summary_TIMESTAMP.txt` - List of unique API endpoints discovered

## Cloudflare Issues?

If you cannot bypass Cloudflare verification with the automated script, see **MANUAL_CAPTURE.md** for an alternative method using your browser's built-in DevTools. This method:
- Works with any browser (no automation detection)
- Bypasses all Cloudflare protections
- Produces the same output format
- Gives you full control

## Next Steps

After capturing network logs:
1. Analyze the JSON files to identify relevant API endpoints
2. Understand the request parameters and response formats
3. Implement a C-based HTTP client to replicate these API calls
4. Build the real-time listener in C for production use
