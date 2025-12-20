# Manual Network Capture (Cloudflare Bypass)

If the automated script cannot bypass Cloudflare verification, use this manual method instead.

## Method: Browser DevTools HAR Export

This method uses your regular browser (not automated), so Cloudflare won't block you.

### Steps:

1. **Open Firefox or Chrome normally** (not through the script)

2. **Open Developer Tools**:
   - Press `F12` or `Ctrl+Shift+I`
   - Go to the **Network** tab

3. **Enable persistent logs**:
   - Check the box for "Persist logs" or "Preserve log"
   - This ensures network traffic isn't cleared on page navigation

4. **Clear existing logs**:
   - Click the trash/clear icon to start fresh

5. **Visit gmgn.ai**:
   - Navigate to https://gmgn.ai
   - Complete Cloudflare verification
   - Navigate to the trenches section
   - Set up all your desired filters

6. **Let it run**:
   - Keep the page open for 5-10 minutes
   - Let the API calls populate with filtered coin data

7. **Export the network log**:
   - Right-click anywhere in the Network tab
   - Select "Save All As HAR" or "Export HAR"
   - Save to: `logger/logs/manual_capture.har`

8. **Convert HAR to JSON** (I'll help you with this):
   - Run: `node convert_har.js manual_capture.har`
   - This will extract API endpoints and filter parameters

## Why This Works

- Uses your regular browser session (no automation detection)
- Cloudflare sees normal user behavior
- HAR format captures everything the automated script would
- You maintain full control over the process

## Next Steps

After you export the HAR file, I'll analyze it to extract:
- API endpoints for coin listings
- Filter parameters and their values
- Request/response formats
- Authentication headers if needed
