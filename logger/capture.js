/**
 * gmgn Network Logger
 *
 * This script uses Puppeteer to visit gmgn.ai, capture network traffic,
 * and save all API calls to the logs directory for analysis.
 *
 * Usage:
 *   node capture.js [duration_in_minutes]
 *
 * The script will:
 * 1. Launch a browser window
 * 2. Navigate to gmgn.ai
 * 3. Allow you to manually set up filters
 * 4. Capture all network requests/responses
 * 5. Save logs to ./logs/ directory
 */

const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

const CONFIG = {
    url: 'https://gmgn.ai',
    duration: parseInt(process.argv[2]) || 5, // Default 5 minutes
    headless: false, // Keep browser visible for manual filter setup
    logsDir: path.join(__dirname, 'logs')
};

const networkLogs = [];
let requestCounter = 0;

/**
 * Format timestamp for log files
 */
function getTimestamp() {
    const now = new Date();
    return now.toISOString().replace(/[:.]/g, '-');
}

/**
 * Save captured network logs to file
 */
function saveLogs() {
    const timestamp = getTimestamp();
    const filename = `network_capture_${timestamp}.json`;
    const filepath = path.join(CONFIG.logsDir, filename);

    const logData = {
        captureTime: new Date().toISOString(),
        duration: CONFIG.duration,
        url: CONFIG.url,
        totalRequests: networkLogs.length,
        requests: networkLogs
    };

    fs.writeFileSync(filepath, JSON.stringify(logData, null, 2));
    console.log(`\n[SAVE] Logs saved to: ${filename}`);
    console.log(`[INFO] Total requests captured: ${networkLogs.length}`);

    // Also save a summary of unique endpoints
    const endpoints = [...new Set(networkLogs.map(log => {
        try {
            const url = new URL(log.url);
            return url.pathname;
        } catch {
            return log.url;
        }
    }))];

    const summaryFile = `endpoints_summary_${timestamp}.txt`;
    const summaryPath = path.join(CONFIG.logsDir, summaryFile);
    fs.writeFileSync(summaryPath, endpoints.join('\n'));
    console.log(`[SAVE] Endpoints summary saved to: ${summaryFile}`);
}

/**
 * Main capture function
 */
async function captureNetworkTraffic() {
    console.log('[START] gmgn Network Logger');
    console.log(`[CONFIG] URL: ${CONFIG.url}`);
    console.log(`[CONFIG] Duration: ${CONFIG.duration} minutes`);
    console.log(`[CONFIG] Logs directory: ${CONFIG.logsDir}`);

    // Ensure logs directory exists
    if (!fs.existsSync(CONFIG.logsDir)) {
        fs.mkdirSync(CONFIG.logsDir, { recursive: true });
    }

    const browser = await puppeteer.launch({
        headless: CONFIG.headless,
        args: ['--no-sandbox', '--disable-setuid-sandbox'],
        defaultViewport: { width: 1920, height: 1080 }
    });

    const page = await browser.newPage();

    // Enable request interception
    await page.setRequestInterception(true);

    // Capture all requests
    page.on('request', request => {
        const logEntry = {
            id: ++requestCounter,
            timestamp: new Date().toISOString(),
            method: request.method(),
            url: request.url(),
            resourceType: request.resourceType(),
            headers: request.headers(),
            postData: request.postData()
        };

        console.log(`[REQ ${requestCounter}] ${request.method()} ${request.url()}`);
        networkLogs.push(logEntry);

        request.continue();
    });

    // Capture responses
    page.on('response', async response => {
        const request = response.request();
        const logEntry = networkLogs.find(log =>
            log.url === request.url() &&
            log.method === request.method()
        );

        if (logEntry) {
            logEntry.status = response.status();
            logEntry.statusText = response.statusText();
            logEntry.responseHeaders = response.headers();

            // Capture response body for API calls
            if (request.resourceType() === 'xhr' ||
                request.resourceType() === 'fetch' ||
                request.url().includes('/api/')) {
                try {
                    logEntry.responseBody = await response.text();
                } catch (error) {
                    logEntry.responseBodyError = error.message;
                }
            }

            console.log(`[RES ${logEntry.id}] ${response.status()} ${request.url()}`);
        }
    });

    console.log(`\n[NAVIGATE] Opening ${CONFIG.url}...`);
    await page.goto(CONFIG.url, { waitUntil: 'networkidle2' });

    console.log('\n========================================');
    console.log('BROWSER WINDOW IS NOW OPEN');
    console.log('Please set up your filters manually');
    console.log(`Capture will run for ${CONFIG.duration} minutes`);
    console.log('All network traffic is being logged...');
    console.log('========================================\n');

    // Wait for specified duration
    const durationMs = CONFIG.duration * 60 * 1000;
    await new Promise(resolve => setTimeout(resolve, durationMs));

    console.log('\n[STOP] Capture duration completed');

    // Save logs before closing
    saveLogs();

    await browser.close();
    console.log('[DONE] Browser closed');
}

// Run the capture
captureNetworkTraffic()
    .catch(error => {
        console.error('[ERROR]', error);
        process.exit(1);
    });
