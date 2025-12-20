/**
 * gmgn Network Logger
 *
 * This script uses Playwright to visit gmgn.ai, capture network traffic,
 * and save all API calls to the logs directory for analysis.
 *
 * Usage:
 *   node capture.js [duration_in_minutes]
 *
 * The script will:
 * 1. Launch a Chrome browser window
 * 2. Navigate to gmgn.ai
 * 3. Allow you to manually set up filters
 * 4. Capture all network requests/responses
 * 5. Save logs to ./logs/ directory
 */

const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const CONFIG = {
    url: 'https://gmgn.ai',
    duration: parseInt(process.argv[2]) || 5,
    headless: false,
    logsDir: path.join(__dirname, 'logs'),
    startDelay: 30
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
    console.log('[START] gmgn Network Logger (Chrome)');
    console.log(`[CONFIG] URL: ${CONFIG.url}`);
    console.log(`[CONFIG] Duration: ${CONFIG.duration} minutes`);
    console.log(`[CONFIG] Start delay: ${CONFIG.startDelay} seconds`);
    console.log(`[CONFIG] Logs directory: ${CONFIG.logsDir}`);

    if (!fs.existsSync(CONFIG.logsDir)) {
        fs.mkdirSync(CONFIG.logsDir, { recursive: true });
    }

    const browser = await chromium.launch({
        headless: CONFIG.headless,
        args: [
            '--disable-blink-features=AutomationControlled',
            '--disable-dev-shm-usage',
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-web-security',
            '--disable-features=IsolateOrigins,site-per-process',
            '--disable-infobars',
            '--window-size=1920,1080',
            '--start-maximized',
            '--disable-notifications',
            '--disable-popup-blocking',
            '--disable-save-password-bubble',
            '--disable-translate',
            '--disable-background-timer-throttling',
            '--disable-backgrounding-occluded-windows',
            '--disable-renderer-backgrounding'
        ]
    });

    const context = await browser.newContext({
        viewport: { width: 1920, height: 1080 },
        userAgent: 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        locale: 'en-US',
        timezoneId: 'America/New_York',
        permissions: ['geolocation', 'notifications'],
        geolocation: { latitude: 40.7128, longitude: -74.0060 },
        extraHTTPHeaders: {
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"Linux"',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Upgrade-Insecure-Requests': '1'
        }
    });

    await context.addInitScript(() => {
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined
        });
        Object.defineProperty(navigator, 'plugins', {
            get: () => [1, 2, 3, 4, 5]
        });
        Object.defineProperty(navigator, 'languages', {
            get: () => ['en-US', 'en']
        });
        window.chrome = {
            runtime: {}
        };
    });

    const page = await context.newPage();

    console.log(`\n[NAVIGATE] Opening ${CONFIG.url}...`);
    await page.goto(CONFIG.url, { waitUntil: 'load' }).catch(() => {});

    console.log('\n========================================');
    console.log('CHROME BROWSER IS NOW OPEN');
    console.log('');
    console.log('COMPLETE CLOUDFLARE VERIFICATION NOW');
    console.log('You have 30 seconds before logging starts');
    console.log('');
    console.log('After verification, navigate to trenches');
    console.log('and set up your filters');
    console.log('========================================\n');

    for (let i = CONFIG.startDelay; i > 0; i--) {
        process.stdout.write(`\r[WAIT] Starting network logging in ${i} seconds...`);
        await new Promise(resolve => setTimeout(resolve, 1000));
    }
    console.log('\r[START] Network logging is now ACTIVE\n');

    page.on('request', request => {
        let postData = null;
        try {
            postData = request.postDataJSON();
        } catch (e) {
            postData = request.postData();
        }

        const logEntry = {
            id: ++requestCounter,
            timestamp: new Date().toISOString(),
            method: request.method(),
            url: request.url(),
            resourceType: request.resourceType(),
            headers: request.headers(),
            postData: postData
        };

        console.log(`[REQ ${requestCounter}] ${request.method()} ${request.url()}`);
        networkLogs.push(logEntry);
    });

    page.on('response', async response => {
        const request = response.request();
        const logEntry = networkLogs.find(log =>
            log.url === request.url() &&
            log.method === request.method() &&
            log.timestamp
        );

        if (logEntry) {
            logEntry.status = response.status();
            logEntry.statusText = response.statusText();
            logEntry.responseHeaders = response.headers();

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

    console.log('========================================');
    console.log(`Capture will run for ${CONFIG.duration} minutes`);
    console.log('All network traffic is being logged...');
    console.log('========================================\n');

    const durationMs = CONFIG.duration * 60 * 1000;
    await new Promise(resolve => setTimeout(resolve, durationMs));

    console.log('\n[STOP] Capture duration completed');

    saveLogs();

    await browser.close();
    console.log('[DONE] Browser closed');
}

captureNetworkTraffic()
    .catch(error => {
        console.error('[ERROR]', error);
        process.exit(1);
    });
