/**
 * HAR to JSON Converter
 *
 * Converts browser-exported HAR files to the same format as capture.js
 * This allows manual network capture when Cloudflare blocks automation.
 *
 * Usage:
 *   node convert_har.js <har_file>
 */

const fs = require('fs');
const path = require('path');

if (process.argv.length < 3) {
    console.error('Usage: node convert_har.js <har_file>');
    console.error('Example: node convert_har.js logs/manual_capture.har');
    process.exit(1);
}

const harFile = process.argv[2];

if (!fs.existsSync(harFile)) {
    console.error(`Error: File not found: ${harFile}`);
    process.exit(1);
}

console.log(`[INFO] Reading HAR file: ${harFile}`);
const harData = JSON.parse(fs.readFileSync(harFile, 'utf8'));

const networkLogs = [];
let requestCounter = 0;

harData.log.entries.forEach(entry => {
    const logEntry = {
        id: ++requestCounter,
        timestamp: entry.startedDateTime,
        method: entry.request.method,
        url: entry.request.url,
        resourceType: entry._resourceType || 'unknown',
        headers: {},
        postData: null,
        status: entry.response.status,
        statusText: entry.response.statusText,
        responseHeaders: {}
    };

    entry.request.headers.forEach(h => {
        logEntry.headers[h.name.toLowerCase()] = h.value;
    });

    entry.response.headers.forEach(h => {
        logEntry.responseHeaders[h.name.toLowerCase()] = h.value;
    });

    if (entry.request.postData) {
        try {
            logEntry.postData = JSON.parse(entry.request.postData.text);
        } catch {
            logEntry.postData = entry.request.postData.text;
        }
    }

    if (entry.response.content && entry.response.content.text) {
        logEntry.responseBody = entry.response.content.text;
    }

    networkLogs.push(logEntry);
});

const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
const outputFile = `network_capture_${timestamp}.json`;
const outputPath = path.join(path.dirname(harFile), outputFile);

const logData = {
    captureTime: new Date().toISOString(),
    source: 'manual_har_export',
    sourceFile: harFile,
    totalRequests: networkLogs.length,
    requests: networkLogs
};

fs.writeFileSync(outputPath, JSON.stringify(logData, null, 2));

console.log(`\n[SAVE] Converted to: ${outputFile}`);
console.log(`[INFO] Total requests: ${networkLogs.length}`);

const endpoints = [...new Set(networkLogs.map(log => {
    try {
        const url = new URL(log.url);
        return url.pathname;
    } catch {
        return log.url;
    }
}))];

const summaryFile = `endpoints_summary_${timestamp}.txt`;
const summaryPath = path.join(path.dirname(harFile), summaryFile);
fs.writeFileSync(summaryPath, endpoints.join('\n'));
console.log(`[SAVE] Endpoints summary: ${summaryFile}`);

const apiCalls = networkLogs.filter(log =>
    log.url.includes('/api/') ||
    log.url.includes('gmgn.ai') && (log.resourceType === 'xhr' || log.resourceType === 'fetch')
);

console.log(`[INFO] API calls found: ${apiCalls.length}`);
console.log('\n[DONE] Conversion complete');
