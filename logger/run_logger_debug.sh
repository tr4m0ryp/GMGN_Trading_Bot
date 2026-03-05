#!/bin/bash
#
# GMGN Token Logger Debug Startup Script
# Runs logger with verbose debug output enabled
#
# Usage: ./run_logger_debug.sh
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEBUG_LOG="/tmp/gmgn_debug.log"

# Clear previous debug log
rm -f "${DEBUG_LOG}"

# Enable debug logging
export GMGN_DEBUG=1

echo "Debug logging enabled - output will be written to ${DEBUG_LOG}"
echo "Press Ctrl+C to stop the logger and view debug output"
echo ""

# Run the normal startup script which will inherit the debug flag
"${SCRIPT_DIR}/run_logger.sh" "$@"

# Show debug log after exit
if [[ -f "${DEBUG_LOG}" ]]; then
    echo ""
    echo "=================================================="
    echo "DEBUG LOG OUTPUT:"
    echo "=================================================="
    cat "${DEBUG_LOG}"
    echo "=================================================="
    echo ""
    echo "Full debug log saved at: ${DEBUG_LOG}"
else
    echo ""
    echo "WARNING: No debug log was created at ${DEBUG_LOG}"
    echo "This means no WebSocket messages were received."
fi
