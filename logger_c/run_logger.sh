#!/bin/bash
#
# GMGN Token Logger Startup Script
# Sets required Cloudflare session cookies for API access
#
# Usage: ./run_logger.sh
#

# Extract cf_clearance from your browser cookies
# Firefox: Dev Tools > Storage > Cookies > gmgn.ai > cf_clearance
# Chrome: Dev Tools > Application > Cookies > gmgn.ai > cf_clearance
export GMGN_CF_CLEARANCE="CMR_Gfxo4GohZAESLxRyxLqUVvkxjRgovw5kod.op9I-1766234079-1.2.1.1-anez50vK3VE9yX3GVPJ6SH49MhRq15arbxqx9nUofCaauv14nbp.T9chtSKiSPhAX6HqIdw69ysF_f9NQpO_vsWhLOA9..iVQkDYz6mBebYfPBGojPvAgeOF_PsAvCbpAKMiCOiamFybcGXsdbconGSt_NFV2wS88l8ailEEM.FLgWizhOokjXzrzSaMwif.kfMZ_ZX.mc06Me6gr_XptPtLOL.HTqERMIvTLNS2lLQ"

# Optional: Set other cookies for better compatibility
export GMGN_GA="GA1.1.1216464152.1766234082"
export GMGN_GA_SESSION="GS1.1.1766242908.4.1.1766245615.56.0.0"
export GMGN_CF_BM="Uq4zhFQAQXqH9RWmM3PSwi7gf2C2jREKOrQVQruUxrM-1766245715-1.0.1.1-_fIf9wp3EdR1R6Z1s3fYLSjwQrFM3AotLSrXxr8txxiBzUP9Xe15hoU3DfsM37ooHOkVhrsAXaYh8XCNIOzBAknQRHbfN26VRxpc4EHRL3I"

echo "Starting GMGN Token Logger with session cookies..."
echo "cf_clearance: ${GMGN_CF_CLEARANCE:0:50}..."
echo ""

# Run the logger, suppressing libwebsockets warnings
./build/gmgn_logger "$@" 2>/dev/null
