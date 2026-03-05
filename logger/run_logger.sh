#!/bin/bash
#
# GMGN Token Logger Startup Script
# Sets required Cloudflare session cookies for API access
#
# Usage: ./run_logger.sh
#

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/logger_$(date +%Y%m%d).log"
ERROR_LOG="${LOG_DIR}/errors_$(date +%Y%m%d).log"

# Create logs directory if it doesn't exist
mkdir -p "${LOG_DIR}"

# =============================================================================
# LOGGING FUNCTIONS
# =============================================================================

log_info() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] [INFO] $*" | tee -a "${LOG_FILE}"
}

log_warn() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] [WARN] $*" | tee -a "${LOG_FILE}" >&2
}

log_error() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] [ERROR] $*" | tee -a "${LOG_FILE}" | tee -a "${ERROR_LOG}" >&2
}

log_success() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] [SUCCESS] $*" | tee -a "${LOG_FILE}"
}

# =============================================================================
# VALIDATION
# =============================================================================

validate_binary() {
    if [[ ! -f "${SCRIPT_DIR}/build/gmgn_logger" ]]; then
        log_error "Binary not found: ${SCRIPT_DIR}/build/gmgn_logger"
        log_error "Run 'make' to build the logger first"
        return 1
    fi
    
    if [[ ! -x "${SCRIPT_DIR}/build/gmgn_logger" ]]; then
        log_error "Binary not executable: ${SCRIPT_DIR}/build/gmgn_logger"
        return 1
    fi
    
    log_info "Binary validated: build/gmgn_logger"
    return 0
}

validate_cookies() {
    local missing=0
    
    if [[ -z "${GMGN_CF_CLEARANCE}" ]]; then
        log_warn "GMGN_CF_CLEARANCE not set - API calls may fail (Cloudflare block)"
        missing=1
    fi
    
    # Check if cookie looks expired (basic check)
    if [[ -n "${GMGN_CF_CLEARANCE}" ]]; then
        # cf_clearance contains timestamp, check if it's recent
        local cookie_timestamp=$(echo "${GMGN_CF_CLEARANCE}" | grep -oP '\d{10}' | head -1)
        local now=$(date +%s)
        if [[ -n "${cookie_timestamp}" ]]; then
            local age=$((now - cookie_timestamp))
            if [[ ${age} -gt 86400 ]]; then
                log_warn "cf_clearance cookie may be expired (${age}s old)"
                log_warn "Consider refreshing cookies from browser"
            fi
        fi
    fi
    
    return ${missing}
}

# =============================================================================
# CLOUDFLARE SESSION COOKIES
# =============================================================================

# Extract cf_clearance from your browser cookies
# Firefox: Dev Tools > Storage > Cookies > gmgn.ai > cf_clearance
# Chrome: Dev Tools > Application > Cookies > gmgn.ai > cf_clearance
export GMGN_CF_CLEARANCE="CMR_Gfxo4GohZAESLxRyxLqUVvkxjRgovw5kod.op9I-1766234079-1.2.1.1-anez50vK3VE9yX3GVPJ6SH49MhRq15arbxqx9nUofCaauv14nbp.T9chtSKiSPhAX6HqIdw69ysF_f9NQpO_vsWhLOA9..iVQkDYz6mBebYfPBGojPvAgeOF_PsAvCbpAKMiCOiamFybcGXsdbconGSt_NFV2wS88l8ailEEM.FLgWizhOokjXzrzSaMwif.kfMZ_ZX.mc06Me6gr_XptPtLOL.HTqERMIvTLNS2lLQ"

# Optional: Set other cookies for better compatibility
export GMGN_GA="GA1.1.1216464152.1766234082"
export GMGN_GA_SESSION="GS1.1.1766242908.4.1.1766245615.56.0.0"
export GMGN_CF_BM="Uq4zhFQAQXqH9RWmM3PSwi7gf2C2jREKOrQVQruUxrM-1766245715-1.0.1.1-_fIf9wp3EdR1R6Z1s3fYLSjwQrFM3AotLSrXxr8txxiBzUP9Xe15hoU3DfsM37ooHOkVhrsAXaYh8XCNIOzBAknQRHbfN26VRxpc4EHRL3I"

# =============================================================================
# MAIN
# =============================================================================

main() {
    log_info "=========================================="
    log_info "GMGN Token Logger Starting"
    log_info "=========================================="
    log_info "Log file: ${LOG_FILE}"
    log_info "Error log: ${ERROR_LOG}"
    
    # Validate binary
    if ! validate_binary; then
        log_error "Startup failed: binary validation"
        exit 1
    fi
    
    # Validate cookies
    validate_cookies
    
    log_info "cf_clearance: ${GMGN_CF_CLEARANCE:0:50}..."
    log_info "Starting logger with args: $*"
    echo ""

    # Track start time for error suppression window
    local start_time=$(date +%s)

    # Run the logger, capture stderr for error logging
    "${SCRIPT_DIR}/build/gmgn_logger" "$@" 2> >(while read -r line; do
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))

        # Suppress all errors for first 5 seconds (libwebsockets initialization noise)
        if [[ ${elapsed} -ge 5 ]]; then
            # Filter out libwebsockets noise, log real errors
            if [[ ! "${line}" =~ ^(lwsl_|lws_) ]]; then
                log_error "${line}"
            fi
        fi
    done)
    
    local exit_code=$?
    
    if [[ ${exit_code} -eq 0 ]]; then
        log_info "Logger exited normally"
    elif [[ ${exit_code} -eq 130 ]]; then
        log_info "Logger stopped by user (Ctrl+C)"
    else
        log_error "Logger exited with code: ${exit_code}"
    fi
    
    log_info "=========================================="
    log_info "Session ended"
    log_info "=========================================="
    
    exit ${exit_code}
}

# Run main with all arguments
main "$@"
