#!/bin/bash
#
# macOS High Sierra Quick Install Script
#
# Installs all dependencies and builds the GMGN logger and AI data collector.
# Tested on macOS 10.13 High Sierra.
#
# Usage:
#   ./install_macos.sh
#
# Prerequisites:
#   - Xcode Command Line Tools (will be installed if missing)
#   - Homebrew (will be installed if missing)
#
# @date 2025-12-20

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo "  GMGN Trading Bot - macOS Installer"
echo "========================================"
echo ""

# Check if running on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "Error: This script is for macOS only"
    exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Step 1: Install Xcode Command Line Tools if missing
echo "[1/5] Checking Xcode Command Line Tools..."
if ! xcode-select -p &>/dev/null; then
    echo "Installing Xcode Command Line Tools..."
    xcode-select --install
    echo ""
    echo "Please complete the Xcode installation dialog, then re-run this script."
    exit 0
else
    echo "  Xcode Command Line Tools: OK"
fi

# Step 2: Install Homebrew if missing
echo ""
echo "[2/5] Checking Homebrew..."
if ! command_exists brew; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Add Homebrew to PATH for this session
    if [[ -f /opt/homebrew/bin/brew ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    elif [[ -f /usr/local/bin/brew ]]; then
        eval "$(/usr/local/bin/brew shellenv)"
    fi
else
    echo "  Homebrew: OK"
    
    # Fix Homebrew permissions for High Sierra
    echo "  Fixing Homebrew permissions..."
    BREW_PREFIX=$(brew --prefix)
    if [[ -d "$BREW_PREFIX/Homebrew" ]]; then
        sudo chown -R $(whoami) "$BREW_PREFIX/Homebrew" 2>/dev/null || true
    fi
    if [[ -d "$BREW_PREFIX/var/homebrew" ]]; then
        sudo chown -R $(whoami) "$BREW_PREFIX/var/homebrew" 2>/dev/null || true
    fi
    if [[ -d "$BREW_PREFIX/Cellar" ]]; then
        sudo chown -R $(whoami) "$BREW_PREFIX/Cellar" 2>/dev/null || true
    fi
    if [[ -d "$BREW_PREFIX/Caskroom" ]]; then
        sudo chown -R $(whoami) "$BREW_PREFIX/Caskroom" 2>/dev/null || true
    fi
    sudo chown -R $(whoami) "$BREW_PREFIX/lib" 2>/dev/null || true
    sudo chown -R $(whoami) "$BREW_PREFIX/include" 2>/dev/null || true
    
    # Update Homebrew
    echo "  Updating Homebrew..."
    brew update 2>/dev/null || echo "  Warning: brew update failed, continuing..."
fi

# Step 3: Install dependencies via Homebrew
echo ""
echo "[3/5] Installing dependencies..."

# Determine Homebrew prefix
if [[ -d /opt/homebrew ]]; then
    BREW_PREFIX="/opt/homebrew"
else
    BREW_PREFIX="/usr/local"
fi

DEPS="libwebsockets cjson curl openssl@1.1"

for dep in $DEPS; do
    if brew list "$dep" &>/dev/null; then
        echo "  $dep: already installed"
    else
        echo "  Installing $dep..."
        brew install "$dep" 2>&1 || {
            echo "  Warning: Failed to install $dep via brew, trying with sudo..."
            sudo chown -R $(whoami) "$BREW_PREFIX" 2>/dev/null || true
            brew install "$dep"
        }
    fi
done

# Step 4: Set up environment for compilation
echo ""
echo "[4/5] Setting up build environment..."

# OpenSSL paths (use 1.1 for High Sierra compatibility)
if [[ -d "$BREW_PREFIX/opt/openssl@1.1" ]]; then
    OPENSSL_PREFIX="$BREW_PREFIX/opt/openssl@1.1"
elif [[ -d "$BREW_PREFIX/opt/openssl@3" ]]; then
    OPENSSL_PREFIX="$BREW_PREFIX/opt/openssl@3"
else
    OPENSSL_PREFIX="$BREW_PREFIX/opt/openssl"
fi

echo "  BREW_PREFIX: $BREW_PREFIX"
echo "  OPENSSL_PREFIX: $OPENSSL_PREFIX"

# Build flags
EXTRA_CFLAGS="-I$BREW_PREFIX/include -I$OPENSSL_PREFIX/include"
EXTRA_LDFLAGS="-L$BREW_PREFIX/lib -L$OPENSSL_PREFIX/lib"

# Step 5: Build projects
echo ""
echo "[5/5] Building projects..."

# Build logger_c
echo ""
echo "Building logger_c..."
cd "$SCRIPT_DIR/logger_c"

make clean 2>/dev/null || true
make CFLAGS="-Wall -Wextra -Werror -pedantic -std=c11 -O2 -D_POSIX_C_SOURCE=200809L -D_GNU_SOURCE -Iinclude $EXTRA_CFLAGS" \
     LDFLAGS="$EXTRA_LDFLAGS -lwebsockets -lcjson -lssl -lcrypto -lpthread -lcurl"

if [[ -f build/gmgn_logger ]]; then
    echo "  logger_c: BUILD SUCCESS"
else
    echo "  logger_c: BUILD FAILED"
    exit 1
fi

# Build ai_data
echo ""
echo "Building ai_data..."
cd "$SCRIPT_DIR/ai_data"

make clean 2>/dev/null || true
make CFLAGS="-Wall -Wextra -Werror -pedantic -std=c11 -O2 -D_POSIX_C_SOURCE=200809L -D_GNU_SOURCE -Iinclude -I../logger_c/include $EXTRA_CFLAGS" \
     LDFLAGS="$EXTRA_LDFLAGS -lwebsockets -lcjson -lssl -lcrypto -lpthread -lcurl"

if [[ -f build/ai_data_logger ]]; then
    echo "  ai_data: BUILD SUCCESS"
else
    echo "  ai_data: BUILD FAILED"
    exit 1
fi

# Done
echo ""
echo "========================================"
echo "  Installation Complete!"
echo "========================================"
echo ""
echo "To run the logger:"
echo "  cd $SCRIPT_DIR/logger_c"
echo "  ./run_logger.sh"
echo ""
echo "To run the AI data collector:"
echo "  cd $SCRIPT_DIR/ai_data"
echo "  ./run"
echo ""
echo "Optional: Set Cloudflare cookies for API access:"
echo "  export GMGN_CF_CLEARANCE='your_cookie'"
echo "  export GMGN_CF_BM='your_cookie'"
echo ""
