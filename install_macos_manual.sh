#!/bin/bash
#
# macOS Manual Install Script (No Homebrew)
#
# Downloads and builds all dependencies from source.
# Works on macOS 10.13 High Sierra and later.
#
# Usage:
#   ./install_macos_manual.sh
#
# @date 2025-12-20

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPS_DIR="$SCRIPT_DIR/deps"
BUILD_DIR="$DEPS_DIR/build"
INSTALL_DIR="$DEPS_DIR/local"

# Dependency versions
OPENSSL_VERSION="1.1.1w"
CJSON_VERSION="1.7.17"
LIBWEBSOCKETS_VERSION="4.3.3"

echo "========================================"
echo "  GMGN Trading Bot - Manual Installer"
echo "  (No Homebrew Required)"
echo "========================================"
echo ""

# Check if running on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "Error: This script is for macOS only"
    exit 1
fi

# Check for Xcode Command Line Tools
echo "[1/6] Checking Xcode Command Line Tools..."
if ! xcode-select -p &>/dev/null; then
    echo "Installing Xcode Command Line Tools..."
    xcode-select --install
    echo ""
    echo "Please complete the Xcode installation dialog, then re-run this script."
    exit 0
fi
echo "  OK"

# Check for required tools
echo ""
echo "[2/6] Checking required tools..."
for tool in curl tar make cc cmake; do
    if ! command -v $tool &>/dev/null; then
        echo "  Error: $tool not found"
        if [[ "$tool" == "cmake" ]]; then
            echo ""
            echo "  CMake is required. Install it from:"
            echo "  https://cmake.org/download/"
            echo ""
            echo "  Or run: curl -LO https://github.com/Kitware/CMake/releases/download/v3.28.1/cmake-3.28.1-macos-universal.tar.gz"
        fi
        exit 1
    fi
    echo "  $tool: OK"
done

# Create directories
echo ""
echo "[3/6] Setting up directories..."
mkdir -p "$BUILD_DIR"
mkdir -p "$INSTALL_DIR/lib"
mkdir -p "$INSTALL_DIR/include"
echo "  Deps dir: $DEPS_DIR"
echo "  Install dir: $INSTALL_DIR"

# Download and build OpenSSL
echo ""
echo "[4/6] Building OpenSSL $OPENSSL_VERSION..."
cd "$BUILD_DIR"

if [[ ! -f "$INSTALL_DIR/lib/libssl.a" ]]; then
    if [[ ! -d "openssl-$OPENSSL_VERSION" ]]; then
        echo "  Downloading..."
        curl -LO "https://www.openssl.org/source/openssl-$OPENSSL_VERSION.tar.gz"
        tar xzf "openssl-$OPENSSL_VERSION.tar.gz"
    fi
    
    cd "openssl-$OPENSSL_VERSION"
    echo "  Configuring..."
    ./Configure darwin64-x86_64-cc --prefix="$INSTALL_DIR" no-shared 2>&1 | tail -3
    echo "  Building (this may take a few minutes)..."
    make -j$(sysctl -n hw.ncpu) 2>&1 | tail -5
    echo "  Installing..."
    make install_sw 2>&1 | tail -3
    echo "  OpenSSL: OK"
else
    echo "  OpenSSL: already built"
fi

# Download and build cJSON
echo ""
echo "[5/6] Building cJSON $CJSON_VERSION..."
cd "$BUILD_DIR"

if [[ ! -f "$INSTALL_DIR/lib/libcjson.a" ]]; then
    if [[ ! -d "cJSON-$CJSON_VERSION" ]]; then
        echo "  Downloading..."
        curl -LO "https://github.com/DaveGamble/cJSON/archive/refs/tags/v$CJSON_VERSION.tar.gz"
        tar xzf "v$CJSON_VERSION.tar.gz"
    fi
    
    cd "cJSON-$CJSON_VERSION"
    mkdir -p build && cd build
    echo "  Configuring..."
    cmake .. -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
             -DENABLE_CJSON_TEST=OFF \
             -DBUILD_SHARED_LIBS=OFF \
             -DBUILD_SHARED_AND_STATIC_LIBS=OFF 2>&1 | tail -3
    echo "  Building..."
    make -j$(sysctl -n hw.ncpu) 2>&1 | tail -3
    echo "  Installing..."
    make install 2>&1 | tail -3
    echo "  cJSON: OK"
else
    echo "  cJSON: already built"
fi

# Download and build libwebsockets
echo ""
echo "[6/6] Building libwebsockets $LIBWEBSOCKETS_VERSION..."
cd "$BUILD_DIR"

if [[ ! -f "$INSTALL_DIR/lib/libwebsockets.a" ]]; then
    if [[ ! -d "libwebsockets-$LIBWEBSOCKETS_VERSION" ]]; then
        echo "  Downloading..."
        curl -LO "https://github.com/warmcat/libwebsockets/archive/refs/tags/v$LIBWEBSOCKETS_VERSION.tar.gz"
        tar xzf "v$LIBWEBSOCKETS_VERSION.tar.gz"
    fi
    
    cd "libwebsockets-$LIBWEBSOCKETS_VERSION"
    mkdir -p build && cd build
    echo "  Configuring..."
    cmake .. -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
             -DOPENSSL_ROOT_DIR="$INSTALL_DIR" \
             -DOPENSSL_INCLUDE_DIR="$INSTALL_DIR/include" \
             -DOPENSSL_CRYPTO_LIBRARY="$INSTALL_DIR/lib/libcrypto.a" \
             -DOPENSSL_SSL_LIBRARY="$INSTALL_DIR/lib/libssl.a" \
             -DLWS_WITH_SSL=ON \
             -DLWS_WITH_SHARED=OFF \
             -DLWS_WITHOUT_TESTAPPS=ON \
             -DLWS_WITHOUT_TEST_SERVER=ON \
             -DLWS_WITHOUT_TEST_CLIENT=ON 2>&1 | tail -5
    echo "  Building..."
    make -j$(sysctl -n hw.ncpu) 2>&1 | tail -3
    echo "  Installing..."
    make install 2>&1 | tail -3
    echo "  libwebsockets: OK"
else
    echo "  libwebsockets: already built"
fi

# Build the project
echo ""
echo "========================================"
echo "  Building GMGN Trading Bot"
echo "========================================"

# Set compiler flags
CFLAGS="-Wall -Wextra -Werror -pedantic -std=c11 -O2"
CFLAGS="$CFLAGS -D_POSIX_C_SOURCE=200809L -D_GNU_SOURCE"
CFLAGS="$CFLAGS -I$INSTALL_DIR/include"

LDFLAGS="-L$INSTALL_DIR/lib"
LDFLAGS="$LDFLAGS -lwebsockets -lcjson -lssl -lcrypto"
LDFLAGS="$LDFLAGS -lpthread -lcurl -lz"

# macOS needs these frameworks
LDFLAGS="$LDFLAGS -framework CoreFoundation -framework Security"

echo ""
echo "Building logger_c..."
cd "$SCRIPT_DIR/logger_c"

# Clean
rm -rf build
mkdir -p build/obj

# Compile source files
for src in src/*.c; do
    obj="build/obj/$(basename "$src" .c).o"
    echo "  Compiling $(basename "$src")..."
    cc $CFLAGS -Iinclude -c "$src" -o "$obj"
done

# Link
echo "  Linking..."
cc $CFLAGS build/obj/*.o -o build/gmgn_logger $LDFLAGS

if [[ -f build/gmgn_logger ]]; then
    echo "  logger_c: BUILD SUCCESS"
else
    echo "  logger_c: BUILD FAILED"
    exit 1
fi

echo ""
echo "Building ai_data..."
cd "$SCRIPT_DIR/ai_data"

# Clean
rm -rf build
mkdir -p build/obj

# Compile AI source files
for src in src/*.c; do
    obj="build/obj/$(basename "$src" .c).o"
    echo "  Compiling $(basename "$src")..."
    cc $CFLAGS -Iinclude -I../logger_c/include -c "$src" -o "$obj"
done

# Link with logger_c objects (except gmgn_logger.o which has main)
LOGGER_OBJS=""
for obj in ../logger_c/build/obj/*.o; do
    if [[ "$(basename "$obj")" != "gmgn_logger.o" ]]; then
        LOGGER_OBJS="$LOGGER_OBJS $obj"
    fi
done

echo "  Linking..."
cc $CFLAGS build/obj/*.o $LOGGER_OBJS -o build/ai_data_logger $LDFLAGS

if [[ -f build/ai_data_logger ]]; then
    echo "  ai_data: BUILD SUCCESS"
else
    echo "  ai_data: BUILD FAILED"
    exit 1
fi

# Create run script with library paths
echo ""
echo "Creating run scripts..."

cat > "$SCRIPT_DIR/logger_c/run_macos.sh" << 'RUNSCRIPT'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DYLD_LIBRARY_PATH="$SCRIPT_DIR/../deps/local/lib:$DYLD_LIBRARY_PATH"
exec "$SCRIPT_DIR/build/gmgn_logger" "$@"
RUNSCRIPT
chmod +x "$SCRIPT_DIR/logger_c/run_macos.sh"

cat > "$SCRIPT_DIR/ai_data/run_macos.sh" << 'RUNSCRIPT'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DYLD_LIBRARY_PATH="$SCRIPT_DIR/../deps/local/lib:$DYLD_LIBRARY_PATH"
mkdir -p "$SCRIPT_DIR/data"
exec "$SCRIPT_DIR/build/ai_data_logger" --data-dir "$SCRIPT_DIR/data" "$@"
RUNSCRIPT
chmod +x "$SCRIPT_DIR/ai_data/run_macos.sh"

# Done
echo ""
echo "========================================"
echo "  Installation Complete!"
echo "========================================"
echo ""
echo "Dependencies installed to: $INSTALL_DIR"
echo ""
echo "To run the logger:"
echo "  cd $SCRIPT_DIR/logger_c"
echo "  ./run_macos.sh"
echo ""
echo "To run the AI data collector:"
echo "  cd $SCRIPT_DIR/ai_data"
echo "  ./run_macos.sh"
echo ""
echo "Optional: Set Cloudflare cookies for API access:"
echo "  export GMGN_CF_CLEARANCE='your_cookie'"
echo "  export GMGN_CF_BM='your_cookie'"
echo ""
