# GMGN Trenches Token Logger - Makefile
#
# Build configuration for the GMGN logger application.
#
# Usage:
#   make          - Build release version
#   make debug    - Build with debug symbols and sanitizers
#   make clean    - Remove build artifacts
#   make install  - Install to /usr/local/bin
#   make deps     - Install system dependencies (Ubuntu/Debian)
#   make deps-mac - Install system dependencies (macOS via Homebrew)
#
# @date 2025-12-20

# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -Werror -pedantic -std=c11 -O2
CFLAGS += -D_POSIX_C_SOURCE=200809L -D_GNU_SOURCE
DEBUG_FLAGS = -g -O0 -DDEBUG -fsanitize=address -fsanitize=undefined
LDFLAGS = -lwebsockets -lcjson -lssl -lcrypto -lpthread -lcurl

# macOS detection and Homebrew paths
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    # Detect Homebrew prefix (Apple Silicon vs Intel)
    BREW_PREFIX := $(shell brew --prefix 2>/dev/null || echo /usr/local)
    OPENSSL_PREFIX := $(BREW_PREFIX)/opt/openssl@3
    CFLAGS += -I$(BREW_PREFIX)/include -I$(OPENSSL_PREFIX)/include
    LDFLAGS := -L$(BREW_PREFIX)/lib -L$(OPENSSL_PREFIX)/lib $(LDFLAGS)
endif

# Directories
SRC_DIR = src
INC_DIR = include
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/obj

# Source files
SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRCS))

# Target
TARGET = $(BUILD_DIR)/gmgn_logger

# Default target
.PHONY: all
all: $(TARGET)

# Create directories
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Compile source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -I$(INC_DIR) -c $< -o $@

# Link target
$(TARGET): $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(OBJS) -o $@ $(LDFLAGS)
	@echo ""
	@echo "Build successful: $(TARGET)"
	@echo ""

# Debug build
.PHONY: debug
debug: CFLAGS += $(DEBUG_FLAGS)
debug: LDFLAGS += -fsanitize=address -fsanitize=undefined
debug: clean $(TARGET)
	@echo "Debug build complete with sanitizers enabled"

# Clean build artifacts
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
	@echo "Build directory cleaned"

# Install to system
.PHONY: install
install: $(TARGET)
	install -m 755 $(TARGET) /usr/local/bin/gmgn_logger
	@echo "Installed to /usr/local/bin/gmgn_logger"

# Uninstall from system
.PHONY: uninstall
uninstall:
	rm -f /usr/local/bin/gmgn_logger
	@echo "Uninstalled gmgn_logger"

# Install dependencies (Ubuntu/Debian)
.PHONY: deps
deps:
	@echo "Installing dependencies..."
	sudo apt-get update
	sudo apt-get install -y \
		libwebsockets-dev \
		libcjson-dev \
		libssl-dev \
		libcurl4-openssl-dev \
		build-essential
	@echo "Dependencies installed"

# Install dependencies (macOS via Homebrew)
.PHONY: deps-mac
deps-mac:
	@echo "Installing dependencies via Homebrew..."
	brew install libwebsockets cjson curl openssl@3
	@echo "Dependencies installed"

# Check if dependencies are available
.PHONY: check-deps
check-deps:
	@echo "Checking dependencies..."
	@pkg-config --exists libwebsockets && echo "  libwebsockets: OK" || echo "  libwebsockets: MISSING"
	@pkg-config --exists libcjson && echo "  libcjson: OK" || echo "  libcjson: MISSING"
	@pkg-config --exists openssl && echo "  openssl: OK" || echo "  openssl: MISSING"
	@echo ""

# Run the application
.PHONY: run
run: $(TARGET)
	$(TARGET)

# Run with verbose output
.PHONY: run-verbose
run-verbose: $(TARGET)
	$(TARGET) -v -v

# Format source code (requires clang-format)
.PHONY: format
format:
	find $(SRC_DIR) $(INC_DIR) -name "*.c" -o -name "*.h" | xargs clang-format -i

# Static analysis (requires cppcheck)
.PHONY: analyze
analyze:
	cppcheck --enable=all --std=c11 -I$(INC_DIR) $(SRC_DIR)

# Help
.PHONY: help
help:
	@echo "GMGN Logger - Build Targets"
	@echo ""
	@echo "  make          - Build release version"
	@echo "  make debug    - Build with debug symbols and sanitizers"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make install  - Install to /usr/local/bin"
	@echo "  make deps     - Install system dependencies (Ubuntu/Debian)"
	@echo "  make run      - Build and run"
	@echo "  make run-verbose - Build and run with verbose output"
	@echo "  make check-deps  - Check if dependencies are installed"
	@echo "  make format   - Format source code"
	@echo "  make analyze  - Run static analysis"
	@echo "  make help     - Show this help"
	@echo ""
