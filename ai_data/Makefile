# AI Data Collector - Makefile
#
# Build configuration for the AI data collection application.
# Links to logger_c object files for reuse of WebSocket, filtering, etc.
#
# Usage:
#   make          - Build release version
#   make debug    - Build with debug symbols and sanitizers
#   make clean    - Remove build artifacts
#
# Prerequisites:
#   - logger_c must be built first (cd ../logger_c && make)
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
DATA_DIR = data

# Logger_c directories (relative path)
LOGGER_C_DIR = ../logger_c
LOGGER_C_INC = $(LOGGER_C_DIR)/include
LOGGER_C_OBJ = $(LOGGER_C_DIR)/build/obj

# Source files (ai_data specific)
AI_SRCS = $(SRC_DIR)/ai_main.c $(SRC_DIR)/ai_data_collector.c
AI_OBJS = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(AI_SRCS))

# Logger_c object files to link (exclude main)
LOGGER_OBJS = $(LOGGER_C_OBJ)/filter.o \
              $(LOGGER_C_OBJ)/json_parser.o \
              $(LOGGER_C_OBJ)/output.o \
              $(LOGGER_C_OBJ)/token_tracker.o \
              $(LOGGER_C_OBJ)/websocket_client.o

# Target
TARGET = $(BUILD_DIR)/ai_data_logger

# Include paths
INCLUDES = -I$(INC_DIR) -I$(LOGGER_C_INC)

# Default target
.PHONY: all
all: check_logger $(TARGET)

# Check if logger_c is built
.PHONY: check_logger
check_logger:
	@if [ ! -f "$(LOGGER_C_OBJ)/filter.o" ]; then \
		echo "Error: logger_c not built. Run 'make' in ../logger_c first."; \
		exit 1; \
	fi

# Create directories
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(DATA_DIR):
	mkdir -p $(DATA_DIR)

# Compile ai_data source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Link target
$(TARGET): $(AI_OBJS) $(LOGGER_OBJS) | $(BUILD_DIR) $(DATA_DIR)
	$(CC) $(CFLAGS) $(AI_OBJS) $(LOGGER_OBJS) -o $@ $(LDFLAGS)
	@echo ""
	@echo "Build successful: $(TARGET)"
	@echo "Data output dir:  $(DATA_DIR)/"
	@echo ""

# Debug build
.PHONY: debug
debug: CFLAGS += $(DEBUG_FLAGS)
debug: LDFLAGS += -fsanitize=address -fsanitize=undefined
debug: clean $(TARGET)
	@echo "Debug build complete with sanitizers enabled"

# Build logger_c first
.PHONY: build_logger
build_logger:
	$(MAKE) -C $(LOGGER_C_DIR)

# Build everything (logger_c + ai_data)
.PHONY: all_deps
all_deps: build_logger all

# Clean build artifacts
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
	@echo "Build directory cleaned"

# Clean everything including logger_c
.PHONY: clean_all
clean_all: clean
	$(MAKE) -C $(LOGGER_C_DIR) clean

# Run the application
.PHONY: run
run: $(TARGET)
	./run

# Help
.PHONY: help
help:
	@echo "AI Data Collector - Build Targets"
	@echo ""
	@echo "  make              Build ai_data_logger (requires logger_c built)"
	@echo "  make all_deps     Build logger_c first, then ai_data_logger"
	@echo "  make debug        Build with debug symbols and sanitizers"
	@echo "  make clean        Remove ai_data build artifacts"
	@echo "  make clean_all    Remove all build artifacts (including logger_c)"
	@echo "  make run          Run the application"
	@echo "  make help         Show this help"
	@echo ""
	@echo "Prerequisites:"
	@echo "  cd ../logger_c && make"
	@echo ""
