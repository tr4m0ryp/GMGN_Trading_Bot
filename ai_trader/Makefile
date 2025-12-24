# AI Trader Makefile
# Build system for the AI trading bot with paper trading and Jito integration

# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -Werror -O2 -std=c11 -D_GNU_SOURCE
CFLAGS += -I./include -I../logger_c/include
LDFLAGS = -lpthread -lcurl -lcjson -lm -lsodium

# Debug build
DEBUG_CFLAGS = -g -O0 -DDEBUG -fsanitize=address

# Directories
SRC_DIR = src
BUILD_DIR = build
INCLUDE_DIR = include

# Source files organized by category
CORE_SRCS = $(SRC_DIR)/core/config_loader.c \
            $(SRC_DIR)/core/trade_logger.c \
            $(SRC_DIR)/core/paper_trader.c \
            $(SRC_DIR)/core/model_loader.c \
            $(SRC_DIR)/core/ai_trader.c

INTEGRATION_SRCS = $(SRC_DIR)/integrations/chart_fetcher.c \
                   $(SRC_DIR)/integrations/jito_client.c \
                   $(SRC_DIR)/integrations/logger_integration.c \
                   $(SRC_DIR)/integrations/telegram_bot.c

CRYPTO_SRCS = $(SRC_DIR)/crypto/base58.c \
              $(SRC_DIR)/crypto/solana_wallet.c

SRCS = $(CORE_SRCS) $(INTEGRATION_SRCS) $(CRYPTO_SRCS)

# Object files (flatten to build dir)
OBJS = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRCS))
DEPS = $(OBJS:.o=.d)

# Main targets
TARGET = $(BUILD_DIR)/ai_trader
TEST_CONFIG = $(BUILD_DIR)/test_config
TEST_PAPER = $(BUILD_DIR)/test_paper
TEST_CHART = $(BUILD_DIR)/test_chart

# Default target
all: $(TARGET)

# Main executable (placeholder - will add main.c)
$(TARGET): $(OBJS)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Object files
$(BUILD_DIR)/core/%.o: $(SRC_DIR)/core/%.c
	@mkdir -p $(BUILD_DIR)/core
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/integrations/%.o: $(SRC_DIR)/integrations/%.c
	@mkdir -p $(BUILD_DIR)/integrations
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/crypto/%.o: $(SRC_DIR)/crypto/%.c
	@mkdir -p $(BUILD_DIR)/crypto
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

# Test targets
test_config: $(BUILD_DIR)/config_loader.o
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -DTEST_CONFIG_LOADER \
		$(SRC_DIR)/config_loader.c \
		-o $(TEST_CONFIG) $(LDFLAGS)
	@echo "Running config loader test..."
	@./$(TEST_CONFIG) || true

test_paper: $(BUILD_DIR)/paper_trader.o $(BUILD_DIR)/trade_logger.o
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -DTEST_PAPER_TRADER \
		$(SRC_DIR)/paper_trader.c \
		$(SRC_DIR)/trade_logger.c \
		-o $(TEST_PAPER) $(LDFLAGS)
	@echo "Running paper trader test..."
	@./$(TEST_PAPER) || true

test_chart: $(BUILD_DIR)/chart_fetcher.o $(BUILD_DIR)/config_loader.o
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -DTEST_CHART_FETCHER \
		$(SRC_DIR)/chart_fetcher.c \
		$(SRC_DIR)/config_loader.c \
		-o $(TEST_CHART) $(LDFLAGS)
	@echo "Running chart fetcher test..."
	@./$(TEST_CHART) || true

# Debug build
debug: CFLAGS += $(DEBUG_CFLAGS)
debug: clean all

# Clean
clean:
	rm -rf $(BUILD_DIR)

# Install dependencies (Debian/Ubuntu)
deps:
	@echo "Installing dependencies..."
	sudo apt-get update
	sudo apt-get install -y libcurl4-openssl-dev libcjson-dev

# Help
help:
	@echo "AI Trader Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all         - Build main executable"
	@echo "  debug       - Build with debug symbols and sanitizers"
	@echo "  test_config - Test configuration loader"
	@echo "  test_paper  - Test paper trader"
	@echo "  test_chart  - Test chart fetcher"
	@echo "  clean       - Remove build artifacts"
	@echo "  deps        - Install system dependencies"

# Include dependency files
-include $(DEPS)

.PHONY: all debug clean deps help test_config test_paper test_chart
