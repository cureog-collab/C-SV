# This Makefile is configured for local testing. It compiles both:
#   1. Library source files (in src/)
#   2. Local test files (in test/, containing main.c)
# It then links them with libtenCor.a into a single executable: bin/run_dataframe
#
# The test/ directory is untracked (.gitignore). To use this repository as 
# a standalone library, remove TEST_DIR build rules and reconfigure 
# the target to build a .a or .so library instead.

CC = gcc
CFLAGS = -Wall -Wextra -O3 -I./include
LDFLAGS = -L./lib -ltenCor -lm

SRC_DIR = src
TEST_DIR = test
OBJ_DIR = obj
BIN_DIR = bin

SRCS = $(wildcard $(SRC_DIR)/*.c)
TEST_SRCS = $(wildcard $(TEST_DIR)/*.c)

OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/$(SRC_DIR)/%.o, $(SRCS))
TEST_OBJS = $(patsubst $(TEST_DIR)/%.c, $(OBJ_DIR)/$(TEST_DIR)/%.o, $(TEST_SRCS))

TARGET = $(BIN_DIR)/run_dataframe

.PHONY: all clean run directories

all: directories $(TARGET)

directories:
	@mkdir -p $(OBJ_DIR)/$(SRC_DIR)
	@mkdir -p $(OBJ_DIR)/$(TEST_DIR)
	@mkdir -p $(BIN_DIR)

$(TARGET): $(OBJS) $(TEST_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJ_DIR)/$(SRC_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/$(TEST_DIR)/%.o: $(TEST_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

run: all
	./$(TARGET)