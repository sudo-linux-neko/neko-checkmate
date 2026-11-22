CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -mtune=native -funroll-loops -msse4.1 -mpopcnt -flto
CXXFLAGS_DEBUG = -std=c++17 -g -O0 -fsanitize=address
LDFLAGS = -lpthread -flto

SRC_DIR = src
BUILD_DIR = build
TARGET = neko_meow

SOURCES = $(SRC_DIR)/main.cpp \
          $(SRC_DIR)/bitboard.cpp \
          $(SRC_DIR)/board.cpp \
          $(SRC_DIR)/movegen.cpp \
          $(SRC_DIR)/nnue.cpp \
          $(SRC_DIR)/search.cpp \
          $(SRC_DIR)/evaluate.cpp \
          $(SRC_DIR)/selfplay.cpp \
          $(SRC_DIR)/training.cpp \
          $(SRC_DIR)/uci.cpp

OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

.PHONY: all clean debug pgo-build pgo-use pgo clean-pgo

all: $(TARGET)

$(TARGET): $(BUILD_DIR) $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJECTS) $(LDFLAGS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

debug: CXXFLAGS = $(CXXFLAGS_DEBUG)
debug: LDFLAGS = -lpthread
debug: clean $(TARGET)

clean:
	rm -rf $(BUILD_DIR) $(TARGET) *.gcda *.gcno *.profraw prof.dat

# ============================================================
# PGO (Profile-Guided Optimization) build
# ============================================================
# Usage: make pgo
# This runs: generate profile -> bench -> optimized build
pgo: clean
	@echo "=== Step 1: Building instrumented binary ==="
	$(CXX) -std=c++17 -O3 -march=native -mtune=native -funroll-loops -msse4.1 -mpopcnt -fprofile-generate -flto -o $(TARGET)_instr $(SOURCES) $(LDFLAGS)
	@echo "=== Step 2: Running bench to generate profile ==="
	./$(TARGET)_instr bench > /dev/null 2>&1 || true
	./$(TARGET)_instr bench
	@echo "=== Step 3: Building optimized binary with profile ==="
	$(CXX) -std=c++17 -O3 -march=native -mtune=native -funroll-loops -msse4.1 -mpopcnt -fprofile-use -flto -o $(TARGET) $(SOURCES) $(LDFLAGS)
	rm -f $(TARGET)_instr *.gcda *.gcno
	@echo "=== PGO build complete ==="

clean-pgo:
	rm -f *.gcda *.gcno *.profraw prof.dat $(TARGET)_instr

fast:
	$(CXX) -std=c++17 -O1 -o $(TARGET) $(SOURCES) $(LDFLAGS)
