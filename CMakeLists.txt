cmake_minimum_required(VERSION 3.14)
project(NekoMeow LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags for optimization
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -funroll-loops -flto")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# Default to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Source files
set(SOURCES
    src/main.cpp
    src/bitboard.cpp
    src/board.cpp
    src/nnue.cpp
    src/search.cpp
    src/evaluate.cpp
    src/movegen.cpp
    src/selfplay.cpp
    src/training.cpp
    src/uci.cpp
)

# Create executable
add_executable(neko_meow ${SOURCES})

target_include_directories(neko_meow PRIVATE src)

# Link pthreads for std::thread
find_package(Threads REQUIRED)
target_link_libraries(neko_meow PRIVATE Threads::Threads)

# Enable SIMD optimizations
target_compile_options(neko_meow PRIVATE -msse4.1 -mpopcnt)

# Install
install(TARGETS neko_meow DESTINATION bin)
