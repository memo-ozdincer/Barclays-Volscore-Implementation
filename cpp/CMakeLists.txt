cmake_minimum_required(VERSION 3.10)
project(volscore_project)

set(CMAKE_CXX_STANDARD 17)

# Add a simple build for the library
add_library(volscore_lib
    include/volscore.hpp
    src/volscore.cpp
)

target_include_directories(volscore_lib PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# Tests
enable_testing()
add_executable(test_volscore tests/test_volscore.cpp)
target_link_libraries(test_volscore PRIVATE volscore_lib)
add_test(NAME VolScoreTests COMMAND test_volscore)
