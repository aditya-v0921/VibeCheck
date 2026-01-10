cmake_minimum_required(VERSION 3.16)
project(vibe_check LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Find required packages
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "pybind11 version: ${pybind11_VERSION}")

# Build Python extension module: vibe_core
pybind11_add_module(vibe_core vibeCheckCore.cpp)

target_include_directories(vibe_core PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(vibe_core PRIVATE ${OpenCV_LIBS})

# Enable link-time optimization for release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set_property(TARGET vibe_core PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# Installation
install(TARGETS vibe_core LIBRARY DESTINATION .)