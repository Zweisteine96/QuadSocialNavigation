# This file tells the 'cmake' command how to build our project.

# Minimum version of CMake required
cmake_minimum_required(VERSION 3.10)
# Project name
project(SocialVAECpp)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Find the LibTorch Library ---
# This tells CMake to look for LibTorch in the folder we provide during the build step.
find_package(Torch REQUIRED)

# --- Define our Executable ---
# We name our final program 'prediction_app'.
# It is built from these two source files.
add_executable(prediction_app
    src/main.cpp
    src/predictor.cpp
)

# --- Tell our program where to find header files ---
# This adds the 'include' directory to the list of places to search for .hpp files.
target_include_directories(prediction_app PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# --- Link our program against LibTorch ---
# This connects our code to the PyTorch C++ libraries.
target_link_libraries(prediction_app PRIVATE ${TORCH_LIBRARIES})

# This ensures C++ ABI compatibility.
if(NOT ${TORCH_CXX_FLAGS} STREQUAL "")
  target_compile_options(prediction_app PRIVATE ${TORCH_CXX_FLAGS})
endif()