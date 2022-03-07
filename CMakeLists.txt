# CMake version
cmake_minimum_required(VERSION 3.12)

# Project name
project(SLucAM VERSION 1.0)

# C++ version
set(CMAKE_CXX_STANDARD 11)

### OPENCV ###
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

### Eigen ###
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Where to put binary and library files
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# Where to find the headers to include with #include <...>
include_directories(${PROJECT_SOURCE_DIR}/include)

# Add the CMakeLists.txt of the src folder
add_subdirectory(src)