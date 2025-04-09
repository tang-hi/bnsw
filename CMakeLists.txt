cmake_minimum_required(VERSION 3.0...3.26)

project(bnsw LANGUAGES CXX)

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Try to find Eigen3 in standard locations
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/usr/include/eigen3" "/usr/local/include/eigen3")

# Use Conan-generated paths if available
find_package(spdlog REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(Catch2 REQUIRED)

add_executable(utils_test test/utils_test.cpp)
target_link_libraries(utils_test Catch2::Catch2WithMain Eigen3::Eigen)
target_include_directories(utils_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(l2_distance_test test/dist_alg/l2_distance_test.cpp)
target_link_libraries(l2_distance_test Catch2::Catch2WithMain Eigen3::Eigen)
target_include_directories(l2_distance_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(l2_distance_test PRIVATE -Wall -Wextra -Wpedantic -Werror -mavx2 -march=native)
