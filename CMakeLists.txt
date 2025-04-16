cmake_minimum_required(VERSION 3.0...3.26)

project(bnsw LANGUAGES CXX)

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Try to find Eigen3 in standard locations
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/usr/include/eigen3" "/usr/local/include/eigen3")

# Detect host architecture and set appropriate compiler flags
if(CMAKE_SYSTEM_PROCESSOR MATCHES "amd64|x86_64")
  set(ARCH_FLAGS "-march=native")
  message(STATUS "Detected x86_64 architecture, enabling AVX2 instructions")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  set(ARCH_FLAGS "-march=native")
  message(STATUS "Detected ARM64 architecture, enabling native optimizations")
else()
  set(ARCH_FLAGS "-march=native")
  message(STATUS "Unknown architecture: ${CMAKE_SYSTEM_PROCESSOR}, using generic optimizations")
endif()

# Use Conan-generated paths if available
find_package(spdlog REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(Catch2 REQUIRED)
find_package(fmt REQUIRED)

# Automatically find all test files using glob patterns
file(GLOB_RECURSE TEST_FILES "test/*.cpp")
message(STATUS "Found test files: ${TEST_FILES}")

# Add a combined all_tests target
add_executable(all_tests ${TEST_FILES})
target_link_libraries(all_tests Catch2::Catch2WithMain Eigen3::Eigen spdlog::spdlog fmt::fmt)
target_include_directories(all_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(all_tests PRIVATE -Wall -Wextra -Wpedantic -Werror ${ARCH_FLAGS} -g)

add_executable(utils_test test/utils_test.cpp)
target_link_libraries(utils_test Catch2::Catch2WithMain Eigen3::Eigen)
target_include_directories(utils_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(utils_test PRIVATE -Wall -Wextra -Wpedantic -Werror ${ARCH_FLAGS}  -g)

add_executable(l2_distance_test test/dist_alg/l2_distance_test.cpp)
target_link_libraries(l2_distance_test Catch2::Catch2WithMain Eigen3::Eigen)
target_include_directories(l2_distance_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(l2_distance_test PRIVATE -Wall -Wextra -Wpedantic -Werror ${ARCH_FLAGS} -g)

add_executable(adsampling_test test/adsampling_test.cpp)
target_link_libraries(adsampling_test Catch2::Catch2WithMain Eigen3::Eigen)
target_include_directories(adsampling_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(adsampling_test PRIVATE -Wall -Wextra -Wpedantic -Werror ${ARCH_FLAGS} -g)

add_executable(bnsw_test test/bnsw_test.cpp)
target_link_libraries(bnsw_test Catch2::Catch2WithMain Eigen3::Eigen)
target_include_directories(bnsw_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(bnsw_test PRIVATE -Wall -Wextra -Wpedantic ${ARCH_FLAGS} -g -O0)



add_executable(sift1m demo/sift1m.cpp)
target_link_libraries(sift1m Eigen3::Eigen spdlog::spdlog)
target_include_directories(sift1m PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(sift1m PRIVATE -Wall -Wextra -Wpedantic ${ARCH_FLAGS} -g -O3)

add_executable(adsampling_bench benchmark/adsampling.cpp)
target_link_libraries(adsampling_bench Catch2::Catch2WithMain Eigen3::Eigen spdlog::spdlog)
target_include_directories(adsampling_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(adsampling_bench PRIVATE -Wall -Wextra -Wpedantic ${ARCH_FLAGS} -g -O2)