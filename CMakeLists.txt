cmake_minimum_required(VERSION 3.18)
project(symbolic_autodiff LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# LibTorch: pass -DCMAKE_PREFIX_PATH=/path/to/libtorch or set Torch_DIR.
# scripts/setup_libtorch.sh downloads a CPU build into third_party/libtorch.
if(NOT DEFINED CMAKE_PREFIX_PATH AND EXISTS "${CMAKE_SOURCE_DIR}/third_party/libtorch")
  list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/third_party/libtorch")
endif()
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_compile_options(-Wall -Wextra -Wno-unused-parameter)

# Torch-free core: IR, hash-consing, type inference, builder, analysis,
# autodiff, SLP/liveness, tree baseline, benchmark utilities.
add_library(sad_core STATIC
  src/ir.cpp
  src/hashcons.cpp
  src/type_inference.cpp
  src/graph_builder.cpp
  src/analysis.cpp
  src/autodiff.cpp
  src/slp.cpp
  src/tree_ast.cpp
  src/tasks.cpp
  src/bench.cpp)
target_include_directories(sad_core PUBLIC include)

# ATen/LibTorch dependent part: executor, profiler, autograd baseline.
add_library(sad_exec STATIC
  src/executor.cpp
  src/profiler.cpp
  src/torch_baseline.cpp)
target_include_directories(sad_exec PUBLIC include)
target_link_libraries(sad_exec PUBLIC sad_core "${TORCH_LIBRARIES}")

add_executable(sad_tests tests/test_main.cpp tests/test_graph.cpp tests/test_numeric.cpp)
target_link_libraries(sad_tests PRIVATE sad_exec)

add_executable(sad_bench benchmarks/bench_main.cpp)
target_link_libraries(sad_bench PRIVATE sad_exec)

enable_testing()
add_test(NAME sad_tests COMMAND sad_tests)
