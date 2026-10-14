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
# CUDA-specific code (caching allocator statistics) is compiled only when the
# LibTorch build ships CUDA libraries; everything else is device agnostic.
if(TORCH_CUDA_LIBRARIES OR TORCH_LIBRARIES MATCHES "cuda")
  message(STATUS "LibTorch has CUDA: enabling SAD_WITH_CUDA")
  add_compile_definitions(SAD_WITH_CUDA=1)
else()
  message(STATUS "LibTorch without CUDA: --device cuda will be unavailable")
endif()

if(MSVC)
  add_compile_options(/W4 /wd4100 /wd4244 /wd4267 /wd4251 /wd4275 /bigobj /EHsc)
  add_compile_definitions(NOMINMAX _CRT_SECURE_NO_WARNINGS)
else()
  add_compile_options(-Wall -Wextra -Wno-unused-parameter)
endif()

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

# On Windows the LibTorch DLLs must sit next to the executables.
if(MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  foreach(tgt sad_tests sad_bench)
    add_custom_command(TARGET ${tgt} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS} $<TARGET_FILE_DIR:${tgt}>
      COMMAND_EXPAND_LISTS)
  endforeach()
endif()

enable_testing()
add_test(NAME sad_tests COMMAND sad_tests)

add_executable(sad_scaling benchmarks/scaling_main.cpp)
target_link_libraries(sad_scaling PRIVATE sad_core)
