cmake_minimum_required(VERSION 3.20)

project(Synapse LANGUAGES CXX)

option(SYNAPSE_BUILD_TESTS "Build Synapse tests" ON)
option(SYNAPSE_BUILD_BENCHMARKS "Build Synapse benchmarks" ON)

add_library(synapse
    src/memory_pool.cpp
    src/profiler.cpp
    src/simd.cpp
    src/tensor.cpp
)

target_include_directories(synapse PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_features(synapse PUBLIC cxx_std_17)

if(SYNAPSE_BUILD_TESTS)
    enable_testing()
    add_executable(synapse_tests tests/test_tensor.cpp)
    target_link_libraries(synapse_tests PRIVATE synapse)
    add_test(NAME synapse_tests COMMAND synapse_tests)
endif()

if(SYNAPSE_BUILD_BENCHMARKS)
    add_executable(synapse_bench benchmarks/benchmark_tensor.cpp)
    target_link_libraries(synapse_bench PRIVATE synapse)
endif()
