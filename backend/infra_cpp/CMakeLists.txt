cmake_minimum_required(VERSION 3.14)
project(RecallicVectorEngine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ----------------------------------------------------------------------------
# SIMD backend for the dot product (see vector_engine.cpp):
#   ARM64  -> NEON (always available)
#   x86-64 -> AVX2+FMA when the compiler supports it, otherwise scalar
# The selected backend is reported at runtime by GET /status and the tools.
# ----------------------------------------------------------------------------
include(CheckCXXCompilerFlag)
set(SIMD_FLAGS "")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    check_cxx_compiler_flag("-mavx2 -mfma" HAS_AVX2_FMA)
    if(HAS_AVX2_FMA AND NOT RECALLIC_DISABLE_AVX2)
        set(SIMD_FLAGS -mavx2 -mfma)
    endif()
endif()

find_package(Threads REQUIRED)

add_library(recallic_index STATIC vector_engine.cpp)
target_compile_options(recallic_index PUBLIC ${SIMD_FLAGS})
target_link_libraries(recallic_index PUBLIC Threads::Threads)

add_executable(memory_service main.cpp)
target_link_libraries(memory_service PRIVATE recallic_index)

add_executable(benchmark_hnsw benchmark_hnsw.cpp)
target_link_libraries(benchmark_hnsw PRIVATE recallic_index)

enable_testing()
add_executable(test_vector_engine test_vector_engine.cpp)
target_link_libraries(test_vector_engine PRIVATE recallic_index)
add_test(NAME vector_engine COMMAND test_vector_engine)
# A hang (e.g. a lock that starves writers) must fail fast instead of eating CI time.
set_tests_properties(vector_engine PROPERTIES TIMEOUT 300)

# Same tests under ThreadSanitizer (data-race detection).
option(RECALLIC_TSAN "Build the ThreadSanitizer test target" ON)
if(RECALLIC_TSAN)
    add_executable(test_vector_engine_tsan test_vector_engine.cpp vector_engine.cpp)
    target_compile_options(test_vector_engine_tsan PRIVATE -fsanitize=thread -O1 -g ${SIMD_FLAGS})
    target_link_options(test_vector_engine_tsan PRIVATE -fsanitize=thread)
    target_link_libraries(test_vector_engine_tsan PRIVATE Threads::Threads)
    add_test(NAME vector_engine_tsan COMMAND test_vector_engine_tsan)
    set_tests_properties(vector_engine_tsan PROPERTIES TIMEOUT 900)
endif()
