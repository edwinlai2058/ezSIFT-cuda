cmake_minimum_required(VERSION 3.8)
set(CMAKE_CUDA_COMPILER /opt/cuda-12.6/bin/nvcc)
set(CUDAToolkit_ROOT /opt/cuda-12.6)
set(CMAKE_CUDA_ARCHITECTURES 61)


project(ezsift LANGUAGES CXX CUDA)




# 【修改 1】加入這行，讓 CMake 尋找 OpenMP
find_package(OpenMP REQUIRED)

option(EZSIFT_BUILD_EXAMPLES "Build with examples" ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
endforeach()

set(EZSIFT_ROOT ${PROJECT_SOURCE_DIR}/../../)
set(EZSIFT_INCLUDE_DIR ${EZSIFT_ROOT}/include)

# CUDA Compiler Options
# --- Compiler Flags ---
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -pg")
set(CMAKE_BUILD_TYPE Release)


# --- Baseline ---
set(EZSIFT_SRC_BASELINE_DIR ${PROJECT_SOURCE_DIR}/../../src_baseline)
file(GLOB_RECURSE SRC_BASELINE RELATIVE ${PROJECT_SOURCE_DIR} "${EZSIFT_SRC_BASELINE_DIR}/*.cpp")
add_library(ezsift_baseline ${SRC_BASELINE})
set_property(TARGET ezsift_baseline PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(ezsift_baseline PUBLIC ${EZSIFT_SRC_BASELINE_DIR} ${EZSIFT_INCLUDE_DIR})
target_link_libraries(ezsift_baseline PUBLIC OpenMP::OpenMP_CXX)

# --- Optimized ---
set(EZSIFT_SRC_OPT_DIR ${PROJECT_SOURCE_DIR}/../../src_optimized)
file(GLOB_RECURSE SRC_OPT RELATIVE ${PROJECT_SOURCE_DIR} "${EZSIFT_SRC_OPT_DIR}/*.cpp" "${EZSIFT_SRC_OPT_DIR}/*.cu")
add_library(ezsift_opt ${SRC_OPT})
set_property(TARGET ezsift_opt PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(ezsift_opt PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(ezsift_opt PUBLIC ${EZSIFT_SRC_OPT_DIR} ${EZSIFT_INCLUDE_DIR})
target_link_libraries(ezsift_opt PUBLIC OpenMP::OpenMP_CXX)

if(EZSIFT_BUILD_EXAMPLES)
  # Use relative path since EZSIFT_ROOT is undefined
  add_subdirectory(../../examples/benchmark ${CMAKE_BINARY_DIR}/examples/benchmark)
endif()