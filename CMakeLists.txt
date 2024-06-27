# ==============================================================================
# \file
# \brief      CMake configuration for building the CoralGemm project
# \date       2020-2024
# \author     Jakub Kurzak
# \copyright  Advanced Micro Devices, Inc.
#
cmake_minimum_required(VERSION 3.21)

project(CoralGemm)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Set build type to Release by default" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
      "Debug" "Release" "MinSizeRel" "RelWithDebInfo")

endif()

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options to choose between HIP and CUDA
option(USE_HIP "Use HIP for AMD GPUs" ON)
option(USE_CUDA "Use CUDA for NVIDIA GPUs" OFF)

if(USE_HIP AND USE_CUDA)
  message(FATAL_ERROR "Cannot use both HIP and CUDA at the same time.")

# ------------------------------------------------------------------------------
# Setup for building with HIP (AMD GPUs)
#
elseif(USE_HIP)
  enable_language(HIP)
  find_package(rocblas REQUIRED)
  find_package(hipblas REQUIRED)
  find_package(rocrand REQUIRED)
  find_package(hiprand REQUIRED)

  file(GLOB SOURCES "src/*.cpp")
  set_source_files_properties(
    ${SOURCES} PROPERTIES LANGUAGE HIP)
add_executable(gemm ${SOURCES})

target_link_libraries(gemm PRIVATE
    roc::rocblas
    roc::rocrand
    roc::hipblas
    hip::hiprand)

target_compile_definitions(gemm PRIVATE USE_HIP)

# ------------------------------------------------------------------------------
# Setup for building with CUDA (NVIDIA GPUs)
#
elseif(USE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  file(GLOB SOURCES "src/*.cpp")
  set_source_files_properties(
      ${SOURCES} PROPERTIES LANGUAGE CUDA)
  add_executable(gemm ${SOURCES})

  target_link_libraries(gemm PRIVATE
      CUDA::cublas
      CUDA::curand)

  target_compile_definitions(gemm PRIVATE USE_CUDA)
  target_compile_options(gemm PRIVATE --expt-relaxed-constexpr)

else()
  message(FATAL_ERROR "Either USE_HIP or USE_CUDA must be set.")

endif()
