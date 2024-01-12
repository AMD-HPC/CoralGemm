cmake_minimum_required(VERSION 3.21)

project(CoralGemm)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Set build type to Release by default" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

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
    roc::hipblas roc::rocblas
    roc::rocrand hip::hiprand)
