cmake_minimum_required(VERSION 3.21)

project(CoralGemm)

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
