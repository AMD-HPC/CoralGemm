//------------------------------------------------------------------------------
/// \file
/// \brief      HIP to CUDA name replacements
/// \date       2020-2024
/// \author     Jakub Kurzak
/// \copyright  Advanced Micro Devices, Inc.
///
#pragma once

#if defined(USE_HIP)
    #include <hipblas/hipblas.h>
    #include <hipblaslt/hipblaslt.h>
    #include <hip/hip_fp8.h>
#elif defined(USE_CUDA)
#endif

struct TypeConstant {
    hipDataType          hip_;
    hipblasDatatype_t    hipblas_;
    hipblasComputeType_t compute_;

    hipDataType          hip() { return hip_; }
    hipblasDatatype_t    hipblas() { return hipblas_; }
    hipblasComputeType_t compute() { return compute_; }
};

constexpr hipDataType HIP_INVALID = static_cast<hipDataType>(1023);
constexpr hipblasDatatype_t HIPBLAS_INVALID = static_cast<hipblasDatatype_t>(255);
constexpr hipblasComputeType_t COMPUTE_INVALID = static_cast<hipblasComputeType_t>(15);

const TypeConstant R_64F = { HIP_R_64F,     HIPBLAS_R_64F,   HIPBLAS_COMPUTE_64F };
const TypeConstant C_64F = { HIP_C_64F,     HIPBLAS_C_64F,   HIPBLAS_COMPUTE_64F };
const TypeConstant R_32F = { HIP_R_32F,     HIPBLAS_R_32F,   HIPBLAS_COMPUTE_32F };
const TypeConstant C_32F = { HIP_C_32F,     HIPBLAS_C_32F,   HIPBLAS_COMPUTE_32F };
const TypeConstant R_16F = { HIP_R_16F,     HIPBLAS_R_16F,   HIPBLAS_COMPUTE_16F };
const TypeConstant R_16B = { HIP_R_16BF,    HIPBLAS_R_16B,   COMPUTE_INVALID     };
const TypeConstant R_8F  = { HIP_R_8F_E4M3, HIPBLAS_INVALID, COMPUTE_INVALID     };
const TypeConstant R_8B  = { HIP_R_8F_E5M2, HIPBLAS_INVALID, COMPUTE_INVALID     };
const TypeConstant R_32I = { HIP_R_32I,     HIPBLAS_R_32I,   HIPBLAS_COMPUTE_32I };
const TypeConstant R_8I  = { HIP_R_8I,      HIPBLAS_R_8I,    COMPUTE_INVALID     };

enum struct fp8 : __hip_fp8_storage_t {};
enum struct bf8 : __hip_fp8_storage_t {};
