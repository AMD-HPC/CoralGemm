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

const TypeConstant R_64F = { HIP_R_64F,  HIPBLAS_R_64F, HIPBLAS_COMPUTE_64F };
const TypeConstant C_64F = { HIP_C_64F,  HIPBLAS_C_64F, HIPBLAS_COMPUTE_64F };
const TypeConstant R_32F = { HIP_R_32F,  HIPBLAS_R_32F, HIPBLAS_COMPUTE_32F };
const TypeConstant C_32F = { HIP_C_32F,  HIPBLAS_C_32F, HIPBLAS_COMPUTE_32F };
const TypeConstant R_16F = { HIP_R_16F,  HIPBLAS_R_16F, HIPBLAS_COMPUTE_16F };
const TypeConstant R_16B = { HIP_R_16BF, HIPBLAS_R_16B, COMPUTE_INVALID     };
const TypeConstant R_32I = { HIP_R_32I,  HIPBLAS_R_32I, HIPBLAS_COMPUTE_32I };
const TypeConstant R_8I  = { HIP_R_8I,   HIPBLAS_R_8I,  COMPUTE_INVALID     };

    // HIP_R_32F   =  0,
    // HIP_R_64F   =  1,
    // HIP_R_16F   =  2,
    // HIP_R_8I    =  3,
    // HIP_C_32F   =  4,
    // HIP_C_64F   =  5,
    // HIP_C_16F   =  6,
    // HIP_C_8I    =  7,
    // HIP_R_8U    =  8,
    // HIP_C_8U    =  9,
    // HIP_R_32I   = 10,
    // HIP_C_32I   = 11,
    // HIP_R_32U   = 12,
    // HIP_C_32U   = 13,
    // HIP_R_16BF  = 14,
    // HIP_C_16BF  = 15,
    // HIP_R_4I    = 16,
    // HIP_C_4I    = 17,
    // HIP_R_4U    = 18,
    // HIP_C_4U    = 19,
    // HIP_R_16I   = 20,
    // HIP_C_16I   = 21,
    // HIP_R_16U   = 22,
    // HIP_C_16U   = 23,
    // HIP_R_64I   = 24,
    // HIP_C_64I   = 25,
    // HIP_R_64U   = 26,
    // HIP_C_64U   = 27,
    // HIP_R_8F_E4M3 = 28,
    // HIP_R_8F_E5M2 = 29,
    // // HIP specific Data Types
    // HIP_R_8F_E4M3_FNUZ = 1000,
    // HIP_R_8F_E5M2_FNUZ = 1001

    // // Note that these types are taken from cuBLAS. With the rocBLAS backend, currently hipBLAS will
    // // convert to rocBLAS types to get equivalent functionality where supported.
    // HIPBLAS_COMPUTE_16F           = 0, /**< compute will be at least 16-bit precision */
    // HIPBLAS_COMPUTE_16F_PEDANTIC  = 1, /**< compute will be exactly 16-bit precision */
    // HIPBLAS_COMPUTE_32F           = 2, /**< compute will be at least 32-bit precision */
    // HIPBLAS_COMPUTE_32F_PEDANTIC  = 3, /**< compute will be exactly 32-bit precision */
    // HIPBLAS_COMPUTE_32F_FAST_16F  = 4, /**< 32-bit input can use 16-bit compute */
    // HIPBLAS_COMPUTE_32F_FAST_16BF = 5, /**< 32-bit input can is bf16 compute */
    // HIPBLAS_COMPUTE_32F_FAST_TF32
    // = 6, /**< 32-bit input can use tensor cores w/ TF32 compute. Only supported with cuBLAS backend currently */
    // HIPBLAS_COMPUTE_64F          = 7, /**< compute will be at least 64-bit precision */
    // HIPBLAS_COMPUTE_64F_PEDANTIC = 8, /**< compute will be exactly 64-bit precision */
    // HIPBLAS_COMPUTE_32I          = 9, /**< compute will be at least 32-bit integer precision */
    // HIPBLAS_COMPUTE_32I_PEDANTIC = 10, /**< compute will be exactly 32-bit integer precision */

    // | HIP_R_16F  | HIP_R_16F  | HIP_R_16F  | HIPBLAS_COMPUTE_16F |
    // | HIP_R_16F  | HIP_R_16F  | HIP_R_16F  | HIPBLAS_COMPUTE_32F |
    // | HIP_R_16F  | HIP_R_16F  | HIP_R_32F  | HIPBLAS_COMPUTE_32F |
    // | HIP_R_16BF | HIP_R_16BF | HIP_R_16BF | HIPBLAS_COMPUTE_32F |
    // | HIP_R_16BF | HIP_R_16BF | HIP_R_32F  | HIPBLAS_COMPUTE_32F |
    // | HIP_R_32F  | HIP_R_32F  | HIP_R_32F  | HIPBLAS_COMPUTE_32F |
    // | HIP_R_64F  | HIP_R_64F  | HIP_R_64F  | HIPBLAS_COMPUTE_64F |
    // | HIP_R_8I   | HIP_R_8I   | HIP_R_32I  | HIPBLAS_COMPUTE_32I |
    // | HIP_C_32F  | HIP_C_32F  | HIP_C_32F  | HIPBLAS_COMPUTE_32F |
    // | HIP_C_64F  | HIP_C_64F  | HIP_C_64F  | HIPBLAS_COMPUTE_64F |

    // #define HIPBLAS_R_16F HIP_R_16F
    // #define HIPBLAS_R_32F HIP_R_32F
    // #define HIPBLAS_R_64F HIP_R_64F
    // #define HIPBLAS_C_16F HIP_C_16F
    // #define HIPBLAS_C_32F HIP_C_32F
    // #define HIPBLAS_C_64F HIP_C_64F
    // #define HIPBLAS_R_8I HIP_R_8I
    // #define HIPBLAS_R_8U HIP_R_8U
    // #define HIPBLAS_R_32I HIP_R_32I
    // #define HIPBLAS_R_32U HIP_R_32U
    // #define HIPBLAS_C_8I HIP_C_8I
    // #define HIPBLAS_C_8U HIP_C_8U
    // #define HIPBLAS_C_32I HIP_C_32I
    // #define HIPBLAS_C_32U HIP_C_32U
    // #define HIPBLAS_R_16B HIP_R_16BF
    // #define HIPBLAS_C_16B HIP_C_16BF
    // #define HIPBLAS_DATATYPE_INVALID hipDataType(31) // Temporary until hipblasDatatype_t is gone.
