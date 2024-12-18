//------------------------------------------------------------------------------
/// \file
/// \brief      C++ wrappers for the hipRAND routines used in CoralGemm
/// \date       2020-2024
/// \author     Jakub Kurzak
/// \copyright  Advanced Micro Devices, Inc.
///
#pragma once

#include "Exception.h"
#include "FloatTypes.h"

#include <complex>
#include <limits>

#if defined(USE_HIP)
    #include <hiprand/hiprand.h>
    #include <hip/hip_runtime.h>
    #include <hip/hip_bfloat16.h>
#elif defined(USE_CUDA)
    #include <curand.h>
#endif

/// C++ wrappers for hipRAND
namespace hiprand {

//------------------------------------------------------------------------------
template <typename T>
__global__
void int2float_kernel(std::size_t len, unsigned int* src, T* dst);

template <typename T>
void int2float(std::size_t len, unsigned int* src, T* dst);

//------------------------------------------------------------------------------
/// Generate uniform (FP8).
inline
void generateUniform(
    hiprandGenerator_t generator, fp8* A, std::size_t len)
{
    unsigned int* uint32_A;
    HIP_CALL(hipMalloc(&uint32_A, sizeof(unsigned int)*len));
    HIPRAND_CALL(hiprandGenerate(generator, uint32_A, len));
    int2float(len, uint32_A, A);
    HIP_CALL(hipFree(uint32_A));
}

/// Generate uniform (BF8).
inline
void generateUniform(
    hiprandGenerator_t generator, bf8* A, std::size_t len)
{
    unsigned int* uint32_A;
    HIP_CALL(hipMalloc(&uint32_A, sizeof(unsigned int)*len));
    HIPRAND_CALL(hiprandGenerate(generator, uint32_A, len));
    int2float(len, uint32_A, A);
    HIP_CALL(hipFree(uint32_A));
}

/// Generate uniform (float).
inline
void generateUniform(
    hiprandGenerator_t generator, float* A, std::size_t len)
{
    HIPRAND_CALL(hiprandGenerateUniform(generator, A, len));
}

/// Generate uniform (double).
inline
void generateUniform(
    hiprandGenerator_t generator, double* A, std::size_t len)
{
    HIPRAND_CALL(hiprandGenerateUniformDouble(generator, A, len));
}

/// Generate uniform (complex<float>).
inline
void generateUniform(
    hiprandGenerator_t generator, std::complex<float>* A, std::size_t len)
{
    HIPRAND_CALL(hiprandGenerateUniform(generator, (float*)A, len*2));
}

/// Generate uniform (complex<double>).
inline
void generateUniform(
    hiprandGenerator_t generator, std::complex<double>* A, std::size_t len)
{
    HIPRAND_CALL(hiprandGenerateUniformDouble(generator, (double*)A, len*2));
}

/// Generate uniform (int8_t).
/// \remark Silly implementation generating len/4 of int32_t.
/// \todo Replace with hiprandGenerateChar() when available.
inline
void generateUniform(
    hiprandGenerator_t generator, int8_t* A, std::size_t len)
{
#if defined(USE_HIP)
    HIPRAND_CALL(hiprandGenerateChar(generator, (unsigned char*)A, len));
#else
    ASSERT(len%4 == 0);
    HIPRAND_CALL(hiprandGenerate(generator, (unsigned int*)A, len/4));
#endif
}

/// Generate uniform (int32_t).
inline
void generateUniform(
    hiprandGenerator_t generator, int32_t* A, std::size_t len)
{
    HIPRAND_CALL(hiprandGenerate(generator, (unsigned int*)A, len));
}

/// Generate uniform (__half).
inline
void generateUniform(
    hiprandGenerator_t generator, __half* A, std::size_t len)
{
#if defined(USE_HIP)
    HIPRAND_CALL(hiprandGenerateUniformHalf(generator, A, len));
#else
    unsigned int* uint32_A;
    HIP_CALL(hipMalloc(&uint32_A, sizeof(unsigned int)*len));
    HIPRAND_CALL(hiprandGenerate(generator, uint32_A, len));
    int2float(len, uint32_A, A);
    HIP_CALL(hipFree(uint32_A));
#endif
}

/// Generate uniform (hip_bfloat16).
inline
void generateUniform(
    hiprandGenerator_t generator, hip_bfloat16* A, std::size_t len)
{
    unsigned int* uint32_A;
    HIP_CALL(hipMalloc(&uint32_A, sizeof(unsigned int)*len));
    HIPRAND_CALL(hiprandGenerate(generator, uint32_A, len));
    int2float(len, uint32_A, A);
    HIP_CALL(hipFree(uint32_A));
}

} // namespace hiprand
