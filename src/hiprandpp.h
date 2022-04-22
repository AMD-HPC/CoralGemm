//------------------------------------------------------------------------------
/// \file
/// \brief      C++ wrappers for the hipRAND routines used in CoralGemm
/// \date       2020-2021
/// \author     Jakub Kurzak
/// \copyright  Advanced Micro Devices, Inc.
///
#pragma once

#include "Exception.h"

#include <complex>
#include <limits>

#if defined(__HIPCC__)
#include <hiprand.h>
#include <hip/hip_bfloat16.h>
#elif defined(__NVCC__)
#include <curand.h>
#endif

/// C++ wrappers for hipRAND
namespace hiprand {

//------------------------------------------------------------------------------
template <typename T>
__global__
void int2float_kernel(std::size_t len, unsigned int* src, T* dst)
{
    std::size_t pos = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (pos < len) {
        float uint32_val = float(src[pos]);
        float uint32_max = float(std::numeric_limits<unsigned int>::max());
        float scaled_val = uint32_val/uint32_max;
        dst[pos] = T(scaled_val);
    }
}

//------------------------------------------------------------------------------
template <typename T>
inline
void int2float(std::size_t len, unsigned int* src, T* dst)
{
    constexpr int block_size = 256;
    int num_blocks = len%block_size == 0 ? len/block_size : len/block_size + 1;
    int2float_kernel<T><<<dim3(num_blocks), dim3(block_size)>>>(len, src, dst);
}

//------------------------------------------------------------------------------
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
#if defined(__HIPCC__)
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
#if defined(__HIPCC__)
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
