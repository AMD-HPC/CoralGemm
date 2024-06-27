//------------------------------------------------------------------------------
/// \file
/// \brief      auxiliary functions for C++ hipRAND wrappers in CoralGemm
/// \date       2020-2024
/// \author     Jakub Kurzak
/// \copyright  Advanced Micro Devices, Inc.
///
#include "hiprandpp.h"

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
void int2float(std::size_t len, unsigned int* src, T* dst)
{
    constexpr int block_size = 256;
    int num_blocks = len%block_size == 0 ? len/block_size : len/block_size + 1;
    int2float_kernel<T><<<dim3(num_blocks), dim3(block_size)>>>(len, src, dst);
}

template
void int2float(std::size_t len, unsigned int* src, __half* dst);

template
void int2float(std::size_t len, unsigned int* src, hip_bfloat16* dst);

} // namespace hiprand
