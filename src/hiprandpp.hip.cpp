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
void int2float_kernel(std::size_t len, unsigned int* src, T* dst, bool is_fnuz)
{
    std::size_t pos = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (pos < len) {
        float uint32_val = float(src[pos]);
        float uint32_max = float(std::numeric_limits<unsigned int>::max());
        float scaled_val = uint32_val/uint32_max;

        if constexpr (std::is_same<T, fp8>::value) {
            if (is_fnuz)
                dst[pos] = fp8(
                    internal::cast_to_f8<double, true >(
                        scaled_val, /*wm*/3, /*we*/4));
            else
                dst[pos] = fp8(
                    internal::cast_to_f8<double, false>(
                        scaled_val, /*wm*/3, /*we*/4));
        }
        else if constexpr (std::is_same<T, bf8>::value) {
            if (is_fnuz)
                dst[pos] = bf8(
                    internal::cast_to_f8<double, true >(
                        scaled_val, /*wm*/2, /*we*/5));
            else
                dst[pos] = bf8(
                    internal::cast_to_f8<double, false>(
                        scaled_val, /*wm*/2, /*we*/5));
        }
        else {
            dst[pos] = T(scaled_val);
        }
    }
}

//------------------------------------------------------------------------------
template <typename T>
void int2float(std::size_t len, unsigned int* src, T* dst, bool is_fnuz)
{
    constexpr int block_size = 256;
    int num_blocks = len%block_size == 0 ? len/block_size : len/block_size + 1;
    int2float_kernel<T><<<dim3(num_blocks), dim3(block_size)>>>(len, src, dst,
                                                                is_fnuz);
}

template
void int2float(std::size_t len, unsigned int* src, fp8* dst, bool is_fnuz);

template
void int2float(std::size_t len, unsigned int* src, bf8* dst, bool is_fnuz);

template
void int2float(std::size_t len, unsigned int* src, __half* dst, bool is_fnuz);

template
void int2float(std::size_t len, unsigned int* src, hip_bfloat16* dst,
               bool is_fnuz);

} // namespace hiprand
