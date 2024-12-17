//------------------------------------------------------------------------------
/// \file
/// \brief      implementations of some BatchArray methods
/// \date       2020-2024
/// \author     Jakub Kurzak
/// \copyright  Advanced Micro Devices, Inc.
///
#include <hip/hip_fp8.h>

#include "BatchArray.h"

//------------------------------------------------------------------------------
/// \brief
///     Populates the batch with a specific value.
///
template <typename T>
__global__
void generateConstKernel(
    int m, int n, T** d_array, int lda, double val)
{
    T value;
    if constexpr (std::is_same<T, fp8>::value) {
        value = fp8(internal::cast_to_f8<double, true>(val, /*wm*/3, /*we*/4));
    }
    else if constexpr (std::is_same<T, bf8>::value) {
        value = bf8(internal::cast_to_f8<double, true>(val, /*wm*/2, /*we*/5));
    }
    else {
        value = T(val);
    }

    T* A = d_array[blockIdx.y];
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    for (int j = 0; j < n; ++j) {
        if (i < m)
            A[std::size_t(lda)*j + i] = value;
    }
}

//------------------------------------------------------------------------------
/// \brief
///     Checks if the batch contains a specific value.
///
template <typename T>
__global__
void validateConstKernel(
    int m, int n, T** d_array, int lda, double val)
{
    T* A = d_array[blockIdx.y];
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    for (int j = 0; j < n; ++j) {
        if (i < m) {
            double value;
            if constexpr (std::is_same<T, fp8>::value) {
                value = internal::cast_from_f8<double, true>(
                    static_cast<__hip_fp8_storage_t>(A[j]), /*wm*/3, /*we*/4);
            }
            else if constexpr (std::is_same<T, bf8>::value) {
                value = internal::cast_from_f8<double, true>(
                    static_cast<__hip_fp8_storage_t>(A[j]), /*wm*/2, /*we*/5);
            }
            else if constexpr (std::is_same<T, std::complex<float>>::value ||
                               std::is_same<T, std::complex<double>>::value) {
                value = std::real(A[j]);
            }
            else {
                value = static_cast<double>(A[j]);
            }
            assert(value == val);
        }
    }
}

//------------------------------------------------------------------------------
/// \brief
///     Populates the batch with a specific value.
///
template <typename T>
void BatchArray<T>::generateConstant(int device_id, double val)
{
    HIP_CALL(hipSetDevice(device_id));
    int group_size = 256;
    int groups_per_matrix =
        m_%group_size == 0 ? m_/group_size : m_/group_size+1;
    dim3 grid_size(groups_per_matrix, batch_count_);
    generateConstKernel<<<grid_size, group_size>>>(
        m_, n_, d_array_, ld_, val);
}

template
void BatchArray<int8_t>::generateConstant(int device_id, double val);

template
void BatchArray<fp8>::generateConstant(int device_id, double val);

template
void BatchArray<bf8>::generateConstant(int device_id, double val);

template
void BatchArray<__half>::generateConstant(int device_id, double val);

template
void BatchArray<hip_bfloat16>::generateConstant(int device_id, double val);

template
void BatchArray<int>::generateConstant(int device_id, double val);

template
void BatchArray<float>::generateConstant(int device_id, double val);

template
void BatchArray<double>::generateConstant(int device_id, double val);

template
void BatchArray<std::complex<float>>::generateConstant(int device_id,
                                                       double val);
template
void BatchArray<std::complex<double>>::generateConstant(int device_id,
                                                        double val);

//------------------------------------------------------------------------------
/// \brief
///     Checks if the batch contains a specific value.
///
template <typename T>
void BatchArray<T>::validateConstant(int device_id, double val)
{
    HIP_CALL(hipSetDevice(device_id));
    int group_size = 256;
    int groups_per_matrix =
        m_%group_size == 0 ? m_/group_size : m_/group_size+1;
    dim3 grid_size(groups_per_matrix, batch_count_);
    validateConstKernel<<<grid_size, group_size>>>(
        m_, n_, d_array_, ld_, val);
}

template
void BatchArray<int8_t>::validateConstant(int device_id, double val);

template
void BatchArray<fp8>::validateConstant(int device_id, double val);

template
void BatchArray<bf8>::validateConstant(int device_id, double val);

template
void BatchArray<__half>::validateConstant(int device_id, double val);

template
void BatchArray<hip_bfloat16>::validateConstant(int device_id, double val);

template
void BatchArray<int>::validateConstant(int device_id, double val);

template
void BatchArray<float>::validateConstant(int device_id, double val);

template
void BatchArray<double>::validateConstant(int device_id, double val);

template
void BatchArray<std::complex<float>>::validateConstant(int device_id,
                                                       double val);
template
void BatchArray<std::complex<double>>::validateConstant(int device_id,
                                                        double val);
