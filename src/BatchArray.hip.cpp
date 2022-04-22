//------------------------------------------------------------------------------
/// \file
/// \brief      implementations of some BatchArray methods
/// \date       2020-2021
/// \author     Jakub Kurzak
/// \copyright  Advanced Micro Devices, Inc.
///
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
    T value = (T)val;
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
    T value = (T)val;
    T* A = d_array[blockIdx.y];
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    for (int j = 0; j < n; ++j) {
        if (i < m)
            assert(A[std::size_t(lda)*j + i] == value);
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
