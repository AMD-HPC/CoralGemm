//------------------------------------------------------------------------------
/// \file
/// \brief      DeviceBatchArray class declaration and inline routines
/// \date       2020-2024
/// \author     Jakub Kurzak
/// \copyright  Advanced Micro Devices, Inc.
///
#pragma once

#include "BatchArray.h"
#include "Exception.h"

//------------------------------------------------------------------------------
/// \brief
///     Represents an array of matrices in device memory
///     Inherits from the BatchArray class.
///
template <typename T>
class DeviceBatchArray: public BatchArray<T> {
public:
    DeviceBatchArray(TypeConstant type,
                     int m, int n, int ld,
                     int batch_count,
                     int device_id = 0);
    ~DeviceBatchArray();

private:
    int device_id_; ///< number of the device storing the array
};

//------------------------------------------------------------------------------
/// \brief
///     Creates a DeviceBatchArray object.
///
/// \param[in] type
///     the data type, e.g., HIPBLAS_R_64F
///
/// \param[in] m, n, ld
///     the width, height, and leading dimension of the matrices in the batch.
///
/// \param[in] batch_count
///     the number of matrices in the batch
///
/// \param[in] device_id
///     the number of the device storing the array.
///
template <typename T>
inline
DeviceBatchArray<T>::DeviceBatchArray(TypeConstant type,
                                      int m, int n, int ld,
                                      int batch_count,
                                      int device_id)
    : BatchArray<T>(type, m, n, ld, batch_count),
      device_id_(device_id)
{
    HIP_CALL(hipSetDevice(device_id_));
    HIP_CALL(hipMalloc(&this->data_, sizeof(T)*ld*n*batch_count));

    HIP_CALL(hipHostMalloc(&this->h_array_, sizeof(T*)*batch_count));
    for (int i = 0; i < batch_count; ++i)
        this->h_array_[i] = this->data_+(std::size_t)ld*n*i;

    HIP_CALL(hipMalloc(&this->d_array_, sizeof(T*)*batch_count));
    HIP_CALL(hipMemcpy(this->d_array_, this->h_array_, sizeof(T*)*batch_count,
                       hipMemcpyHostToDevice));

    HIPBLASLT_CALL(hipblasLtMatrixLayoutCreate(&this->layout_,
                                               type.hip(),
                                               m, n, ld));
}

//------------------------------------------------------------------------------
template <typename T>
inline
DeviceBatchArray<T>::~DeviceBatchArray()
{
    (void)hipSetDevice(device_id_);
    (void)hipblasLtMatrixLayoutDestroy(this->layout_);
    (void)hipFree(this->d_array_);
    (void)hipHostFree(this->h_array_);
    (void)hipFree(this->data_);
}
