//------------------------------------------------------------------------------
/// \file
/// \brief      HostBatchArray class declaration and inline routines
/// \date       2020-2023
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
class HostBatchArray: public BatchArray<T> {
public:
    HostBatchArray(hipblasDatatype_t type,
                   int m, int n, int ld,
                   int batch_count,
                   bool coherent = true);
    ~HostBatchArray();
};

//------------------------------------------------------------------------------
/// \brief
///     Creates a HostBatchArray object.
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
/// \param[in] coherent
///     If true, host memory allocated as hipHostMallocCoherent (not cached).
///     If false, host memory allocated as hipHostMallocNonCoherent (cached).
///
template <typename T>
inline
HostBatchArray<T>::HostBatchArray(hipblasDatatype_t type,
                                  int m, int n, int ld,
                                  int batch_count,
                                  bool coherent)
    : BatchArray<T>(type, m, n, ld, batch_count)
{
#if defined(__HIP_PLATFORM_AMD__)
    HIP_CALL(hipHostMalloc(&this->data_, sizeof(T)*ld*n*batch_count,
                           coherent ? hipHostMallocCoherent
                                    : hipHostMallocNonCoherent));
#else
    HIP_CALL(hipHostMalloc(&this->data_, sizeof(T)*ld*n*batch_count));
#endif
    HIP_CALL(hipHostMalloc(&this->h_array_, sizeof(T*)*batch_count));
    for (int i = 0; i < batch_count; ++i)
        this->h_array_[i] = this->data_+(std::size_t)ld*n*i;

    HIP_CALL(hipMalloc(&this->d_array_, sizeof(T*)*batch_count));
    HIP_CALL(hipMemcpy(this->d_array_, this->h_array_, sizeof(T*)*batch_count,
                       hipMemcpyHostToDevice));

}

//------------------------------------------------------------------------------
template <typename T>
inline
HostBatchArray<T>::~HostBatchArray()
{
    (void)hipFree(this->d_array_);
    (void)hipFree(this->h_array_);
    (void)hipFree(this->data_);
}
