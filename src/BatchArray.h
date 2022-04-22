//------------------------------------------------------------------------------
/// \file
/// \brief      BatchArray class declaration and inline routines
/// \date       2020-2021
/// \author     Jakub Kurzak
/// \copyright  Advanced Micro Devices, Inc.
///
#pragma once

#include "BaseBatchArray.h"
#include "Exception.h"
#include "hiprandpp.h"

//------------------------------------------------------------------------------
template <typename T>
__global__
void generateConstKernel(int m, int n, T** d_array, int lda, double val);

template <typename T>
__global__
void validateConstKernel(int m, int n, T** d_array, int lda, double val);

//------------------------------------------------------------------------------
/// \brief
///     Precision-templated class for represents an array of matrices
///     in either device memory or page-locked host memory.
///     Inherits from the BaseBatchArray class.
///     Serves as the parent class for HostBatchArray and DeviceBatchArray.
///
template <typename T>
class BatchArray: public BaseBatchArray {
public:
    /// \brief
    ///     Creates a BatchArray object.
    ///
    /// \param[in] type
    ///     the data type, e.g., HIPBLAS_R_64F
    ///
    /// \param[in] m, n, ld
    ///     the width, height, and leading dimension of the matrices
    ///
    /// \param[in] batch_count
    ///     the number of matrices in the batch
    ///
    BatchArray(hipblasDatatype_t type,
               int m, int n, int ld,
               int batch_count)
        : BaseBatchArray(type, m, n, ld, batch_count) {}
    ~BatchArray() {}

    /// Returns the pointer to the memory occupied by the batch.
    virtual void* data() const override { return data_; }


    /// Returns the pointer of a specific matrix in the batch.
    virtual void* h_array(int i) const override { return h_array_[i]; }

    /// Returns the array of pointers in device memory.
    virtual void** d_array() const override { return (void**)d_array_; }

    /// Populates the batch with random data.
    void generateUniform(int device_id, hiprandGenerator_t generator) override
    {
        HIP_CALL(hipSetDevice(device_id));
        hiprand::generateUniform(generator, data_, ld_*n_*batch_count_);
    }

    /// Populates the batch with a specific value.
    void generateConstant(int device_id, double val) override;

    /// Checks if the batch contains a specific value.
    void validateConstant(int device_id, double val) override;

protected:
    T* data_;     ///< the pointer to the memory occupied by the batch
    T** h_array_; ///< the array of pointers in host memory
    T** d_array_; ///< the array of pointers in device memory
};
