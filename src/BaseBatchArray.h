//------------------------------------------------------------------------------
/// \file
/// \brief      BaseBatchArray class declaration and inline routines
/// \date       2020-2021
/// \author     Jakub Kurzak
/// \copyright  Advanced Micro Devices, Inc.
///
#pragma once

#if defined(__HIP_PLATFORM_AMD__)
#include <hipblas.h>
#include <hiprand.h>
#include <hip/hip_bfloat16.h>
#elif defined(__HIP_PLATFORM_NVIDIA__)
#include <cublas_v2.h>
#include <curand.h>
#include "hipblas2cublas.h"
#include "hiprand2curand.h"
#endif

//------------------------------------------------------------------------------
/// \brief
///     Base class for representing arrays in host memory or device memory.
///     Serves as the parent class for the templated class BatchArray.
///
class BaseBatchArray {
public:
    static BaseBatchArray* make(hipblasDatatype_t type,
                                int m, int n, int ld,
                                int batch_count,
                                int device_id,
                                bool coherent = true);

    /// \brief
    ///     Creates a BaseBatchArray object.
    ///
    /// \param[in] type
    ///     the data type, e.g., HIPBLAS_R_64F
    ///
    /// \param[in] m
    ///     the height of the matrices in the batch
    ///
    /// \param[in] n
    ///     the width of the matrices in the batch
    ///
    /// \param[in] ld
    ///     the leading dimension of each matrix in the batch
    ///
    /// \param[in] batch_count
    ///     the number of matrices in the batch
    ///
    BaseBatchArray(hipblasDatatype_t type,
                   int m, int n, int ld,
                   int batch_count)
        : type_(type),
          m_(m), n_(n), ld_(ld),
          batch_count_(batch_count) {}
    virtual ~BaseBatchArray() {}

    int m() const { return m_; }
    int n() const { return n_; }
    int ld() const { return ld_; }
    hipblasDatatype_t type() const { return type_; }

    /// Returns the pointer to the memory occupied by the batch.
    virtual void* data() const = 0;

    /// Returns the pointer of a specific matrix in the batch.
    virtual void* h_array(int i) const = 0;

    /// Returns the array of pointers in device memory.
    virtual void** d_array() const = 0;

    /// Populates the batch with random data.
    virtual void generateUniform(int device_id,
                                 hiprandGenerator_t generator) = 0;

    /// Populates the batch with a specific value.
    virtual void generateConstant(int device_id, double val) = 0;

    /// Checks if the batch contains a specific value.
    virtual void validateConstant(int device_id, double val) = 0;

protected:
    hipblasDatatype_t type_; ///< the data type, e.g., HIPBLAS_R_64F
    int m_;                  ///< the height of each matrices in the batch
    int n_;                  ///< the width of each matrix in the batch
    int ld_;                 ///< the leading dimension of each matrix
    int batch_count_;        ///< the number of matrices in the batch
};
