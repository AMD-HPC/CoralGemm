//------------------------------------------------------------------------------
/// \file
/// \brief      BaseBatchArray class declaration and inline routines
/// \date       2020-2021
/// \author     Jakub Kurzak
/// \copyright  Advanced Micro Devices, Inc.
///
#pragma once

#if defined(__HIPCC__)
#include <hipblas.h>
#include <hiprand.h>
#elif defined(__NVCC__)
#include <cublas_v2.h>
#include <curand.h>
#include "hiprand2curand.h"
#endif

//------------------------------------------------------------------------------
class BaseBatchArray {
public:
    static BaseBatchArray* make(hipblasDatatype_t type,
                                int m, int n, int ld,
                                int batch_count,
                                int device_id,
                                bool coherent = true);

    BaseBatchArray(hipblasDatatype_t type,
                   int m, int n, int ld,
                   int batch_count)
        : type_(type),
          m_(m), n_(n), ld_(ld),
          batch_count_(batch_count) {}
    ~BaseBatchArray() {}

    int m() const { return m_; }
    int n() const { return n_; }
    int ld() const { return ld_; }
    hipblasDatatype_t type() const { return type_; }

    virtual void* data() const = 0;
    virtual void* h_array(int i) const = 0;
    virtual void** d_array() const = 0;

    virtual void generateUniform(int device_id,
                                 hiprandGenerator_t generator) = 0;
    virtual void generateConstant(int device_id, double val) = 0;

protected:
    hipblasDatatype_t type_; ///< data type, e.g., HIPBLAS_R_64F
    int m_;                  ///< height of each matrix in the batch
    int n_;                  ///< width of each matrix in the batch
    int ld_;                 ///< leading dimension of each matrix in the batch
    int batch_count_;        ///< number of matrices in the batch
};
