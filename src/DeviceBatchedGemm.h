//------------------------------------------------------------------------------
/// \file
/// \brief      DeviceBatchedGemm class declaration and inline routines
/// \date       2020-2021
/// \author     Jakub Kurzak
/// \copyright  Advanced Micro Devices, Inc.
///
#pragma once

#include "Exception.h"
#include "BatchedGemm.h"

#include <vector>

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <hipblas.h>
#include <hiprand.h>
#include "hipblaspp.h"
#elif defined(__NVCC__)
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include "hip2cuda.h"
#include "hipblas2cublas.h"
#include "hiprand2curand.h"
#include "hipblaspp.h"
#endif

//------------------------------------------------------------------------------
class DeviceBatchedGemm: public BatchedGemm {
public:
    DeviceBatchedGemm(hipblasDatatype_t compute_type,
                      hipblasOperation_t op_a,
                      hipblasOperation_t op_b,
                      BaseBatchArray* a,
                      BaseBatchArray* b,
                      BaseBatchArray* c,
                      int batch_count,
                      void const* alpha,
                      void const* beta,
                      double operations,
                      int device_id = 0);
    ~DeviceBatchedGemm();

    void generateUniform() override
    {
        a_->generateUniform(device_id_, hiprand_generator_);
        b_->generateUniform(device_id_, hiprand_generator_);
        c_->generateUniform(device_id_, hiprand_generator_);
    }

    void generateConstant(double val) override
    {
        a_->generateConstant(device_id_, val);
        b_->generateConstant(device_id_, val);
        c_->generateConstant(device_id_, val);
    }

    void run(Mode mode) override;
    double getGflops(Mode mode) override;

private:
    void runGemm();
    void runBatchedGemm();
    void runStridedBatchedGemm();
    void runGemmEx();
    void runBatchedGemmEx();
    void runStridedBatchedGemmEx();

    int device_id_; ///< number of the device executing the operation

    hipStream_t hip_stream_;
    hipblasHandle_t hipblas_handle_;
    hiprandGenerator_t hiprand_generator_;

    std::vector<hipEvent_t> start;
    std::vector<hipEvent_t> stop;
};
