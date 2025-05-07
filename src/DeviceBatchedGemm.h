//------------------------------------------------------------------------------
/// \file
/// \brief      DeviceBatchedGemm class declaration and inline routines
/// \date       2020-2024
/// \author     Jakub Kurzak
/// \copyright  Advanced Micro Devices, Inc.
///
#pragma once

#include "Exception.h"
#include "BatchedGemm.h"

#include <vector>

#if defined(USE_HIP)
    #include <hip/hip_runtime.h>
    #include <hipblas/hipblas.h>
    #include <hipblaslt/hipblaslt.h>
    #include <hipblaslt/hipblaslt-ext.hpp>
    #include <hiprand/hiprand.h>
    #include "hipblaspp.h"
#elif defined(USE_CUDA)
    #include <cuda_runtime.h>
    #include <cublas_v2.h>
    #include <curand.h>
    #include "hip2cuda.h"
    #include "hipblas2cublas.h"
    #include "hiprand2curand.h"
    #include "hipblaspp.h"
#endif

//------------------------------------------------------------------------------
/// \brief
///     Implements benchmarking of matrix multiplication on GPUs.
///     Inherits from the BatchedGemm class.
///
/// \todo
///     Explore direct access to pointer arrays in host memory
///     (check the effects of using hipHostMallocNonCoherent).
///
class DeviceBatchedGemm: public BatchedGemm {
public:
    DeviceBatchedGemm(TypeConstant compute_type,
                      hipblasOperation_t op_a,
                      hipblasOperation_t op_b,
                      BaseBatchArray* a,
                      BaseBatchArray* b,
                      BaseBatchArray* c,
                      int batch_count,
                      void const* alpha,
                      void const* beta,
                      double operations,
                      int device_id,
                      bool lt);
    ~DeviceBatchedGemm();

    /// Populates the batch with random data.
    void generateUniform() override
    {
        a_->generateUniform(device_id_, hiprand_generator_);
        b_->generateUniform(device_id_, hiprand_generator_);
        c_->generateUniform(device_id_, hiprand_generator_);
    }

    /// Populates the batch with a specific value.
    void generateConstant(double val) override
    {
        a_->generateConstant(device_id_, val);
        b_->generateConstant(device_id_, val);
        c_->generateConstant(device_id_, val);
    }

    void validateConstant(double val) override
    {
        c_->validateConstant(device_id_, val);
    }

    void run(Mode mode) override;
    std::pair<double, double> getGflops(Mode mode) override;

private:
    void runGemm();
    void runBatchedGemm();
    void runStridedBatchedGemm();
    void runGemmEx();
    void runBatchedGemmEx();
    void runStridedBatchedGemmEx();
    void runGemmLt();
    void runBatchedGemmLt();

    int device_id_; ///< the number of the device executing the operation
    bool lt_;       ///< true if using hipBLASLt

    hipStream_t hip_stream_;                      ///< stream
    hipblasHandle_t hipblas_handle_;              ///< hipBLAS handle
    hipblasLtHandle_t hipblaslt_handle_;          ///< hipBLASLt handle
    void* hipblaslt_workspace_;                   ///< hipBLASLt workspace
    std::size_t hipblaslt_workspace_size_;        ///< hipBLASLt workspace size
    hipblasLtMatmulDesc_t hipblaslt_matmul_desc_; ///< hipBLASLt mm descriptor
    hiprandGenerator_t hiprand_generator_;        ///< random number generator

    std::vector<hipEvent_t> start; ///< events for recording start times
    std::vector<hipEvent_t> stop;  ///< events for recording end times
};
