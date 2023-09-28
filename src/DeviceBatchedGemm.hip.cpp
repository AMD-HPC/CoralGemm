//------------------------------------------------------------------------------
/// \file
/// \brief      implementations of DeviceBatchedGemm methods
/// \date       2020-2023
/// \author     Jakub Kurzak
/// \copyright  Advanced Micro Devices, Inc.
///
#include "DeviceBatchedGemm.h"

#include <algorithm>

//------------------------------------------------------------------------------
/// \brief
///     Creates a DeviceBatchedGemm object.
///
/// \param[in] compute_type
///     the computing precision (data type),
///     e.g., HIPBLAS_R_32F, HIPBLAS_R_64F, HIPBLAS_C_32F, HIPBLAS_C_64F
///
/// \param[in] op_a, op_b
///     the transposition opetations for A and B,
///     i.e., HIPBLAS_OP_N, HIPBLAS_OP_T, or HIPBLAS_OP_C
///
/// \param[in] a, b, c
///     the input arrays A and B, and the output array C
///
/// \param[in] batch_count
///     the number of matrices in the batch
///
/// \param[in] alpha, beta
///     the alpha and beta factors in matrix multiplication
///
/// \param[in] operations
///     number of floating point operations
///     for one member of the batch (single matrix multiply)
///
/// \param[in] device_id
///     the number of the device executing the operation
///
DeviceBatchedGemm::DeviceBatchedGemm(hipblasDatatype_t compute_type,
                                     hipblasOperation_t op_a,
                                     hipblasOperation_t op_b,
                                     BaseBatchArray* a,
                                     BaseBatchArray* b,
                                     BaseBatchArray* c,
                                     int batch_count,
                                     void const* alpha,
                                     void const* beta,
                                     double operations,
                                     int device_id)
    : BatchedGemm(compute_type,
                  op_a, op_b,
                  a, b, c,
                  batch_count,
                  alpha, beta,
                  operations),
      device_id_(device_id)
{
    // Set device, create stream.
    HIP_CALL(hipSetDevice(device_id_));
    HIP_CALL(hipStreamCreate(&hip_stream_));

    // Create hipBLAS handle, assign stream.
    HIPBLAS_CALL(hipblasCreate(&hipblas_handle_));
    HIPBLAS_CALL(hipblasSetStream(hipblas_handle_, hip_stream_));

    // Create hipRAND generator, assign stream.
    HIPRAND_CALL(hiprandCreateGenerator(&hiprand_generator_,
                                        HIPRAND_RNG_PSEUDO_DEFAULT));
    HIPRAND_CALL(hiprandSetStream(hiprand_generator_, hip_stream_));

    // Create evvents.
    start.resize(batch_count);
    stop.resize(batch_count);
    for (int i = 0; i < batch_count; ++i) {
        HIP_CALL(hipEventCreate(&start[i]));
        HIP_CALL(hipEventCreate(&stop[i]));
    }
}

//------------------------------------------------------------------------------
DeviceBatchedGemm::~DeviceBatchedGemm()
{
    // Set the device.
    hipSetDevice(device_id_);

    // Destroy all the handles.
    hiprandDestroyGenerator(hiprand_generator_);
    hipblasDestroy(hipblas_handle_);
    hipStreamDestroy(hip_stream_);

    // Destroy events.
    for (int i = 0; i < batch_count_; ++i) {
        hipEventDestroy(start[i]);
        hipEventDestroy(stop[i]);
    }
}

//------------------------------------------------------------------------------
/// \brief
///     Runs the workload.
///
/// \param[in] mode
///     the mode of operation:
///     - Standard:        regular GEMM, one precision
///     - Batched:         batched GEMM, one precision
///     - StridedBatched:  strided GEMM, one precision
///     - StandardEx:      regular GEMM, multi-precision
///     - BatchedEx:       batched GEMM, multi-precision
///     - StridedBatchedEx strided GEMM, multi-precision
///
void DeviceBatchedGemm::run(Mode mode)
{
    // Set the device.
    hipSetDevice(device_id_);

    switch(mode) {
        case Mode::Standard:
            runGemm();
            break;
        case Mode::Batched:
            runBatchedGemm();
            break;
        case Mode::StridedBatched:
            runStridedBatchedGemm();
            break;
        case Mode::StandardEx:
            runGemmEx();
            break;
        case Mode::BatchedEx:
            runBatchedGemmEx();
            break;
        case Mode::StridedBatchedEx:
            runStridedBatchedGemmEx();
            break;
        default:
            ERROR("Unsupported mode");
    }
}

//------------------------------------------------------------------------------
/// \brief
///     Runs the workload.
///     Invokes the standard API in a loop.
///     Collects the times for all calls.
///
void DeviceBatchedGemm::runGemm()
{
    // Check if all types are the same.
    ASSERT(a_->type() == b_->type(), "Missing \"ex\" in the command line?");
    ASSERT(b_->type() == c_->type(), "Missing \"ex\" in the command line?");

    // Call in a loop, record events.
    // Use the standard API (not Ex).
    for (int i = 0; i < batch_count_; ++i) {
        HIP_CALL(hipEventRecord(start[i]));
        hipblas::gemm(c_->type(),
                      hipblas_handle_,
                      op_a_,
                      op_b_,
                      c_->m(),
                      c_->n(),
                      op_a_ == HIPBLAS_OP_N ? a_->n() : a_->m(),
                      alpha_, a_->h_array(i), a_->ld(),
                              b_->h_array(i), b_->ld(),
                      beta_,  c_->h_array(i), c_->ld());
        HIP_CALL(hipEventRecord(stop[i]));
    }
}

//------------------------------------------------------------------------------
/// \brief
///     Runs the workload.
///     Invokes the batched API once and collects the time.
///
void DeviceBatchedGemm::runBatchedGemm()
{
    // Check if all types are the same.
    ASSERT(a_->type() == b_->type(), "Missing \"ex\" in the command line?");
    ASSERT(b_->type() == c_->type(), "Missing \"ex\" in the command line?");

    // Call once, record start and stop.
    // Use the standard API (not Ex).
    HIP_CALL(hipEventRecord(start[0]));
    hipblas::gemmBatched(c_->type(),
                         hipblas_handle_,
                         op_a_,
                         op_b_,
                         c_->m(),
                         c_->n(),
                         op_a_ == HIPBLAS_OP_N ? a_->n() : a_->m(),
                         alpha_, a_->d_array(), a_->ld(),
                                 b_->d_array(), b_->ld(),
                         beta_,  c_->d_array(), c_->ld(),
                         batch_count_);
    HIP_CALL(hipEventRecord(stop[0]));
}

//------------------------------------------------------------------------------
/// \brief
///     Runs the workload.
///     Invokes the strided batched API once and collects the time.
///
void DeviceBatchedGemm::runStridedBatchedGemm()
{
    // Check if all types are the same.
    ASSERT(a_->type() == b_->type(), "Missing \"ex\" in the command line?");
    ASSERT(b_->type() == c_->type(), "Missing \"ex\" in the command line?");

    // Compute strides (= sizes of matrices).
    size_t stride_a = size_t(a_->ld())*(HIPBLAS_OP_N ? a_->n() : a_->m());
    size_t stride_b = size_t(b_->ld())*(HIPBLAS_OP_N ? b_->n() : b_->m());
    size_t stride_c = size_t(c_->ld())*c_->n();

    // Call once, record start and stop.
    // Use the standard API (not Ex).
    HIP_CALL(hipEventRecord(start[0]));
    hipblas::gemmStridedBatched(c_->type(),
                                hipblas_handle_,
                                op_a_,
                                op_b_,
                                c_->m(),
                                c_->n(),
                                op_a_ == HIPBLAS_OP_N ? a_->n() : a_->m(),
                                alpha_, a_->data(), a_->ld(), stride_a,
                                        b_->data(), b_->ld(), stride_b,
                                beta_,  c_->data(), c_->ld(), stride_c,
                                batch_count_);
    HIP_CALL(hipEventRecord(stop[0]));
}

//------------------------------------------------------------------------------
/// \brief
///     Runs the workload.
///     Invokes the Ex API in a loop.
///     Collects the times of all calls.
///
void DeviceBatchedGemm::runGemmEx()
{
    // Call in a loop, record events.
    for (int i = 0; i < batch_count_; ++i) {
        HIP_CALL(hipEventRecord(start[i]));
        HIPBLAS_CALL(
            hipblasGemmEx(hipblas_handle_,
                          op_a_,
                          op_b_,
                          c_->m(),
                          c_->n(),
                          op_a_ == HIPBLAS_OP_N ? a_->n() : a_->m(),
                          alpha_, a_->h_array(i), a_->type(), a_->ld(),
                                  b_->h_array(i), b_->type(), b_->ld(),
                          beta_,  c_->h_array(i), c_->type(), c_->ld(),
                          compute_type_,
                          HIPBLAS_GEMM_DEFAULT));
        HIP_CALL(hipEventRecord(stop[i]));
    }
}

//------------------------------------------------------------------------------
/// \brief
///     Runs the workload.
///     Invokes the batched Ex API once and collects the time.
///
void DeviceBatchedGemm::runBatchedGemmEx()
{
    // Call once, record start and stop.
    HIP_CALL(hipEventRecord(start[0]));
    HIPBLAS_CALL(
        hipblasGemmBatchedEx(hipblas_handle_,
                             op_a_,
                             op_b_,
                             c_->m(),
                             c_->n(),
                             op_a_ == HIPBLAS_OP_N ? a_->n() : a_->m(),
                             alpha_,
                             (void const**)a_->d_array(), a_->type(), a_->ld(),
                             (void const**)b_->d_array(), b_->type(), b_->ld(),
                             beta_,
                             c_->d_array(), c_->type(), c_->ld(),
                             batch_count_,
                             compute_type_,
                             HIPBLAS_GEMM_DEFAULT));
    HIP_CALL(hipEventRecord(stop[0]));
}

//------------------------------------------------------------------------------
/// \brief
///     Runs the workload.
///     Invokes the strided batched Ex API once and collects the time.
///
void DeviceBatchedGemm::runStridedBatchedGemmEx()
{
    // Compute strides (= sizes of matrices).
    size_t stride_a = size_t(a_->ld())*(HIPBLAS_OP_N ? a_->n() : a_->m());
    size_t stride_b = size_t(b_->ld())*(HIPBLAS_OP_N ? b_->n() : b_->m());
    size_t stride_c = size_t(c_->ld())*c_->n();

    // Call once, record start and stop.
    HIP_CALL(hipEventRecord(start[0]));
    hipblasGemmStridedBatchedEx(hipblas_handle_,
                                op_a_,
                                op_b_,
                                c_->m(),
                                c_->n(),
                                op_a_ == HIPBLAS_OP_N ? a_->n() : a_->m(),
                                alpha_, a_->data(), a_->type(), a_->ld(), stride_a,
                                        b_->data(), b_->type(), b_->ld(), stride_b,
                                beta_,  c_->data(), c_->type(), c_->ld(), stride_c,
                                batch_count_,
                                compute_type_,
                                HIPBLAS_GEMM_DEFAULT);
    HIP_CALL(hipEventRecord(stop[0]));
}

//------------------------------------------------------------------------------
/// \brief
///     Returns the GLOPS number for the run.
///
/// \param[in] mode
///     the mode of operation:
///     - Standard:        regular GEMM, one precision
///     - Batched:         batched GEMM, one precision
///     - StridedBatched:  strided GEMM, one precision
///     - StandardEx:      regular GEMM, multi-precision
///     - BatchedEx:       batched GEMM, multi-precision
///     - StridedBatchedEx strided GEMM, multi-precision
///
std::pair<double, double> DeviceBatchedGemm::getGflops(Mode mode)
{
    // Set the device.
    hipSetDevice(device_id_);

    double gflops;
    double time_in_sec;
    if (mode == Mode::Standard || mode == Mode::StandardEx) {
        // Report GFLOPS based on the median time.
        std::vector<float> elapsed(batch_count_);
        HIP_CALL(hipEventSynchronize(stop[batch_count_-1]));
        for (int i = 0; i < batch_count_; ++i)
            hipEventElapsedTime(&elapsed[i], start[i], stop[i]);

        std::sort(elapsed.begin(), elapsed.end(), std::greater<float>());
        double median_time = elapsed[batch_count_/2];
        time_in_sec = median_time/1e3;
        gflops = operations_/time_in_sec/1e9;
    }
    else {
        // Report GFLOPS based on the single run.
        /// \todo Possibly introduce iterations.
        float elapsed;
        HIP_CALL(hipEventSynchronize(stop[0]));
        hipEventElapsedTime(&elapsed, start[0], stop[0]);
        time_in_sec = elapsed/1e3;
        gflops = operations_*batch_count_/time_in_sec/1e9;
    }
    return {gflops, time_in_sec};
}
