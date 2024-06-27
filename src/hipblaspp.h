 //------------------------------------------------------------------------------
/// \file
/// \brief      C++ wrappers for the hipBLAS routines used in CoralGemm
/// \date       2020-2024
/// \author     Jakub Kurzak
/// \copyright  Advanced Micro Devices, Inc.
///
#pragma once

#include "Exception.h"

#include <complex>

#if defined(USE_HIP)
#include <hipblas/hipblas.h>
#elif defined(USE_CUDA)
#include <cublas_v2.h>
#endif

/// C++ wrappers for hipBLAS
namespace hipblas {

//------------------------------------------------------------------------------
/// standard GEMM (float)
inline
void gemm(hipblasHandle_t handle,
          hipblasOperation_t op_a,
          hipblasOperation_t op_b,
          int m, int n, int k,
          float const* alpha,
          float const* A, int lda,
          float const* B, int ldb,
          float const* beta,
          float* C, int ldc)
{
    HIPBLAS_CALL(hipblasSgemm(handle,
                              op_a, op_b,
                              m, n, k,
                              alpha, A, lda,
                                     B, ldb,
                              beta,  C, ldc));
}

/// standard GEMM (double)
inline
void gemm(hipblasHandle_t handle,
          hipblasOperation_t op_a,
          hipblasOperation_t op_b,
          int m, int n, int k,
          double const* alpha,
          double const* A, int lda,
          double const* B, int ldb,
          double const* beta,
          double* C, int ldc)
{
    HIPBLAS_CALL(hipblasDgemm(handle,
                              op_a, op_b,
                              m, n, k,
                              alpha, A, lda,
                                     B, ldb,
                              beta,  C, ldc));
}

/// standard GEMM (complex<float>)
inline
void gemm(hipblasHandle_t handle,
          hipblasOperation_t op_a,
          hipblasOperation_t op_b,
          int m, int n, int k,
          std::complex<float> const* alpha,
          std::complex<float> const* A, int lda,
          std::complex<float> const* B, int ldb,
          std::complex<float> const* beta,
          std::complex<float>* C, int ldc)
{
    HIPBLAS_CALL(hipblasCgemm(handle,
                              op_a, op_b,
                              m, n, k,
                              (hipblasComplex const*)alpha,
                              (hipblasComplex const*)A, lda,
                              (hipblasComplex const*)B, ldb,
                              (hipblasComplex const*)beta,
                              (hipblasComplex*)C, ldc));
}

/// standard GEMM (complex<double>)
inline
void gemm(hipblasHandle_t handle,
          hipblasOperation_t op_a,
          hipblasOperation_t op_b,
          int m, int n, int k,
          std::complex<double> const* alpha,
          std::complex<double> const* A, int lda,
          std::complex<double> const* B, int ldb,
          std::complex<double> const* beta,
          std::complex<double>* C, int ldc)
{
    HIPBLAS_CALL(hipblasZgemm(handle,
                              op_a, op_b,
                              m, n, k,
                              (hipblasDoubleComplex const*)alpha,
                              (hipblasDoubleComplex const*)A, lda,
                              (hipblasDoubleComplex const*)B, ldb,
                              (hipblasDoubleComplex const*)beta,
                              (hipblasDoubleComplex*)C, ldc));
}

/// standard GEMM with dispatch based on `hipblasDatatype_t`
inline
void gemm(hipblasDatatype_t type,
          hipblasHandle_t handle,
          hipblasOperation_t op_a,
          hipblasOperation_t op_b,
          int m, int n, int k,
          void const* alpha,
          void const* A, int lda,
          void const* B, int ldb,
          void const* beta,
          void* C, int ldc)
{
    switch (type) {
        case HIPBLAS_R_32F:
            gemm(handle,
                 op_a, op_b,
                 m, n, k,
                 (float const*)alpha,
                 (float const*)A, lda,
                 (float const*)B, ldb,
                 (float const*)beta,
                 (float*)C, ldc);
            break;
        case HIPBLAS_R_64F:
            gemm(handle,
                 op_a, op_b,
                 m, n, k,
                 (double const*)alpha,
                 (double const*)A, lda,
                 (double const*)B, ldb,
                 (double const*)beta,
                 (double*)C, ldc);
            break;
        case HIPBLAS_C_32F:
            gemm(handle,
                 op_a, op_b,
                 m, n, k,
                 (std::complex<float> const*)alpha,
                 (std::complex<float> const*)A, lda,
                 (std::complex<float> const*)B, ldb,
                 (std::complex<float> const*)beta,
                 (std::complex<float>*)C, ldc);
            break;
        case HIPBLAS_C_64F:
            gemm(handle,
                 op_a, op_b,
                 m, n, k,
                 (std::complex<double> const*)alpha,
                 (std::complex<double> const*)A, lda,
                 (std::complex<double> const*)B, ldb,
                 (std::complex<double> const*)beta,
                 (std::complex<double>*)C, ldc);
            break;
        default:
            ERROR("Unsupported data type.");
    }
}

//------------------------------------------------------------------------------
/// batched GEMM (float)
inline
void gemmBatched(hipblasHandle_t handle,
                 hipblasOperation_t op_a,
                 hipblasOperation_t op_b,
                 int m, int n, int k,
                 float const* alpha,
                 float* const* A, int lda,
                 float* const* B, int ldb,
                 float const* beta,
                 float** C, int ldc,
                 int batch_count)
{
    HIPBLAS_CALL(hipblasSgemmBatched(handle,
                                     op_a, op_b,
                                     m, n, k,
                                     alpha, A, lda,
                                            B, ldb,
                                     beta,  C, ldc,
                                     batch_count));
}

/// batched GEMM (double)
inline
void gemmBatched(hipblasHandle_t handle,
                 hipblasOperation_t op_a,
                 hipblasOperation_t op_b,
                 int m, int n, int k,
                 double const* alpha,
                 double* const* A, int lda,
                 double* const* B, int ldb,
                 double const* beta,
                 double** C, int ldc,
                 int batch_count)
{
    HIPBLAS_CALL(hipblasDgemmBatched(handle,
                                     op_a, op_b,
                                     m, n, k,
                                     alpha, A, lda,
                                            B, ldb,
                                     beta,  C, ldc,
                                     batch_count));
}

/// batched GEMM (complex<float>)
inline
void gemmBatched(hipblasHandle_t handle,
                 hipblasOperation_t op_a,
                 hipblasOperation_t op_b,
                 int m, int n, int k,
                 std::complex<float> const* alpha,
                 std::complex<float>* const* A, int lda,
                 std::complex<float>* const* B, int ldb,
                 std::complex<float> const* beta,
                 std::complex<float>** C, int ldc,
                 int batch_count)
{
    HIPBLAS_CALL(hipblasCgemmBatched(handle,
                                     op_a, op_b,
                                     m, n, k,
                                     (hipblasComplex const*)alpha,
                                     (hipblasComplex* const*)A, lda,
                                     (hipblasComplex* const*)B, ldb,
                                     (hipblasComplex const*)beta,
                                     (hipblasComplex**)C, ldc,
                                     batch_count));
}

/// batched GEMM (complex<double>)
inline
void gemmBatched(hipblasHandle_t handle,
                 hipblasOperation_t op_a,
                 hipblasOperation_t op_b,
                 int m, int n, int k,
                 std::complex<double> const* alpha,
                 std::complex<double>* const* A, int lda,
                 std::complex<double>* const* B, int ldb,
                 std::complex<double> const* beta,
                 std::complex<double>** C, int ldc,
                 int batch_count)
{
    HIPBLAS_CALL(hipblasZgemmBatched(handle,
                                     op_a, op_b,
                                     m, n, k,
                                     (hipblasDoubleComplex const*)alpha,
                                     (hipblasDoubleComplex* const*)A, lda,
                                     (hipblasDoubleComplex* const*)B, ldb,
                                     (hipblasDoubleComplex const*)beta,
                                     (hipblasDoubleComplex**)C, ldc,
                                     batch_count));
}

/// batched GEMM with dispatch based on `hipblasDatatype_t`
inline
void gemmBatched(hipblasDatatype_t type,
                 hipblasHandle_t handle,
                 hipblasOperation_t op_a,
                 hipblasOperation_t op_b,
                 int m, int n, int k,
                 void const* alpha,
                 void* const* A, int lda,
                 void* const* B, int ldb,
                 void const* beta,
                 void** C, int ldc,
                 int batch_count)
{
    switch (type) {
        case HIPBLAS_R_32F:
            gemmBatched(handle,
                        op_a, op_b,
                        m, n, k,
                        (float const*)alpha,
                        (float* const*)A, lda,
                        (float* const*)B, ldb,
                        (float const*)beta,
                        (float**)C, ldc,
                        batch_count);
            break;
        case HIPBLAS_R_64F:
            gemmBatched(handle,
                        op_a, op_b,
                        m, n, k,
                        (double const*)alpha,
                        (double* const*)A, lda,
                        (double* const*)B, ldb,
                        (double const*)beta,
                        (double**)C, ldc,
                        batch_count);
            break;
        case HIPBLAS_C_32F:
            gemmBatched(handle,
                        op_a, op_b,
                        m, n, k,
                        (std::complex<float> const*)alpha,
                        (std::complex<float>* const*)A, lda,
                        (std::complex<float>* const*)B, ldb,
                        (std::complex<float> const*)beta,
                        (std::complex<float>**)C, ldc,
                        batch_count);
            break;
        case HIPBLAS_C_64F:
            gemmBatched(handle,
                        op_a, op_b,
                        m, n, k,
                        (std::complex<double> const*)alpha,
                        (std::complex<double>* const*)A, lda,
                        (std::complex<double>* const*)B, ldb,
                        (std::complex<double> const*)beta,
                        (std::complex<double>**)C, ldc,
                        batch_count);
            break;
        default:
            ERROR("Unsupported data type.");
    }
}

//------------------------------------------------------------------------------
/// strided batched GEMM (float)
inline
void gemmStridedBatched(
    hipblasHandle_t handle,
    hipblasOperation_t op_a,
    hipblasOperation_t op_b,
    int m, int n, int k,
    float const* alpha,
    float const* A, int lda, std::size_t strideA,
    float const* B, int ldb, std::size_t strideB,
    float const* beta,
    float* C, int ldc, std::size_t strideC,
    int batch_count)
{
    HIPBLAS_CALL(
        hipblasSgemmStridedBatched(handle,
                                   op_a, op_b,
                                   m, n, k,
                                   alpha, A, lda, strideA,
                                          B, ldb, strideB,
                                   beta,  C, ldc, strideC,
                                   batch_count));
}

/// strided batched GEMM (double)
inline
void gemmStridedBatched(
    hipblasHandle_t handle,
    hipblasOperation_t op_a,
    hipblasOperation_t op_b,
    int m, int n, int k,
    double const* alpha,
    double const* A, int lda, std::size_t strideA,
    double const* B, int ldb, std::size_t strideB,
    double const* beta,
    double* C, int ldc, std::size_t strideC,
    int batch_count)
{
    HIPBLAS_CALL(
        hipblasDgemmStridedBatched(handle,
                                   op_a, op_b,
                                   m, n, k,
                                   alpha, A, lda, strideA,
                                          B, ldb, strideB,
                                   beta,  C, ldc, strideC,
                                   batch_count));
}

/// strided batched GEMM (complex<float>)
inline
void gemmStridedBatched(
    hipblasHandle_t handle,
    hipblasOperation_t op_a,
    hipblasOperation_t op_b,
    int m, int n, int k,
    std::complex<float> const* alpha,
    std::complex<float> const* A, int lda, std::size_t strideA,
    std::complex<float> const* B, int ldb, std::size_t strideB,
    std::complex<float> const* beta,
    std::complex<float>* C, int ldc, std::size_t strideC,
    int batch_count)
{
    HIPBLAS_CALL(
        hipblasCgemmStridedBatched(
            handle,
            op_a, op_b,
            m, n, k,
            (hipblasComplex const*)alpha,
            (hipblasComplex const*)A, lda, strideA,
            (hipblasComplex const*)B, ldb, strideB,
            (hipblasComplex const*)beta,
            (hipblasComplex*)C, ldc, strideC,
            batch_count));
}

/// strided batched GEMM (complex<double>)
inline
void gemmStridedBatched(
    hipblasHandle_t handle,
    hipblasOperation_t op_a,
    hipblasOperation_t op_b,
    int m, int n, int k,
    std::complex<double> const* alpha,
    std::complex<double> const* A, int lda, std::size_t strideA,
    std::complex<double> const* B, int ldb, std::size_t strideB,
    std::complex<double> const* beta,
    std::complex<double>* C, int ldc, std::size_t strideC,
    int batch_count)
{
    HIPBLAS_CALL(
        hipblasZgemmStridedBatched(
            handle,
            op_a, op_b,
            m, n, k,
            (hipblasDoubleComplex const*)alpha,
            (hipblasDoubleComplex const*)A, lda, strideA,
            (hipblasDoubleComplex const*)B, ldb, strideB,
            (hipblasDoubleComplex const*)beta,
            (hipblasDoubleComplex*)C, ldc, strideC,
            batch_count));
}

/// strided batched GEMM with dispatch based on `hipblasDatatype_t`
inline
void gemmStridedBatched(
    hipblasDatatype_t type,
    hipblasHandle_t handle,
    hipblasOperation_t op_a,
    hipblasOperation_t op_b,
    int m, int n, int k,
    void const* alpha,
    void const* A, int lda, std::size_t strideA,
    void const* B, int ldb, std::size_t strideB,
    void const* beta,
    void* C, int ldc, std::size_t strideC,
    int batch_count)
{
    switch (type) {
        case HIPBLAS_R_32F:
            gemmStridedBatched(handle,
                               op_a, op_b,
                               m, n, k,
                               (float const*)alpha,
                               (float const*)A, lda, strideA,
                               (float const*)B, ldb, strideB,
                               (float const*)beta,
                               (float*)C, ldc, strideC,
                               batch_count);
            break;
        case HIPBLAS_R_64F:
            gemmStridedBatched(handle,
                               op_a, op_b,
                               m, n, k,
                               (double const*)alpha,
                               (double const*)A, lda, strideA,
                               (double const*)B, ldb, strideB,
                               (double const*)beta,
                               (double*)C, ldc, strideC,
                               batch_count);
            break;
        case HIPBLAS_C_32F:
            gemmStridedBatched(handle,
                               op_a, op_b,
                               m, n, k,
                               (std::complex<float> const*)alpha,
                               (std::complex<float> const*)A, lda, strideA,
                               (std::complex<float> const*)B, ldb, strideB,
                               (std::complex<float> const*)beta,
                               (std::complex<float>*)C, ldc, strideC,
                               batch_count);
            break;
        case HIPBLAS_C_64F:
            gemmStridedBatched(handle,
                               op_a, op_b,
                               m, n, k,
                               (std::complex<double> const*)alpha,
                               (std::complex<double> const*)A, lda, strideA,
                               (std::complex<double> const*)B, ldb, strideB,
                               (std::complex<double> const*)beta,
                               (std::complex<double>*)C, ldc, strideC,
                               batch_count);
            break;
        default:
            ERROR("Unsupported data type.");
    }
}

} // namespace hipblas
