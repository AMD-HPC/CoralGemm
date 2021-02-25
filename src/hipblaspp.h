 //------------------------------------------------------------------------------
/// \file
/// \brief      C++ wrappers for the hipBLAS routines used in CoralGemm
/// \date       2020-2021
/// \author     Jakub Kurzak
/// \copyright  Advanced Micro Devices, Inc.
///
#pragma once

#include "Exception.h"

#if defined(__HIPCC__) || defined(__HIP_PLATFORM__)
#include <hipblas.h>
#endif

#if defined(__NVCC__)
#include <cublas_v2.h>
#endif

/// C++ wrappers for hipBLAS routines
namespace hipblas {

//------------------------------------------------------------------------------
void gemm(hipblasHandle_t handle,
          hipblasOperation_t opA, hipblasOperation_t opB,
          int m, int n, int k,
          float* alpha, float* A, int lda,
                        float* B, int ldb,
          float* beta,  float* C, int ldc)
{
    HIPBLAS_CALL(hipblasSgemm(handle,
                              opA, opB,
                              m, n, k,
                              alpha, A, lda,
                                     B, ldb,
                              beta,  C, ldc));
}

void gemm(hipblasHandle_t handle,
          hipblasOperation_t opA, hipblasOperation_t opB,
          int m, int n, int k,
          double* alpha, double* A, int lda,
                         double* B, int ldb,
          double* beta,  double* C, int ldc)
{
    HIPBLAS_CALL(hipblasDgemm(handle,
                              opA, opB,
                              m, n, k,
                              alpha, A, lda,
                                     B, ldb,
                              beta,  C, ldc));
}

void gemm(hipblasHandle_t handle,
          hipblasOperation_t opA, hipblasOperation_t opB,
          int m, int n, int k,
          std::complex<float>* alpha, std::complex<float>* A, int lda,
                                      std::complex<float>* B, int ldb,
          std::complex<float>* beta,  std::complex<float>* C, int ldc)
{
    HIPBLAS_CALL(
        hipblasCgemm(
            handle,
            opA, opB,
            m, n, k,
            (hipblasComplex*)alpha, (hipblasComplex*)A, lda,
                                    (hipblasComplex*)B, ldb,
            (hipblasComplex*)beta,  (hipblasComplex*)C, ldc));
}

void gemm(hipblasHandle_t handle,
          hipblasOperation_t opA, hipblasOperation_t opB,
          int m, int n, int k,
          std::complex<double>* alpha, std::complex<double>* A, int lda,
                                       std::complex<double>* B, int ldb,
          std::complex<double>* beta,  std::complex<double>* C, int ldc)
{
    HIPBLAS_CALL(
        hipblasZgemm(
            handle,
            opA, opB,
            m, n, k,
            (hipblasDoubleComplex*)alpha, (hipblasDoubleComplex*)A, lda,
                                          (hipblasDoubleComplex*)B, ldb,
            (hipblasDoubleComplex*)beta,  (hipblasDoubleComplex*)C, ldc));
}

void gemm(hipblasHandle_t handle,
          hipblasOperation_t opA, hipblasOperation_t opB,
          int m, int n, int k,
          int32_t* alpha, int8_t* A, int lda,
                          int8_t* B, int ldb,
          int32_t* beta, int32_t* C, int ldc)
{
    HIPBLAS_CALL(hipblasGemmEx(handle,
                               opA, opB,
                               m, n, k,
                               alpha, A, HIPBLAS_R_8I,  lda,
                                      B, HIPBLAS_R_8I,  ldb,
                               beta,  C, HIPBLAS_R_32I, ldc,
                               HIPBLAS_R_32I,
                               HIPBLAS_GEMM_DEFAULT));
}

//------------------------------------------------------------------------------
void gemmBatched(hipblasHandle_t handle,
                 hipblasOperation_t opA, hipblasOperation_t opB,
                 int m, int n, int k,
                 float* alpha, float** A, int lda,
                               float** B, int ldb,
                 float* beta,  float** C, int ldc,
                 int batch_size)
{
    HIPBLAS_CALL(hipblasSgemmBatched(handle,
                                     opA, opB,
                                     m, n, k,
                                     alpha, A, lda,
                                            B, ldb,
                                     beta,  C, ldc,
                                     batch_size));
}

void gemmBatched(hipblasHandle_t handle,
                 hipblasOperation_t opA, hipblasOperation_t opB,
                 int m, int n, int k,
                 double* alpha, double** A, int lda,
                                double** B, int ldb,
                 double* beta,  double** C, int ldc,
                 int batch_size)
{
    HIPBLAS_CALL(hipblasDgemmBatched(handle,
                                     opA, opB,
                                     m, n, k,
                                     alpha, A, lda,
                                            B, ldb,
                                     beta,  C, ldc,
                                     batch_size));
}

void gemmBatched(
    hipblasHandle_t handle,
    hipblasOperation_t opA, hipblasOperation_t opB,
    int m, int n, int k,
    std::complex<float>* alpha, std::complex<float>** A, int lda,
                                std::complex<float>** B, int ldb,
    std::complex<float>* beta,  std::complex<float>** C, int ldc,
    int batch_size)
{
    HIPBLAS_CALL(
        hipblasCgemmBatched(
            handle,
            opA, opB,
            m, n, k,
            (hipblasComplex*)alpha, (hipblasComplex**)A, lda,
                                    (hipblasComplex**)B, ldb,
            (hipblasComplex*)beta,  (hipblasComplex**)C, ldc,
            batch_size));
}

void gemmBatched(
    hipblasHandle_t handle,
    hipblasOperation_t opA, hipblasOperation_t opB,
    int m, int n, int k,
    std::complex<double>* alpha, std::complex<double>** A, int lda,
                                 std::complex<double>** B, int ldb,
    std::complex<double>* beta,  std::complex<double>** C, int ldc,
    int batch_size)
{
    HIPBLAS_CALL(
        hipblasZgemmBatched(
            handle,
            opA, opB,
            m, n, k,
            (hipblasDoubleComplex*)alpha, (hipblasDoubleComplex**)A, lda,
                                          (hipblasDoubleComplex**)B, ldb,
            (hipblasDoubleComplex*)beta,  (hipblasDoubleComplex**)C, ldc,
            batch_size));
}

void gemmBatched(hipblasHandle_t handle,
                 hipblasOperation_t opA, hipblasOperation_t opB,
                 int m, int n, int k,
                 int32_t* alpha, int8_t** A, int lda,
                                 int8_t** B, int ldb,
                 int32_t* beta, int32_t** C, int ldc,
                 int batch_size)
{
    HIPBLAS_CALL(
        hipblasGemmBatchedEx(handle,
                             opA, opB,
                             m, n, k,
                             alpha, (const void**)A, HIPBLAS_R_8I,  lda,
                                    (const void**)B, HIPBLAS_R_8I,  ldb,
                             beta,  (      void**)C, HIPBLAS_R_32I, ldc,
                             batch_size,
                             HIPBLAS_R_32I,
                             HIPBLAS_GEMM_DEFAULT));
}

//------------------------------------------------------------------------------
void gemmStridedBatched(
    hipblasHandle_t handle,
    hipblasOperation_t opA, hipblasOperation_t opB,
    int m, int n, int k,
    float* alpha, float* A, int lda, std::size_t strideA,
                  float* B, int ldb, std::size_t strideB,
    float* beta,  float* C, int ldc, std::size_t strideC,
    int batch_size)
{
    HIPBLAS_CALL(hipblasSgemmStridedBatched(handle,
                                            opA, opB,
                                            m, n, k,
                                            alpha, A, lda, strideA,
                                                   B, ldb, strideB,
                                            beta,  C, ldc, strideC,
                                            batch_size));
}

void gemmStridedBatched(
    hipblasHandle_t handle,
    hipblasOperation_t opA, hipblasOperation_t opB,
    int m, int n, int k,
    double* alpha, double* A, int lda, std::size_t strideA,
                   double* B, int ldb, std::size_t strideB,
    double* beta,  double* C, int ldc, std::size_t strideC,
    int batch_size)
{
    HIPBLAS_CALL(hipblasDgemmStridedBatched(handle,
                                            opA, opB,
                                            m, n, k,
                                            alpha, A, lda, strideA,
                                                   B, ldb, strideB,
                                            beta,  C, ldc, strideC,
                                            batch_size));
}

void gemmStridedBatched(
    hipblasHandle_t handle,
    hipblasOperation_t opA, hipblasOperation_t opB,
    int m, int n, int k,
    std::complex<float>* alpha,
    std::complex<float>* A, int lda, std::size_t strideA,
    std::complex<float>* B, int ldb, std::size_t strideB,
    std::complex<float>* beta,
    std::complex<float>* C, int ldc, std::size_t strideC,
    int batch_size)
{
    HIPBLAS_CALL(
        hipblasCgemmStridedBatched(
            handle,
            opA, opB,
            m, n, k,
            (hipblasComplex*)alpha, (hipblasComplex*)A, lda, strideA,
                                    (hipblasComplex*)B, ldb, strideB,
            (hipblasComplex*)beta,  (hipblasComplex*)C, ldc, strideC,
            batch_size));
}

void gemmStridedBatched(
    hipblasHandle_t handle,
    hipblasOperation_t opA, hipblasOperation_t opB,
    int m, int n, int k,
    std::complex<double>* alpha,
    std::complex<double>* A, int lda, std::size_t strideA,
    std::complex<double>* B, int ldb, std::size_t strideB,
    std::complex<double>* beta,
    std::complex<double>* C, int ldc, std::size_t strideC,
    int batch_size)
{
    HIPBLAS_CALL(
        hipblasZgemmStridedBatched(
            handle,
            opA, opB,
            m, n, k,
            (hipblasDoubleComplex*)alpha,
            (hipblasDoubleComplex*)A, lda, strideA,
            (hipblasDoubleComplex*)B, ldb, strideB,
            (hipblasDoubleComplex*)beta,
            (hipblasDoubleComplex*)C, ldc, strideC,
            batch_size));
}

void gemmStridedBatched(
    hipblasHandle_t handle,
    hipblasOperation_t opA, hipblasOperation_t opB,
    int m, int n, int k,
    int32_t* alpha, int8_t* A, int lda, std::size_t strideA,
                    int8_t* B, int ldb, std::size_t strideB,
    int32_t* beta, int32_t* C, int ldc, std::size_t strideC,
    int batch_size)
{
    HIPBLAS_CALL(
        hipblasGemmStridedBatchedEx(handle,
                                    opA, opB,
                                    m, n, k,
                                    alpha, A, HIPBLAS_R_8I,  lda, strideA,
                                           B, HIPBLAS_R_8I,  ldb, strideB,
                                    beta,  C, HIPBLAS_R_32I, ldc, strideC,
                                    batch_size,
                                    HIPBLAS_R_32I,
                                    HIPBLAS_GEMM_DEFAULT));
}

} // namespace hipblas
