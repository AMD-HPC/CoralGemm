//------------------------------------------------------------------------------
/// \file
/// \brief      HIP to CUDA name replacements
/// \date       2020-2021
/// \author     Jakub Kurzak
/// \copyright  Advanced Micro Devices, Inc.
///
#if defined(__NVCC__) && !defined(__HIP_PLATFORM__)
    #define HIPBLAS_GEMM_DEFAULT         CUBLAS_GEMM_DEFAULT
    #define HIPBLAS_OP_N                 CUBLAS_OP_N
    #define HIPBLAS_OP_T                 CUBLAS_OP_T
    #define HIPBLAS_OP_C                 CUBLAS_OP_C
    #define HIPBLAS_R_32I                CUDA_R_32I
    #define HIPBLAS_R_8I                 CUDA_R_8I
    #define HIPBLAS_STATUS_SUCCESS       CUBLAS_STATUS_SUCCESS
    #define hipblasHandle_t              cublasHandle_t
    #define hipblasOperation_t           cublasOperation_t
    #define hipblasComplex               cuComplex
    #define hipblasDoubleComplex         cuDoubleComplex
    #define hipblasCreate                cublasCreate
    #define hipblasDestroy               cublasDestroy
    #define hipblasSetStream             cublasSetStream
    #define hipblasGemmEx                cublasGemmEx
    #define hipblasGemmBatchedEx         cublasGemmBatchedEx
    #define hipblasGemmStridedBatchedEx  cublasGemmStridedBatchedEx
    #define hipblasSgemm                 cublasSgemm
    #define hipblasDgemm                 cublasDgemm
    #define hipblasCgemm                 cublasCgemm
    #define hipblasZgemm                 cublasZgemm
    #define hipblasSgemmBatched          cublasSgemmBatched
    #define hipblasDgemmBatched          cublasDgemmBatched
    #define hipblasCgemmBatched          cublasCgemmBatched
    #define hipblasZgemmBatched          cublasZgemmBatched
    #define hipblasSgemmStridedBatched   cublasSgemmStridedBatched
    #define hipblasDgemmStridedBatched   cublasDgemmStridedBatched
    #define hipblasCgemmStridedBatched   cublasCgemmStridedBatched
    #define hipblasZgemmStridedBatched   cublasZgemmStridedBatched
#endif
