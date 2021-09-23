//------------------------------------------------------------------------------
/// \file
/// \brief      hipBLAS to cuBLAS name replacements
/// \date       2020-2021
/// \author     Jakub Kurzak
/// \copyright  Advanced Micro Devices, Inc.
///
#define hipblasComplex                  cuComplex
#define hipblasCreate                   cublasCreate
#define hipblasDatatype_t               cudaDataType_t
#define hipblasDestroy                  cublasDestroy
#define hipblasDoubleComplex            cuDoubleComplex
#define hipblasHandle_t                 cublasHandle_t
#define hipblasOperation_t              cublasOperation_t
#define hipblasSetStream                cublasSetStream
#define hipblasStatus_t                 cublasStatus_t
#define hipblasGemmEx                   cublasGemmEx
#define hipblasGemmBatchedEx            cublasGemmBatchedEx
#define hipblasGemmStridedBatchedEx     cublasGemmStridedBatchedEx

#define hipblasSgemm                    cublasSgemm
#define hipblasDgemm                    cublasDgemm
#define hipblasCgemm                    cublasCgemm
#define hipblasZgemm                    cublasZgemm
#define hipblasSgemmBatched             cublasSgemmBatched
#define hipblasDgemmBatched             cublasDgemmBatched
#define hipblasCgemmBatched             cublasCgemmBatched
#define hipblasZgemmBatched             cublasZgemmBatched
#define hipblasSgemmStridedBatched      cublasSgemmStridedBatched
#define hipblasDgemmStridedBatched      cublasDgemmStridedBatched
#define hipblasCgemmStridedBatched      cublasCgemmStridedBatched
#define hipblasZgemmStridedBatched      cublasZgemmStridedBatched

#define HIPBLAS_GEMM_DEFAULT            CUBLAS_GEMM_DEFAULT
#define HIPBLAS_OP_N                    CUBLAS_OP_N
#define HIPBLAS_OP_T                    CUBLAS_OP_T
#define HIPBLAS_OP_C                    CUBLAS_OP_C
#define HIPBLAS_R_16F                   CUDA_R_16F
#define HIPBLAS_C_16F                   CUDA_C_16F
#define HIPBLAS_R_16B                   CUDA_R_16BF
#define HIPBLAS_C_16B                   CUDA_C_16BF
#define HIPBLAS_R_32F                   CUDA_R_32F
#define HIPBLAS_C_32F                   CUDA_C_32F
#define HIPBLAS_R_64F                   CUDA_R_64F
#define HIPBLAS_C_64F                   CUDA_C_64F
#define HIPBLAS_R_8I                    CUDA_R_8I
#define HIPBLAS_C_8I                    CUDA_C_8I
#define HIPBLAS_R_8U                    CUDA_R_8U
#define HIPBLAS_C_8U                    CUDA_C_8U
#define HIPBLAS_R_32I                   CUDA_R_32I
#define HIPBLAS_C_32I                   CUDA_C_32I

#define HIPBLAS_STATUS_SUCCESS          CUBLAS_STATUS_SUCCESS
#define HIPBLAS_STATUS_NOT_INITIALIZED  CUBLAS_STATUS_NOT_INITIALIZED
#define HIPBLAS_STATUS_ALLOC_FAILED     CUBLAS_STATUS_ALLOC_FAILED
#define HIPBLAS_STATUS_INVALID_VALUE    CUBLAS_STATUS_INVALID_VALUE
#define HIPBLAS_STATUS_MAPPING_ERROR    CUBLAS_STATUS_MAPPING_ERROR
#define HIPBLAS_STATUS_EXECUTION_FAILED CUBLAS_STATUS_EXECUTION_FAILED
#define HIPBLAS_STATUS_INTERNAL_ERROR   CUBLAS_STATUS_INTERNAL_ERROR
#define HIPBLAS_STATUS_NOT_SUPPORTED    CUBLAS_STATUS_NOT_SUPPORTED
#define HIPBLAS_STATUS_ARCH_MISMATCH    CUBLAS_STATUS_ARCH_MISMATCH
