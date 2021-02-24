//------------------------------------------------------------------------------
/// \file
/// \brief      HIP to CUDA name replacements
/// \date       2020-2021
/// \author     Jakub Kurzak
/// \copyright  Advanced Micro Devices, Inc.
///
#if defined(__NVCC__) && !defined(__HIP_PLATFORM__)
    #define hipSuccess                   cudaSuccess
    #define hipStream_t                  cudaStream_t
    #define hipStreamCreate              cudaStreamCreate
    #define hipStreamDestroy             cudaStreamDestroy
    #define hipGetDeviceCount            cudaGetDeviceCount
    #define hipSetDevice                 cudaSetDevice
    #define hipDeviceSynchronize         cudaDeviceSynchronize
    #define hipHostMalloc                cudaMallocHost
    #define hipMalloc                    cudaMalloc
    #define hipFree                      cudaFree
    #define hipMemcpy                    cudaMemcpy
    #define hipMemcpyHostToDevice        cudaMemcpyHostToDevice
    #define hipEvent_t                   cudaEvent_t
    #define hipEventCreate               cudaEventCreate
    #define hipEventDestroy              cudaEventDestroy
    #define hipEventRecord               cudaEventRecord
    #define hipEventSynchronize          cudaEventSynchronize
    #define hipEventElapsedTime          cudaEventElapsedTime

    #define HIPRAND_RNG_PSEUDO_DEFAULT   CURAND_RNG_PSEUDO_DEFAULT
    #define HIPRAND_STATUS_SUCCESS       CURAND_STATUS_SUCCESS
    #define hiprandGenerator_t           curandGenerator_t
    #define hiprandCreateGenerator       curandCreateGenerator
    #define hiprandDestroyGenerator      curandDestroyGenerator
    #define hiprandSetStream             curandSetStream
    #define hiprandGenerate              curandGenerate
    #define hiprandGenerateUniform       curandGenerateUniform
    #define hiprandGenerateUniformDouble curandGenerateUniformDouble

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
