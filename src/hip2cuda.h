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
#endif
