//------------------------------------------------------------------------------
/// \file
/// \brief      HIP to CUDA name replacements
/// \date       2020-2023
/// \author     Jakub Kurzak
/// \copyright  Advanced Micro Devices, Inc.
///
#define hipDeviceEnablePeerAccess   cudaDeviceEnablePeerAccess
#define hipDeviceSynchronize        cudaDeviceSynchronize
#define hipEventCreate              cudaEventCreate
#define hipEventDestroy             cudaEventDestroy
#define hipEventElapsedTime         cudaEventElapsedTime
#define hipEventRecord              cudaEventRecord
#define hipEventSynchronize         cudaEventSynchronize
#define hipEvent_t                  cudaEvent_t
#define hipError_t                  cudaError_t
#define hipFree                     cudaFree
#define hipGetDeviceCount           cudaGetDeviceCount
#define hipGetErrorName             cudaGetErrorName
#define hipGetErrorString           cudaGetErrorString
#define hipHostMalloc               cudaMallocHost
#define hipMalloc                   cudaMalloc
#define hipMemcpy                   cudaMemcpy
#define hipMemcpyHostToDevice       cudaMemcpyHostToDevice
#define hipSetDevice                cudaSetDevice
#define hipStream_t                 cudaStream_t
#define hipStreamCreate             cudaStreamCreate
#define hipStreamDestroy            cudaStreamDestroy
#define hipSuccess                  cudaSuccess
#define hip_bfloat16                __nv_bfloat16
