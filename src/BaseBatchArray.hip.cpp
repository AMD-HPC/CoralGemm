//------------------------------------------------------------------------------
/// \file
/// \brief      implementations of BaseBatchArray methods
/// \date       2020-2024
/// \author     Jakub Kurzak
/// \copyright  Advanced Micro Devices, Inc.
///
#include "HostBatchArray.h"
#include "DeviceBatchArray.h"

#include <hip/hip_runtime.h>

//------------------------------------------------------------------------------
/// \brief
///     Creates either a HostBatchArray or a DeviceBatchArray.
///
/// \param[in] type
///     the data type, e.g., R_64F
///
/// \param[in] m, n, ld
///     the width, height, and leading dimension of the matrices in the batch
///
/// \param[in] batch_count
///     the number of matrices in the batch
///
/// \param[in] device_id
///     Zero of larger indicates the number of the GPU storing the array.
///     Negative number indicates that the array is stored in host memory.
///
/// \param[in] coherent
///     If true, host memory allocated as hipHostMallocCoherent (not cached).
///     If false, host memory allocated as hipHostMallocNonCoherent (cached).
///
/// \remark
///     Currently, supports a subset of types.
///     The full list of types is available in the cuBLAS User Guide
///     (https://docs.nvidia.com/cuda/cublas/index.html#
///     cuda_datatype_tStream<T>*).
///
BaseBatchArray*
BaseBatchArray::make(TypeConstant type,
                     int m, int n, int ld,
                     int batch_count,
                     int device_id,
                     bool coherent)
{
    if (device_id < 0) {
        // Return HostBatchArray
        switch (type.hip_) {
            case HIP_R_8F_E4M3:
                return new HostBatchArray<fp8>(
                    type, m, n, ld, batch_count, coherent);
            case HIP_R_8F_E5M2:
                return new HostBatchArray<bf8>(
                    type, m, n, ld, batch_count, coherent);
            case HIP_R_16F:
                return new HostBatchArray<__half>(
                    type, m, n, ld, batch_count, coherent);
            case HIP_R_16BF:
                return new HostBatchArray<hip_bfloat16>(
                    type, m, n, ld, batch_count, coherent);
            case HIP_R_32F:
                return new HostBatchArray<float>(
                    type, m, n, ld, batch_count, coherent);
            case HIP_R_64F:
                return new HostBatchArray<double>(
                    type, m, n, ld, batch_count, coherent);
            case HIP_C_32F:
                return new HostBatchArray<std::complex<float>>(
                    type, m, n, ld, batch_count, coherent);
            case HIP_C_64F:
                return new HostBatchArray<std::complex<double>>(
                    type, m, n, ld, batch_count, coherent);
            case HIP_R_8I:
                return new HostBatchArray<int8_t>(
                    type, m, n, ld, batch_count, coherent);
            case HIP_R_32I:
                return new HostBatchArray<int32_t>(
                    type, m, n, ld, batch_count);
            default:
                ERROR("unsupported data type");
        }
    }
    else {
        // Return DeviceBatchArray
        switch (type.hip_) {
            case HIP_R_8F_E4M3:
                return new DeviceBatchArray<fp8>(
                    type, m, n, ld, batch_count, device_id);
            case HIP_R_8F_E5M2:
                return new DeviceBatchArray<bf8>(
                    type, m, n, ld, batch_count, device_id);
            case HIP_R_16F:
                return new DeviceBatchArray<__half>(
                    type, m, n, ld, batch_count, device_id);
            case HIP_R_16BF:
                return new DeviceBatchArray<hip_bfloat16>(
                    type, m, n, ld, batch_count, device_id);
            case HIP_R_32F:
                return new DeviceBatchArray<float>(
                    type, m, n, ld, batch_count, device_id);
            case HIP_R_64F:
                return new DeviceBatchArray<double>(
                    type, m, n, ld, batch_count, device_id);
            case HIP_C_32F:
                return new DeviceBatchArray<std::complex<float>>(
                    type, m, n, ld, batch_count, device_id);
            case HIP_C_64F:
                return new DeviceBatchArray<std::complex<double>>(
                    type, m, n, ld, batch_count, device_id);
            case HIP_R_8I:
                return new DeviceBatchArray<int8_t>(
                    type, m, n, ld, batch_count, device_id);
            case HIP_R_32I:
                return new DeviceBatchArray<int32_t>(
                    type, m, n, ld, batch_count, device_id);
            default:
                ERROR("unsupported data type");
        }
    }
}
