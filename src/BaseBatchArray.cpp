//------------------------------------------------------------------------------
/// \file
/// \brief      implementations of BaseBatchArray methods
/// \date       2020-2021
/// \author     Jakub Kurzak
/// \copyright  Advanced Micro Devices, Inc.
///
#include "HostBatchArray.h"
#include "DeviceBatchArray.h"

//------------------------------------------------------------------------------
// https://docs.nvidia.com/cuda/cublas/index.html#cuda_datatype_tStream<T>*
//
BaseBatchArray*
BaseBatchArray::make(hipblasDatatype_t type,
                     int m, int n, int ld,
                     int batch_count,
                     int device_id,
                     bool coherent)
{
    if (device_id < 0) {
        // Return HostBatchArray
        switch (type) {
//          case HIPBLAS_R_16F:
            case HIPBLAS_R_32F:
                return new HostBatchArray<float>(
                    type, m, n, ld, batch_count, coherent);
            case HIPBLAS_R_64F:
                return new HostBatchArray<double>(
                    type, m, n, ld, batch_count, coherent);
//          case HIPBLAS_C_16F:
            case HIPBLAS_C_32F:
                return new HostBatchArray<std::complex<float>>(
                    type, m, n, ld, batch_count, coherent);
            case HIPBLAS_C_64F:
                return new HostBatchArray<std::complex<double>>(
                    type, m, n, ld, batch_count, coherent);
            case HIPBLAS_R_8I:
                return new HostBatchArray<int8_t>(
                    type, m, n, ld, batch_count, coherent);
//          case HIPBLAS_R_8U:
//              return new HostBatchArray<uint8_t>(
//                  type, m, n, ld, batch_count, coherent);
            case HIPBLAS_R_32I:
                return new HostBatchArray<int32_t>(
                    type, m, n, ld, batch_count);
//          case HIPBLAS_R_32U:
//              return new HostBatchArray<uint32_t>(
//                  type, m, n, ld, batch_count, coherent);
//          case HIPBLAS_C_8I:
//          case HIPBLAS_C_8U:
//          case HIPBLAS_C_32I:
//          case HIPBLAS_C_32U:
//          case HIPBLAS_R_16B:
//          case HIPBLAS_C_16B:
            default:
                ERROR("unsupported data type");
        }
    }
    else {
        // Return DeviceBatchArray
        switch (type) {
//          case HIPBLAS_R_16F:
            case HIPBLAS_R_32F:
                return new DeviceBatchArray<float>(
                    type, m, n, ld, batch_count, device_id);
            case HIPBLAS_R_64F:
                return new DeviceBatchArray<double>(
                    type, m, n, ld, batch_count, device_id);
//          case HIPBLAS_C_16F:
            case HIPBLAS_C_32F:
                return new DeviceBatchArray<std::complex<float>>(
                    type, m, n, ld, batch_count, device_id);
            case HIPBLAS_C_64F:
                return new DeviceBatchArray<std::complex<double>>(
                    type, m, n, ld, batch_count, device_id);
            case HIPBLAS_R_8I:
                return new DeviceBatchArray<int8_t>(
                    type, m, n, ld, batch_count, device_id);
//          case HIPBLAS_R_8U:
//              return new DeviceBatchArray<uint8_t>(
//                  type, m, n, ld, batch_count, device_id);
            case HIPBLAS_R_32I:
                return new DeviceBatchArray<int32_t>(
                    type, m, n, ld, batch_count, device_id);
//          case HIPBLAS_R_32U:
//              return new DeviceBatchArray<uint32_t>(
//                  type, m, n, ld, batch_count, device_id);
//          case HIPBLAS_C_8I:
//          case HIPBLAS_C_8U:
//          case HIPBLAS_C_32I:
//          case HIPBLAS_C_32U:
//          case HIPBLAS_R_16B:
//          case HIPBLAS_C_16B:
            default:
                ERROR("unsupported data type");
        }
    }
}
