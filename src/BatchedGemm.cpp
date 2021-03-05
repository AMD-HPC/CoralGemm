//------------------------------------------------------------------------------
/// \file
/// \brief      implementations of BatchedGemm methods
/// \date       2020-2021
/// \author     Jakub Kurzak
/// \copyright  Advanced Micro Devices, Inc.
///
#include "DeviceBatchedGemm.h"

//------------------------------------------------------------------------------
BatchedGemm*
BatchedGemm::make(std::string type_a_name,
                  std::string type_b_name,
                  std::string type_c_name,
                  std::string compute_type_name,
                  std::string op_a_name,
                  std::string op_b_name,
                  int m, int n, int k,
                  int lda, int ldb, int ldc,
                  int batch_count,
                  void const* alpha,
                  void const* beta,
                  int device_id)
{
    hipblasOperation_t op_a = stringToOp(op_a_name);
    hipblasOperation_t op_b = stringToOp(op_b_name);

    BaseBatchArray* a = BaseBatchArray::make(stringToType(type_a_name),
                                             op_a == HIPBLAS_OP_N ? m : k,
                                             op_a == HIPBLAS_OP_N ? k : m,
                                             lda,
                                             batch_count,
                                             device_id);

    BaseBatchArray* b = BaseBatchArray::make(stringToType(type_b_name),
                                             op_b == HIPBLAS_OP_N ? k : n,
                                             op_b == HIPBLAS_OP_N ? n : k,
                                             ldb,
                                             batch_count,
                                             device_id);

    BaseBatchArray* c = BaseBatchArray::make(stringToType(type_c_name),
                                             m,
                                             n,
                                             ldc,
                                             batch_count,
                                             device_id);

    return new DeviceBatchedGemm(stringToType(compute_type_name),
                                 stringToOp(op_a_name),
                                 stringToOp(op_b_name),
                                 a, b, c,
                                 batch_count,
                                 alpha, beta,
                                 operations(stringToType(type_c_name), m, n, k),
                                 device_id);
}

//------------------------------------------------------------------------------
void
BatchedGemm::makeDevices(std::string type_a_name,
                         std::string type_b_name,
                         std::string type_c_name,
                         std::string compute_type_name,
                         std::string op_a_name,
                         std::string op_b_name,
                         int m, int n, int k,
                         int lda, int ldb, int ldc,
                         int batch_count,
                         void const* alpha,
                         void const* beta,
                         bool host_a,
                         bool host_b,
                         bool host_c,
                         bool coherent_a,
                         bool coherent_b,
                         bool coherent_c,
                         bool shared_a,
                         bool shared_b,
                         std::vector<BatchedGemm*>& dev_gemms)
{
    int num_devices;
    HIP_CALL(hipGetDeviceCount(&num_devices));
    ASSERT(num_devices > 0);
    dev_gemms.resize(num_devices);

    hipblasOperation_t op_a = stringToOp(op_a_name);
    hipblasOperation_t op_b = stringToOp(op_b_name);

    BaseBatchArray* a;
    BaseBatchArray* b;
    for (int device_id = 0; device_id < num_devices; ++device_id) {

        // Enable peer access from device_id to device 0.
        if (device_id > 0 && (shared_a || shared_b))
            enablePeerAccess(device_id, 0);

        if (device_id == 0 || shared_a == false) {
            a = BaseBatchArray::make(stringToType(type_a_name),
                                     op_a == HIPBLAS_OP_N ? m : k,
                                     op_a == HIPBLAS_OP_N ? k : m,
                                     lda,
                                     batch_count,
                                     host_a ? -1 : device_id,
                                     coherent_a);
        }

        if (device_id == 0 || shared_b == false) {
            b = BaseBatchArray::make(stringToType(type_b_name),
                                     op_b == HIPBLAS_OP_N ? k : n,
                                     op_b == HIPBLAS_OP_N ? n : k,
                                     ldb,
                                     batch_count,
                                     host_b ? -1 : device_id,
                                     coherent_b);
        }

        BaseBatchArray* c = BaseBatchArray::make(stringToType(type_c_name),
                                                 m,
                                                 n,
                                                 ldc,
                                                 batch_count,
                                                 host_c ? -1 : device_id,
                                                 coherent_c);

        dev_gemms[device_id] =
            new DeviceBatchedGemm(stringToType(compute_type_name),
                                  stringToOp(op_a_name),
                                  stringToOp(op_b_name),
                                  a, b, c,
                                  batch_count,
                                  alpha, beta,
                                  operations(stringToType(type_c_name), m, n, k),
                                  device_id);
    }
}
