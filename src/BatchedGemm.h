 //------------------------------------------------------------------------------
/// \file
/// \brief      BatchedGemm class declaration and inline routines
/// \date       2020-2021
/// \author     Jakub Kurzak
/// \copyright  Advanced Micro Devices, Inc.
///
#pragma once

#include "Exception.h"
#include "BaseBatchArray.h"

#include <vector>

//------------------------------------------------------------------------------
class BatchedGemm {
public:
    enum class Mode {
        Standard,        ///< regular GEMM, one precision
        Batched,         ///< batched GEMM, one precision
        StridedBatched,  ///< strided GEMM, one precision
        StandardEx,      ///< regular GEMM, multi-precision
        BatchedEx,       ///< batched GEMM, multi-precision
        StridedBatchedEx ///< strided GEMM, multi-precision
    };

    static hipblasOperation_t stringToOp(std::string const& op)
    {
        static std::map<std::string, hipblasOperation_t> ops {
            {"OP_N", HIPBLAS_OP_N},
            {"OP_T", HIPBLAS_OP_T},
            {"OP_C", HIPBLAS_OP_C}
        };
        return ops[op];
    }

    static hipblasDatatype_t stringToType(std::string type)
    {
        static std::map<std::string, hipblasDatatype_t> types {
            {"R_16F", HIPBLAS_R_16F}, // 16 bit floating point, real
            {"R_32F", HIPBLAS_R_32F}, // 32 bit floating point, real
            {"R_64F", HIPBLAS_R_64F}, // 64 bit floating point, real
            {"C_16F", HIPBLAS_C_16F}, // 16 bit floating point, complex
            {"C_32F", HIPBLAS_C_32F}, // 32 bit floating point, complex
            {"C_64F", HIPBLAS_C_64F}, // 64 bit floating point, complex
            {"R_8I",  HIPBLAS_R_8I},  //  8 bit signed integer, real
            {"R_8U",  HIPBLAS_R_8U},  //  8 bit unsigned integer, real
            {"R_32I", HIPBLAS_R_32I}, // 32 bit signed integer, real
            {"R_32U", HIPBLAS_R_32U}, // 32 bit unsigned integer, real
            {"C_8I",  HIPBLAS_C_8I},  //  8 bit signed integer, complex
            {"C_8U",  HIPBLAS_C_8U},  //  8 bit unsigned integer, complex
            {"C_32I", HIPBLAS_C_32I}, // 32 bit signed integer, complex
            {"C_32U", HIPBLAS_C_32U}, // 32 bit unsigned integer, complex
            {"R_16B", HIPBLAS_R_16B}, // 16 bit bfloat, real
            {"C_16B", HIPBLAS_C_16B}, // 16 bit bfloat, complex
        };
        return types[type];
    }

    static double operations(hipblasDatatype_t type, int m, int n, int k)
    {
        switch (type) {
            case HIPBLAS_R_16F:
            case HIPBLAS_R_32F:
            case HIPBLAS_R_64F:
            case HIPBLAS_R_8I:
            case HIPBLAS_R_8U:
            case HIPBLAS_R_32I:
            case HIPBLAS_R_32U:
            case HIPBLAS_R_16B:
                return 2.0*m*n*k;
            case HIPBLAS_C_16F:
            case HIPBLAS_C_32F:
            case HIPBLAS_C_64F:
            case HIPBLAS_C_8I:
            case HIPBLAS_C_8U:
            case HIPBLAS_C_32I:
            case HIPBLAS_C_32U:
            case HIPBLAS_C_16B:
                return 8.0*m*n*k;
            default:
                ERROR("unsupported data type");
        }
    }

    static BatchedGemm* make(std::string type_a_name,
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
                             int device_id = 0);

    static void makeDevices(std::string type_a_name,
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
                            std::vector<BatchedGemm*>& dev_gemms);

    BatchedGemm(hipblasDatatype_t compute_type,
                hipblasOperation_t op_a,
                hipblasOperation_t op_b,
                BaseBatchArray* a,
                BaseBatchArray* b,
                BaseBatchArray* c,
                int batch_count,
                void const* alpha,
                void const* beta,
                double operations)
        : compute_type_(compute_type),
          op_a_(op_a), op_b_(op_b),
          a_(a), b_(b), c_(c),
          batch_count_(batch_count),
          alpha_(alpha), beta_(beta),
          operations_(operations) {}
    ~BatchedGemm() {}

    virtual void generateUniform() = 0;
    virtual void generateConstant(double val) = 0;
    virtual void run(Mode mode) = 0;
    virtual double getGflops(Mode mode) = 0;

protected:
    static void enablePeerAccess(int from, int to)
    {
        static std::map<std::tuple<int, int>, bool> peer_access;
        auto from_to = peer_access.find({from, to});
        if (from_to == peer_access.end()) {
            HIP_CALL(hipSetDevice(from));
            HIP_CALL(hipDeviceEnablePeerAccess(to, 0));
            peer_access[{from, to}] = true;
        }
    }

    hipblasDatatype_t compute_type_;
    hipblasOperation_t op_a_; // HIPBLAS_OP_N, HIPBLAS_OP_T, HIPBLAS_OP_C
    hipblasOperation_t op_b_; // HIPBLAS_OP_N, HIPBLAS_OP_T, HIPBLAS_OP_C
    BaseBatchArray* a_;
    BaseBatchArray* b_;
    BaseBatchArray* c_;
    int batch_count_;
    void const* alpha_;
    void const* beta_;

    double operations_; ///< number of performed floating point operations
};
