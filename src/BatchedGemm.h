//------------------------------------------------------------------------------
/// \file
/// \brief      BatchedGemm class declaration and inline routines
/// \date       2020-2024
/// \author     Jakub Kurzak
/// \copyright  Advanced Micro Devices, Inc.
///
#pragma once

#include "Exception.h"
#include "BaseBatchArray.h"

#include <vector>

//------------------------------------------------------------------------------
/// \brief
///     Implements benchmarking of matrix multiplication.
///     Serves as the parent class for DeviceBatchedGemm.
///
class BatchedGemm {
public:
    /// Indicates which API to use for running the workload.
    enum class Mode {
        Standard,         ///< regular GEMM, one precision
        Batched,          ///< batched GEMM, one precision
        StridedBatched,   ///< strided GEMM, one precision
        StandardEx,       ///< regular GEMM, multi-precision
        BatchedEx,        ///< batched GEMM, multi-precision
        StridedBatchedEx, ///< strided GEMM, multi-precision
        StandardLt,       ///< regular GEMM, hipBLASLt
        BatchedLt         ///< strided GEMM, hibBLASLt
    };

    /// Returns the hipblasOperation_t corresponding to the string.
    static hipblasOperation_t stringToOp(std::string const& op)
    {
        static std::map<std::string, hipblasOperation_t> ops {
            {"OP_N", HIPBLAS_OP_N}, // non-transposed
            {"OP_T", HIPBLAS_OP_T}, // transposed
            {"OP_C", HIPBLAS_OP_C}  // conjugate-transposed
        };
        return ops[op];
    }

    /// Returns the TypeConstant corresponding to the string.
    static TypeConstant stringToType(std::string type)
    {
        static std::map<std::string, TypeConstant> types {
            {"R_64F", R_64F}, // 64 bit floating point, real
            {"R_32F", R_32F}, // 32 bit floating point, real
            {"R_16F", R_16F}, // 16 bit floating point, real
            {"R_16B", R_16B}, // 16 bit bfloat, real
            {"R_8F",  R_8F},  //  8 bit floating poing (e4m3)
            {"R_8B",  R_8B},  //  8 bit bfloat (e5m2)
            {"R_32I", R_32I}, // 32 bit signed integer, real
            {"R_8I",  R_8I},  //  8 bit signed integer, real
            {"C_64F", C_64F}, // 64 bit floating point, complex
            {"C_32F", C_32F}, // 32 bit floating point, complex
        };
        return types[type];
    }

    /// \brief
    ///     Returns the number of floating point operations
    ///     for one member of the batch (single matrix multiply).
    ///     Returns 2*m*n*k for real types and 8*m*n*k for complex types.
    ///
    static double operations(TypeConstant type, int m, int n, int k)
    {
        switch (type.hip_) {
            case HIP_R_64F:
            case HIP_R_32F:
            case HIP_R_16F:
            case HIP_R_16BF:
            case HIP_R_32I:
            case HIP_R_8I:
                return 2.0*m*n*k;
            case HIP_C_64F:
            case HIP_C_32F:
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
                             int device_id,
                             bool lt);

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
                            bool lt,
                            bool host_a,
                            bool host_b,
                            bool host_c,
                            bool coherent_a,
                            bool coherent_b,
                            bool coherent_c,
                            bool shared_a,
                            bool shared_b,
                            std::vector<BatchedGemm*>& dev_gemms);

    /// \brief
    ///     Creates a BatchedGemm object.
    ///
    /// \param[in] compute_type
    ///     the computing precision (data type),
    ///     e.g., HIPBLAS_R_32F, HIPBLAS_R_64F, HIPBLAS_C_32F, HIPBLAS_C_64F
    ///
    /// \param[in] op_a, op_b
    ///     the transposition opetations for A and B,
    ///     i.e., HIPBLAS_OP_N, HIPBLAS_OP_T, or HIPBLAS_OP_C
    ///
    /// \param[in] a, b, c
    ///     the input arrays A and B, and the output array C
    ///
    /// \param[in] batch_count
    ///     the number of matrices in the batch
    ///
    /// \param[in] alpha, beta
    ///     the alpha and beta factors in matrix multiplication
    ///
    /// \param[in] operations
    ///     number of floating point operations
    ///     for one member of the batch (single matrix multiply)
    ///
    BatchedGemm(TypeConstant compute_type,
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
    virtual ~BatchedGemm()
    {
        delete a_;
        delete b_;
        delete c_;
    }

    /// Populates the batch with random data.
    virtual void generateUniform() = 0;

    /// Populates the batch with a specific value.
    virtual void generateConstant(double val) = 0;

    /// Checks if the output contains a specific value.
    virtual void validateConstant(double val) = 0;

    /// Runs the workload.
    virtual void run(Mode mode) = 0;

    /// \brief
    ///     Returns the GLOPS number for the run.
    ///
    /// \param[in] mode
    ///     the mode of operation:
    ///     - Standard:        regular GEMM, one precision
    ///     - Batched:         batched GEMM, one precision
    ///     - StridedBatched:  strided GEMM, one precision
    ///     - StandardEx:      regular GEMM, multi-precision
    ///     - BatchedEx:       batched GEMM, multi-precision
    ///     - StridedBatchedEx strided GEMM, multi-precision
    ///
    virtual std::pair<double, double> getGflops(Mode mode) = 0;

protected:
    /// \brief
    ///     Enables peer access from one device to another.
    ///     Only enables if the path not already enabled.
    ///     Stores a map of enabled connections.
    ///
    /// \param[in] from
    ///     the device accessing the memory of another device
    ///
    /// \param[in] to
    ///     the device being accessed
    ///
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

    TypeConstant compute_type_; ///< e.g. R_32F
    hipblasOperation_t op_a_;   ///< HIPBLAS_OP_[N|T|C]
    hipblasOperation_t op_b_;   ///< HIPBLAS_OP_[N|T|C]
    BaseBatchArray* a_;         ///< the input matrices A
    BaseBatchArray* b_;         ///< the input matrices B
    BaseBatchArray* c_;         ///< the output matrices C
    int batch_count_;           ///< the number of matrices in the batch
    void const* alpha_;         ///< the alpha factor in matrix multiply
    void const* beta_;          ///< the alpha factor in matrix multiply

    double operations_; ///< number of performed floating point operations
};
