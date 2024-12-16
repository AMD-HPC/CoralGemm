//------------------------------------------------------------------------------
/// \file
/// \brief      exception handling
/// \date       2020-2024
/// \author     Jakub Kurzak
/// \copyright  Advanced Micro Devices, Inc.
///
#pragma once

#include <cstdio>
#include <exception>
#include <map>
#include <string>

#if defined(USE_HIP)
    #include <hip/hip_runtime.h>
    #include <hipblas/hipblas.h>
    #include <hiprand/hiprand.h>
#elif defined(USE_CUDA)
    #include <cuda_runtime.h>
    #include <cublas_v2.h>
    #include <curand.h>
    #include "hip2cuda.h"
    #include "hipblas2cublas.h"
    #include "hiprand2curand.h"
#endif

//------------------------------------------------------------------------------
/// \brief
///     Implements the base class for exception handling.
///
class Exception : public std::exception {
public:
    Exception() : std::exception() {}

    Exception(std::string const& msg,
              const char* func, const char* file, int line)
        : std::exception(),
          msg_(msg+"\n"+func+"() | "+file+" | L:"+std::to_string(line)),
          func_(func), file_(file), line_(line) {}

    virtual char const* what() const noexcept override
    {
        return msg_.c_str();
    }

protected:
    void what(std::string const& msg,
              const char* func, const char* file, int line)
    {
        msg_ = msg+"\n"+func+"() | "+file+" | "+std::to_string(line)+"\033[0m";
    }

    std::string msg_;
    std::string func_;
    std::string file_;
    int line_;
};

/// Report errors.
#define ERROR(msg) \
{ \
    throw Exception(std::string("\033[38;5;200mERROR:\033[38;5;255m ")+ \
                    msg, __func__, __FILE__, __LINE__); \
}

//------------------------------------------------------------------------------
/// \brief
///     Implements exception handling for the ERROR_IF macro.
///
class TrueConditionException : public Exception {
public:
    TrueConditionException(const char* condition,
                           const char* func,
                           const char* file,
                           int line)
        : Exception(std::string("\033[38;5;200mERROR:\033[38;5;255m ")+
                    "Condition '"+condition+"' is true.",
                    func, file, line) {}

    TrueConditionException(const char* condition,
                           const char* description,
                           const char* func,
                           const char* file,
                           int line)
        : Exception(std::string("\033[38;5;200mERROR:\033[38;5;255m ")+
                    description+" Condition '"+condition+"' is true.",
                    func, file, line) {}
};

/// Checks error conditions.
#define ERROR_IF(condition, ...) \
{ \
    if (condition) \
        throw TrueConditionException(#condition, ##__VA_ARGS__, \
                                     __func__, __FILE__, __LINE__); \
}

//------------------------------------------------------------------------------
/// \brief
///     Implements exception handling for the ASSERT macro.
///
class FalseConditionException : public Exception {
public:
    FalseConditionException(const char* assertion,
                            const char* func,
                            const char* file,
                            int line)
        : Exception(std::string("\033[38;5;200mERROR:\033[38;5;255m ")+
                    "Assertion '"+assertion+"' is false.",
                    func, file, line) {}

    FalseConditionException(const char* assertion,
                            const char* description,
                            const char* func,
                            const char* file,
                            int line)
        : Exception(std::string("\033[38;5;200mERROR:\033[38;5;255m ")+
                    description+" Assertion '"+assertion+"' is false.",
                    func, file, line) {}
};

/// Checks assertions.
#define ASSERT(assertion, ...) \
{ \
    if (!(assertion)) \
        throw FalseConditionException(#assertion, ##__VA_ARGS__, \
                                      __func__, __FILE__, __LINE__); \
}

//------------------------------------------------------------------------------
/// \brief
///     Implements exception handling for the HIP_CALL macro.
///
class HIPException : public Exception {
public:
    HIPException(const char* call,
                 hipError_t code,
                 const char* func,
                 const char* file,
                 int line)
        : Exception()
    {
        const char* name = hipGetErrorName(code);
        const char* string = hipGetErrorString(code);
        what(std::string("\033[38;5;200mHIP ERROR:\033[38;5;255m ")+
             call+" returned "+name+" ("+string+").",
             func, file, line);
    }

    HIPException(const char* call,
                 hipError_t code,
                 const char* description,
                 const char* func,
                 const char* file,
                 int line)
        : Exception()
    {
        char const* name = hipGetErrorName(code);
        char const* string = hipGetErrorString(code);
        what(std::string("\033[38;5;200mHIP ERROR:\033[38;5;255m ")+
             description+" \n"+ call+" returned "+name+" ("+string+").",
             func, file, line);
    }
};

/// Checks for errors in HIP calls.
#define HIP_CALL(call, ...) \
{ \
    hipError_t code = call; \
    if (code != hipSuccess) \
        throw HIPException(#call, code, ##__VA_ARGS__, \
                           __func__, __FILE__, __LINE__); \
}

//------------------------------------------------------------------------------
/// \brief
///     Implements exception handling for the HIPBLAS_CALL macro.
///
/// \todo
///     Check if error names and strings not out of sync with hipBLAS
///     (https://github.com/ROCmSoftwarePlatform/hipBLAS/blob/develop/library/
///     include/hipblas.h).
///
class HIPBLASException : public Exception {
public:
    HIPBLASException(const char* call,
                     hipblasStatus_t code,
                     const char* func,
                     const char* file,
                     int line)
        : Exception(std::string("\033[38;5;200mHIPBLAS ERROR:\033[38;5;255m ")+
                    call+" returned "+errorName(code)+
                    " ("+errorString(code)+").", func, file, line) {}

    HIPBLASException(const char* call,
                     hipblasStatus_t code,
                     const char* description,
                     const char* func,
                     const char* file,
                     int line)
        : Exception(std::string("\033[38;5;200mHIPBLAS ERROR:\033[38;5;255m ")+
                    description+" \n"+call+" returned "+errorName(code)+
                    " ("+errorString(code)+").", func, file, line) {}

private:
    std::string const& errorName(hipblasStatus_t code)
    {
        static std::map<int, std::string> error_names {
            {HIPBLAS_STATUS_SUCCESS,
            "HIPBLAS_STATUS_SUCCESS"},
            {HIPBLAS_STATUS_NOT_INITIALIZED,
            "HIPBLAS_STATUS_NOT_INITIALIZED"},
            {HIPBLAS_STATUS_ALLOC_FAILED,
            "HIPBLAS_STATUS_ALLOC_FAILED"},
            {HIPBLAS_STATUS_INVALID_VALUE,
            "HIPBLAS_STATUS_INVALID_VALUE"},
            {HIPBLAS_STATUS_MAPPING_ERROR,
            "HIPBLAS_STATUS_MAPPING_ERROR"},
            {HIPBLAS_STATUS_EXECUTION_FAILED,
            "HIPBLAS_STATUS_EXECUTION_FAILED"},
            {HIPBLAS_STATUS_INTERNAL_ERROR,
            "HIPBLAS_STATUS_INTERNAL_ERROR"},
            {HIPBLAS_STATUS_NOT_SUPPORTED,
            "HIPBLAS_STATUS_NOT_SUPPORTED"},
            {HIPBLAS_STATUS_ARCH_MISMATCH,
            "HIPBLAS_STATUS_ARCH_MISMATCH"},
#if !defined(USE_CUDA)
            {HIPBLAS_STATUS_HANDLE_IS_NULLPTR,
            "HIPBLAS_STATUS_HANDLE_IS_NULLPTR"}
#endif
        };
        return error_names[code];
    }

    std::string const& errorString(hipblasStatus_t code)
    {
        static std::map<int, std::string> error_strings {
            {HIPBLAS_STATUS_SUCCESS,
            "Function succeeds"},
            {HIPBLAS_STATUS_NOT_INITIALIZED,
            "HIPBLAS library not initialized"},
            {HIPBLAS_STATUS_ALLOC_FAILED,
            "resource allocation failed"},
            {HIPBLAS_STATUS_INVALID_VALUE,
            "unsupported numerical value was passed to function"},
            {HIPBLAS_STATUS_MAPPING_ERROR,
            "access to GPU memory space failed"},
            {HIPBLAS_STATUS_EXECUTION_FAILED,
            "GPU program failed to execute"},
            {HIPBLAS_STATUS_INTERNAL_ERROR,
            "an internal HIPBLAS operation failed"},
            {HIPBLAS_STATUS_NOT_SUPPORTED,
            "function not implemented"},
            {HIPBLAS_STATUS_ARCH_MISMATCH,
            ""},
#if !defined(USE_CUDA)
            {HIPBLAS_STATUS_HANDLE_IS_NULLPTR,
            "hipBLAS handle is null pointer"}
#endif
        };
        return error_strings[code];
    }
};

/// Checks for errors in HIPBLAS calls.
#define HIPBLAS_CALL(call, ...) \
{ \
    hipblasStatus_t code = call; \
    if (code != HIPBLAS_STATUS_SUCCESS) \
        throw HIPBLASException(#call, code, ##__VA_ARGS__, \
                               __func__, __FILE__, __LINE__); \
}

//------------------------------------------------------------------------------
/// \brief
///     Implements exception handling for the HIPBLASLT_CALL macro.
///
/// \todo
///     Check if error names and strings not out of sync with hipBLAS
///     (https://github.com/ROCmSoftwarePlatform/hipBLAS/blob/develop/library/
///     include/hipblas.h).
///

/// Checks for errors in HIPBLASLT calls.
#define HIPBLASLT_CALL(call, ...) \
    call



//------------------------------------------------------------------------------
/// \brief
///     Implements exception handling for the HIPRAND_CALL macro.
///
/// \todo
///     Check if error names and strings not out of sync with hipRAND
///     (https://github.com/ROCmSoftwarePlatform/rocRAND/blob/develop/library/
///     include/hiprand.h).
///
class HIPRANDException : public Exception {
public:
    HIPRANDException(const char* call,
                     hiprandStatus code,
                     const char* func,
                     const char* file,
                     int line)
        : Exception(std::string("\033[38;5;200mHIPRAND ERROR:\033[38;5;255m ")+
                    call+" returned "+errorName(code)+
                    " ("+errorString(code)+").", func, file, line) {}

    HIPRANDException(const char* call,
                     hiprandStatus code,
                     const char* description,
                     const char* func,
                     const char* file,
                     int line)
        : Exception(std::string("\033[38;5;200mHIPRAND ERROR:\033[38;5;255m ")+
                    description+" \n"+call+" returned "+errorName(code)+
                    " ("+errorString(code)+").", func, file, line) {}

private:
    std::string const& errorName(hiprandStatus code)
    {
        static std::map<int, std::string> error_names {
            {HIPRAND_STATUS_SUCCESS,
            "HIPRAND_STATUS_SUCCESS"},
            {HIPRAND_STATUS_VERSION_MISMATCH,
            "HIPRAND_STATUS_VERSION_MISMATCH"},
            {HIPRAND_STATUS_NOT_INITIALIZED,
            "HIPRAND_STATUS_NOT_INITIALIZED"},
            {HIPRAND_STATUS_ALLOCATION_FAILED,
            "HIPRAND_STATUS_ALLOCATION_FAILED"},
            {HIPRAND_STATUS_TYPE_ERROR,
            "HIPRAND_STATUS_TYPE_ERROR"},
            {HIPRAND_STATUS_OUT_OF_RANGE,
            "HIPRAND_STATUS_OUT_OF_RANGE"},
            {HIPRAND_STATUS_LENGTH_NOT_MULTIPLE,
            "HIPRAND_STATUS_LENGTH_NOT_MULTIPLE"},
            {HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED,
            "HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED"},
            {HIPRAND_STATUS_LAUNCH_FAILURE,
            "HIPRAND_STATUS_LAUNCH_FAILURE"},
            {HIPRAND_STATUS_PREEXISTING_FAILURE,
            "HIPRAND_STATUS_PREEXISTING_FAILURE"},
            {HIPRAND_STATUS_INITIALIZATION_FAILED,
            "HIPRAND_STATUS_INITIALIZATION_FAILED"},
            {HIPRAND_STATUS_ARCH_MISMATCH,
            "HIPRAND_STATUS_ARCH_MISMATCH"},
            {HIPRAND_STATUS_INTERNAL_ERROR,
            "HIPRAND_STATUS_INTERNAL_ERROR"},
#if !defined(USE_CUDA)
            {HIPRAND_STATUS_NOT_IMPLEMENTED,
            "HIPRAND_STATUS_NOT_IMPLEMENTED"}
#endif
        };
        return error_names[code];
    }

    std::string const& errorString(hiprandStatus code)
    {
        static std::map<int, std::string> error_strings {
            {HIPRAND_STATUS_SUCCESS,
            "Success"},
            {HIPRAND_STATUS_VERSION_MISMATCH,
            "Header file and linked library version do not match"},
            {HIPRAND_STATUS_NOT_INITIALIZED,
            "Generator not created"},
            {HIPRAND_STATUS_ALLOCATION_FAILED,
            "Memory allocation failed"},
            {HIPRAND_STATUS_TYPE_ERROR,
            "Generator type is wrong"},
            {HIPRAND_STATUS_OUT_OF_RANGE,
            "Argument out of range"},
            {HIPRAND_STATUS_LENGTH_NOT_MULTIPLE,
            "Requested size is not a multiple of quasirandom generator's "
            "dimension, or requested size is not even "
            "(see hiprandGenerateNormal()), or pointer is misaligned "
            "(see hiprandGenerateNormal())"},
            {HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED,
            "GPU does not have double precision"},
            {HIPRAND_STATUS_LAUNCH_FAILURE,
            "Kernel launch failure"},
            {HIPRAND_STATUS_PREEXISTING_FAILURE,
            "Preexisting failure on library entry"},
            {HIPRAND_STATUS_INITIALIZATION_FAILED,
            "Initialization of HIP failed"},
            {HIPRAND_STATUS_ARCH_MISMATCH,
            "Architecture mismatch, GPU does not support requested feature"},
            {HIPRAND_STATUS_INTERNAL_ERROR,
            "Internal library error"},
#if !defined(USE_CUDA)
            {HIPRAND_STATUS_NOT_IMPLEMENTED,
            "Feature not implemented yet"}
#endif
        };
        return error_strings[code];
    }
};

/// Checks for errors in HIPRAND calls.
#define HIPRAND_CALL(call, ...) \
{ \
    hiprandStatus code = call; \
    if (code != HIPRAND_STATUS_SUCCESS) \
        throw HIPRANDException(#call, code, ##__VA_ARGS__, \
                               __func__, __FILE__, __LINE__); \
}
