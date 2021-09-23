//------------------------------------------------------------------------------
/// \file
/// \brief      hipRAND to cuRAND name replacements
/// \date       2020-2021
/// \author     Jakub Kurzak
/// \copyright  Advanced Micro Devices, Inc.
///
#define hiprandCreateGenerator                      curandCreateGenerator
#define hiprandDestroyGenerator                     curandDestroyGenerator
#define hiprandGenerate                             curandGenerate
#define hiprandGenerateUniform                      curandGenerateUniform
#define hiprandGenerateUniformDouble                curandGenerateUniformDouble
#define hiprandGenerator_t                          curandGenerator_t
#define hiprandSetStream                            curandSetStream
#define hiprandStatus                               curandStatus

#define HIPRAND_RNG_PSEUDO_DEFAULT                  CURAND_RNG_PSEUDO_DEFAULT
#define HIPRAND_STATUS_SUCCESS                      CURAND_STATUS_SUCCESS
#define HIPRAND_STATUS_VERSION_MISMATCH             CURAND_STATUS_VERSION_MISMATCH
#define HIPRAND_STATUS_NOT_INITIALIZED              CURAND_STATUS_NOT_INITIALIZED
#define HIPRAND_STATUS_ALLOCATION_FAILED            CURAND_STATUS_ALLOCATION_FAILED
#define HIPRAND_STATUS_TYPE_ERROR                   CURAND_STATUS_TYPE_ERROR
#define HIPRAND_STATUS_OUT_OF_RANGE                 CURAND_STATUS_OUT_OF_RANGE
#define HIPRAND_STATUS_LENGTH_NOT_MULTIPLE          CURAND_STATUS_LENGTH_NOT_MULTIPLE
#define HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED    CURAND_STATUS_DOUBLE_PRECISION_REQUIRED
#define HIPRAND_STATUS_LAUNCH_FAILURE               CURAND_STATUS_LAUNCH_FAILURE
#define HIPRAND_STATUS_PREEXISTING_FAILURE          CURAND_STATUS_PREEXISTING_FAILURE
#define HIPRAND_STATUS_INITIALIZATION_FAILED        CURAND_STATUS_INITIALIZATION_FAILED
#define HIPRAND_STATUS_ARCH_MISMATCH                CURAND_STATUS_ARCH_MISMATCH
#define HIPRAND_STATUS_INTERNAL_ERROR               CURAND_STATUS_INTERNAL_ERROR
#define HIPRAND_STATUS_NOT_IMPLEMENTED              CURAND_STATUS_NOT_IMPLEMENTED
