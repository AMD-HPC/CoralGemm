//------------------------------------------------------------------------------
/// \file
/// \brief      HIP to CUDA name replacements
/// \date       2020-2021
/// \author     Jakub Kurzak
/// \copyright  Advanced Micro Devices, Inc.
///
#define HIPRAND_RNG_PSEUDO_DEFAULT   CURAND_RNG_PSEUDO_DEFAULT
#define HIPRAND_STATUS_SUCCESS       CURAND_STATUS_SUCCESS
#define hiprandGenerator_t           curandGenerator_t
#define hiprandCreateGenerator       curandCreateGenerator
#define hiprandDestroyGenerator      curandDestroyGenerator
#define hiprandSetStream             curandSetStream
#define hiprandGenerate              curandGenerate
#define hiprandGenerateUniform       curandGenerateUniform
#define hiprandGenerateUniformDouble curandGenerateUniformDouble
