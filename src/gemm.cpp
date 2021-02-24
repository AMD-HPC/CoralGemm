/*
    # build with HIP (ROCm target)
    export ROCM_PATH=/opt/rocm
    $ROCM_PATH/bin/hipcc \
        -I$ROCM_PATH/include \
        -I$ROCM_PATH/include/hiprand -I$ROCM_PATH/include/rocrand \
        -L$ROCM_PATH/lib -lhipblas -lhiprand -lrocblas -lrocrand \
        -O3 -std=c++11 \
        --amdgpu-target=gfx906,gfx908 \
        gemm.cpp -o gemm

    # build with HIP (CUDA target)
    export CUDA_PATH=/usr/local/cuda
    $CUDA_PATH/bin/nvcc -D__HIP_PLATFORM__=NVCC \
        -I/opt/rocm/include \
        -I/opt/rocm/include/hiprand \
        -L$ROCM_PATH/lib -lhipblas -lhiprand -lcublas -lcurand \
        -O3 -std=c++11 \
        -Wno-deprecated-declarations \
        gemm.cpp -o gemm

    # build with CUDA (HIP not required)
    export CUDA_PATH=/usr/local/cuda
    $CUDA_PATH/bin/nvcc \
        -lcublas -lcurand \
        -O3 -std=c++11 \
        -Wno-deprecated-declarations gemm.cpp -o gemm

    ./gemm <S|C|D|Z>          precision
           <NN|NT|TN|...>     transposition of A and B
           <DDD|DDH|DHH|...>  location of A, B, and C
           <M> <N> <K>        dimensions
           <LDA> <LDB> <LDC>  leading dimensions
           <BATCH_SIZE>       number of matrices to sweep
           <TIME_SPAN>        runtime duration in seconds
           [batched]          call the batched routine
           [batched strided]  call the strided batched routine
           [testing]          perform a basic sanity check

    ./gemm D NT DDD 8640 8640 8640 8640 8640 8640  9 300 (16GB  MI60)
    ./gemm S NT DDD 8640 8640 8640 8640 8640 8640 18 300 (16GB  MI60)

    ./gemm D NT DDD 8640 8640 8640 8640 8640 8640 18 300 (32GB MI100)
    ./gemm S NT DDD 8640 8640 8640 8640 8640 8640 36 300 (32GB MI100)

    ./gemm S NN DDD 512 512 512 512 512 512 1000 300
    ./gemm S NN DDD 512 512 512 512 512 512 1000 300 batched
    ./gemm S NN DDD 512 512 512 512 512 512 1000 300 batched strided

    Jakub Kurzak
    AMD Research
    7/20/2020
*/
#if defined(__HIPCC__) || defined(__HIP_PLATFORM__)
    #include <hip/hip_runtime.h>
    #include <hipblas.h>
    #include <hiprand.h>
#endif

#if defined(__NVCC__)
    #include <cuda_runtime.h>
    #include <cublas_v2.h>
    #include <curand.h>
#endif



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

    #define HIPRAND_RNG_PSEUDO_DEFAULT   CURAND_RNG_PSEUDO_DEFAULT
    #define HIPRAND_STATUS_SUCCESS       CURAND_STATUS_SUCCESS
    #define hiprandGenerator_t           curandGenerator_t
    #define hiprandCreateGenerator       curandCreateGenerator
    #define hiprandDestroyGenerator      curandDestroyGenerator
    #define hiprandSetStream             curandSetStream
    #define hiprandGenerate              curandGenerate
    #define hiprandGenerateUniform       curandGenerateUniform
    #define hiprandGenerateUniformDouble curandGenerateUniformDouble

    #define HIPBLAS_GEMM_DEFAULT         CUBLAS_GEMM_DEFAULT
    #define HIPBLAS_OP_N                 CUBLAS_OP_N
    #define HIPBLAS_OP_T                 CUBLAS_OP_T
    #define HIPBLAS_OP_C                 CUBLAS_OP_C
    #define HIPBLAS_R_32I                CUDA_R_32I
    #define HIPBLAS_R_8I                 CUDA_R_8I
    #define HIPBLAS_STATUS_SUCCESS       CUBLAS_STATUS_SUCCESS
    #define hipblasHandle_t              cublasHandle_t
    #define hipblasOperation_t           cublasOperation_t
    #define hipblasComplex               cuComplex
    #define hipblasDoubleComplex         cuDoubleComplex
    #define hipblasCreate                cublasCreate
    #define hipblasDestroy               cublasDestroy
    #define hipblasSetStream             cublasSetStream
    #define hipblasGemmEx                cublasGemmEx
    #define hipblasGemmBatchedEx         cublasGemmBatchedEx
    #define hipblasGemmStridedBatchedEx  cublasGemmStridedBatchedEx
    #define hipblasSgemm                 cublasSgemm
    #define hipblasDgemm                 cublasDgemm
    #define hipblasCgemm                 cublasCgemm
    #define hipblasZgemm                 cublasZgemm
    #define hipblasSgemmBatched          cublasSgemmBatched
    #define hipblasDgemmBatched          cublasDgemmBatched
    #define hipblasCgemmBatched          cublasCgemmBatched
    #define hipblasZgemmBatched          cublasZgemmBatched
    #define hipblasSgemmStridedBatched   cublasSgemmStridedBatched
    #define hipblasDgemmStridedBatched   cublasDgemmStridedBatched
    #define hipblasCgemmStridedBatched   cublasCgemmStridedBatched
    #define hipblasZgemmStridedBatched   cublasZgemmStridedBatched
#endif

#include <algorithm>
#include <cassert>
#include <chrono>
#include <complex>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#define CALL_HIP(call) assert(call == hipSuccess)
#define CALL_HIPRAND(call) assert(call == HIPRAND_STATUS_SUCCESS)
#define CALL_HIPBLAS(call) assert(call == HIPBLAS_STATUS_SUCCESS)

//------------------------------------------------------------------------------
// \brief Sets all entries to the given value.
template <typename T>
__global__
void matrix_set_kernel(int m, int n, T* A, int lda, T val)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    for (int j = 0; j < n; ++j) {
        if (i < m)
            A[i + j*lda] = val;
    }
}

//------------------------------------------------------------------------------
// \brief Sets all entries to the given value.
template <typename T>
void matrix_set(int m, int n, T* A, int lda, T val)
{
    int group_size = 256;
    int num_groups = m%group_size == 0 ? m/group_size : m/group_size+1;
    matrix_set_kernel<<<num_groups, group_size>>>(m, n, A, lda, val);
}

//------------------------------------------------------------------------------
// \brief Asserts that all entries equal to the given value.
template <typename T>
__global__
void matrix_assert_kernel(int m, int n, T* A, int lda, T val)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    for (int j = 0; j < n; ++j) {
        if (i < m) {
            assert(A[i + j*lda] == val);
        }
    }
}

//------------------------------------------------------------------------------
// \brief Asserts that all entries equal to the given value.
template <typename T>
void matrix_assert(int m, int n, T* A, int lda, T val)
{
    int group_size = 256;
    int num_groups = m%group_size == 0 ? m/group_size : m/group_size+1;
    matrix_assert_kernel<<<num_groups, group_size>>>(m, n, A, lda, val);
}

//------------------------------------------------------------------------------
void hiprandGenUni(
    hiprandGenerator_t generator, float* A, std::size_t len)
{
    CALL_HIPRAND(hiprandGenerateUniform(generator, A, len));
}

void hiprandGenUni(
    hiprandGenerator_t generator, double* A, std::size_t len)
{
    CALL_HIPRAND(hiprandGenerateUniformDouble(generator, A, len));
}

void hiprandGenUni(
    hiprandGenerator_t generator, std::complex<float>* A, std::size_t len)
{
    CALL_HIPRAND(hiprandGenerateUniform(generator, (float*)A, len*2));
}

void hiprandGenUni(
    hiprandGenerator_t generator, std::complex<double>* A, std::size_t len)
{
    CALL_HIPRAND(hiprandGenerateUniformDouble(generator, (double*)A, len*2));
}

void hiprandGenUni(
    hiprandGenerator_t generator, int8_t* A, std::size_t len)
{
#if defined(__NVCC__) && !defined(__HIP_PLATFORM__)
    assert(len%4 == 0);
    CALL_HIPRAND(hiprandGenerate(generator, (unsigned int*)A, len/4));
#else
    CALL_HIPRAND(hiprandGenerateChar(generator, (unsigned char*)A, len));
#endif
}

void hiprandGenUni(
    hiprandGenerator_t generator, int32_t* A, std::size_t len)
{
    CALL_HIPRAND(hiprandGenerate(generator, (unsigned int*)A, len));
}

//------------------------------------------------------------------------------
void hipblasGemm(hipblasHandle_t handle,
                 hipblasOperation_t opA, hipblasOperation_t opB,
                 int m, int n, int k,
                 float* alpha, float* A, int lda,
                               float* B, int ldb,
                 float* beta,  float* C, int ldc)
{
    CALL_HIPBLAS(hipblasSgemm(handle,
                              opA, opB,
                              m, n, k,
                              alpha, A, lda,
                                     B, ldb,
                              beta,  C, ldc));
}

void hipblasGemm(hipblasHandle_t handle,
                 hipblasOperation_t opA, hipblasOperation_t opB,
                 int m, int n, int k,
                 double* alpha, double* A, int lda,
                                double* B, int ldb,
                 double* beta,  double* C, int ldc)
{
    CALL_HIPBLAS(hipblasDgemm(handle,
                               opA, opB,
                               m, n, k,
                               alpha, A, lda,
                                      B, ldb,
                               beta,  C, ldc));
}

void hipblasGemm(hipblasHandle_t handle,
                 hipblasOperation_t opA, hipblasOperation_t opB,
                 int m, int n, int k,
                 std::complex<float>* alpha, std::complex<float>* A, int lda,
                                             std::complex<float>* B, int ldb,
                 std::complex<float>* beta,  std::complex<float>* C, int ldc)
{
    CALL_HIPBLAS(
        hipblasCgemm(
            handle,
            opA, opB,
            m, n, k,
            (hipblasComplex*)alpha, (hipblasComplex*)A, lda,
                                    (hipblasComplex*)B, ldb,
            (hipblasComplex*)beta,  (hipblasComplex*)C, ldc));
}

void hipblasGemm(hipblasHandle_t handle,
                 hipblasOperation_t opA, hipblasOperation_t opB,
                 int m, int n, int k,
                 std::complex<double>* alpha, std::complex<double>* A, int lda,
                                              std::complex<double>* B, int ldb,
                 std::complex<double>* beta,  std::complex<double>* C, int ldc)
{
    CALL_HIPBLAS(
        hipblasZgemm(
            handle,
            opA, opB,
            m, n, k,
            (hipblasDoubleComplex*)alpha, (hipblasDoubleComplex*)A, lda,
                                          (hipblasDoubleComplex*)B, ldb,
            (hipblasDoubleComplex*)beta,  (hipblasDoubleComplex*)C, ldc));
}

void hipblasGemm(hipblasHandle_t handle,
                 hipblasOperation_t opA, hipblasOperation_t opB,
                 int m, int n, int k,
                 int32_t* alpha, int8_t* A, int lda,
                                 int8_t* B, int ldb,
                 int32_t* beta, int32_t* C, int ldc)
{
    CALL_HIPBLAS(hipblasGemmEx(handle,
                               opA, opB,
                               m, n, k,
                               alpha, A, HIPBLAS_R_8I,  lda,
                                      B, HIPBLAS_R_8I,  ldb,
                               beta,  C, HIPBLAS_R_32I, ldc,
                               HIPBLAS_R_32I,
                               HIPBLAS_GEMM_DEFAULT));
}

//------------------------------------------------------------------------------
void hipblasGemmBatched(hipblasHandle_t handle,
                        hipblasOperation_t opA, hipblasOperation_t opB,
                        int m, int n, int k,
                        float* alpha, float** A, int lda,
                                      float** B, int ldb,
                        float* beta,  float** C, int ldc,
                        int batch_size)
{
    CALL_HIPBLAS(hipblasSgemmBatched(handle,
                                     opA, opB,
                                     m, n, k,
                                     alpha, A, lda,
                                            B, ldb,
                                     beta,  C, ldc,
                                     batch_size));
}

void hipblasGemmBatched(hipblasHandle_t handle,
                        hipblasOperation_t opA, hipblasOperation_t opB,
                        int m, int n, int k,
                        double* alpha, double** A, int lda,
                                       double** B, int ldb,
                        double* beta,  double** C, int ldc,
                        int batch_size)
{
    CALL_HIPBLAS(hipblasDgemmBatched(handle,
                                     opA, opB,
                                     m, n, k,
                                     alpha, A, lda,
                                            B, ldb,
                                     beta,  C, ldc,
                                     batch_size));
}

void hipblasGemmBatched(
    hipblasHandle_t handle,
    hipblasOperation_t opA, hipblasOperation_t opB,
    int m, int n, int k,
    std::complex<float>* alpha, std::complex<float>** A, int lda,
                                std::complex<float>** B, int ldb,
    std::complex<float>* beta,  std::complex<float>** C, int ldc,
    int batch_size)
{
    CALL_HIPBLAS(
        hipblasCgemmBatched(
            handle,
            opA, opB,
            m, n, k,
            (hipblasComplex*)alpha, (hipblasComplex**)A, lda,
                                    (hipblasComplex**)B, ldb,
            (hipblasComplex*)beta,  (hipblasComplex**)C, ldc,
            batch_size));
}

void hipblasGemmBatched(
    hipblasHandle_t handle,
    hipblasOperation_t opA, hipblasOperation_t opB,
    int m, int n, int k,
    std::complex<double>* alpha, std::complex<double>** A, int lda,
                                 std::complex<double>** B, int ldb,
    std::complex<double>* beta,  std::complex<double>** C, int ldc,
    int batch_size)
{
    CALL_HIPBLAS(
        hipblasZgemmBatched(
            handle,
            opA, opB,
            m, n, k,
            (hipblasDoubleComplex*)alpha, (hipblasDoubleComplex**)A, lda,
                                          (hipblasDoubleComplex**)B, ldb,
            (hipblasDoubleComplex*)beta,  (hipblasDoubleComplex**)C, ldc,
            batch_size));
}

void hipblasGemmBatched(hipblasHandle_t handle,
                        hipblasOperation_t opA, hipblasOperation_t opB,
                        int m, int n, int k,
                        int32_t* alpha, int8_t** A, int lda,
                                        int8_t** B, int ldb,
                        int32_t* beta, int32_t** C, int ldc,
                        int batch_size)
{
    CALL_HIPBLAS(
        hipblasGemmBatchedEx(handle,
                             opA, opB,
                             m, n, k,
                             alpha, (const void**)A, HIPBLAS_R_8I,  lda,
                                    (const void**)B, HIPBLAS_R_8I,  ldb,
                             beta,  (      void**)C, HIPBLAS_R_32I, ldc,
                             batch_size,
                             HIPBLAS_R_32I,
                             HIPBLAS_GEMM_DEFAULT));
}

//------------------------------------------------------------------------------
void hipblasGemmStridedBatched(
    hipblasHandle_t handle,
    hipblasOperation_t opA, hipblasOperation_t opB,
    int m, int n, int k,
    float* alpha, float* A, int lda, std::size_t strideA,
                  float* B, int ldb, std::size_t strideB,
    float* beta,  float* C, int ldc, std::size_t strideC,
    int batch_size)
{
    CALL_HIPBLAS(hipblasSgemmStridedBatched(handle,
                                            opA, opB,
                                            m, n, k,
                                            alpha, A, lda, strideA,
                                                   B, ldb, strideB,
                                            beta,  C, ldc, strideC,
                                            batch_size));
}

void hipblasGemmStridedBatched(
    hipblasHandle_t handle,
    hipblasOperation_t opA, hipblasOperation_t opB,
    int m, int n, int k,
    double* alpha, double* A, int lda, std::size_t strideA,
                   double* B, int ldb, std::size_t strideB,
    double* beta,  double* C, int ldc, std::size_t strideC,
    int batch_size)
{
    CALL_HIPBLAS(hipblasDgemmStridedBatched(handle,
                                            opA, opB,
                                            m, n, k,
                                            alpha, A, lda, strideA,
                                                   B, ldb, strideB,
                                            beta,  C, ldc, strideC,
                                            batch_size));
}

void hipblasGemmStridedBatched(
    hipblasHandle_t handle,
    hipblasOperation_t opA, hipblasOperation_t opB,
    int m, int n, int k,
    std::complex<float>* alpha,
    std::complex<float>* A, int lda, std::size_t strideA,
    std::complex<float>* B, int ldb, std::size_t strideB,
    std::complex<float>* beta,
    std::complex<float>* C, int ldc, std::size_t strideC,
    int batch_size)
{
    CALL_HIPBLAS(
        hipblasCgemmStridedBatched(
            handle,
            opA, opB,
            m, n, k,
            (hipblasComplex*)alpha, (hipblasComplex*)A, lda, strideA,
                                    (hipblasComplex*)B, ldb, strideB,
            (hipblasComplex*)beta,  (hipblasComplex*)C, ldc, strideC,
            batch_size));
}

void hipblasGemmStridedBatched(
    hipblasHandle_t handle,
    hipblasOperation_t opA, hipblasOperation_t opB,
    int m, int n, int k,
    std::complex<double>* alpha,
    std::complex<double>* A, int lda, std::size_t strideA,
    std::complex<double>* B, int ldb, std::size_t strideB,
    std::complex<double>* beta,
    std::complex<double>* C, int ldc, std::size_t strideC,
    int batch_size)
{
    CALL_HIPBLAS(
        hipblasZgemmStridedBatched(
            handle,
            opA, opB,
            m, n, k,
            (hipblasDoubleComplex*)alpha,
            (hipblasDoubleComplex*)A, lda, strideA,
            (hipblasDoubleComplex*)B, ldb, strideB,
            (hipblasDoubleComplex*)beta,
            (hipblasDoubleComplex*)C, ldc, strideC,
            batch_size));
}

void hipblasGemmStridedBatched(
    hipblasHandle_t handle,
    hipblasOperation_t opA, hipblasOperation_t opB,
    int m, int n, int k,
    int32_t* alpha, int8_t* A, int lda, std::size_t strideA,
                    int8_t* B, int ldb, std::size_t strideB,
    int32_t* beta, int32_t* C, int ldc, std::size_t strideC,
    int batch_size)
{
    CALL_HIPBLAS(
        hipblasGemmStridedBatchedEx(handle,
                                    opA, opB,
                                    m, n, k,
                                    alpha, A, HIPBLAS_R_8I,  lda, strideA,
                                           B, HIPBLAS_R_8I,  ldb, strideB,
                                    beta,  C, HIPBLAS_R_32I, ldc, strideC,
                                    batch_size,
                                    HIPBLAS_R_32I,
                                    HIPBLAS_GEMM_DEFAULT));
}

//------------------------------------------------------------------------------
template <typename Ta, typename Tc = Ta>
void time_gemm(hipblasOperation_t opA, hipblasOperation_t opB,
               bool hostA, bool hostB, bool hostC,
               int m, int n, int k,
               int lda, int ldb, int ldc,
               int batch_size, int time_span,
               bool batched, bool strided, bool testing)
{
    int num_devices;
    CALL_HIP(hipGetDeviceCount(&num_devices));
    assert(num_devices > 0);

    //---------------------------------
    // Set up streams, handles, events.
    std::vector<hipStream_t> stream(num_devices);
    std::vector<hipblasHandle_t> handle(num_devices);
    std::vector<hiprandGenerator_t> generator(num_devices);
    std::vector<hipEvent_t> start(num_devices);
    std::vector<hipEvent_t> stop(num_devices);
    std::vector<std::vector<float>> elapsed(num_devices);
    for (int device = 0; device < num_devices; ++device) {
        CALL_HIP(hipSetDevice(device));
        CALL_HIP(hipStreamCreate(&stream[device]));
        CALL_HIPBLAS(hipblasCreate(&handle[device]));
        CALL_HIPBLAS(hipblasSetStream(handle[device], stream[device]));
        CALL_HIPRAND(hiprandCreateGenerator(&generator[device],
                                            HIPRAND_RNG_PSEUDO_DEFAULT));
        CALL_HIPRAND(hiprandSetStream(generator[device], stream[device]));
        CALL_HIP(hipEventCreate(&start[device]));
        CALL_HIP(hipEventCreate(&stop[device]));
        elapsed[device].resize(batch_size);
    }

    //----------------------
    // Allocate data arrays.
    using Tb = Ta;
    std::vector<Ta*> d_A(num_devices);
    std::vector<Tb*> d_B(num_devices);
    std::vector<Tc*> d_C(num_devices);
    std::size_t lenA = (size_t)lda*(opA == HIPBLAS_OP_N ? k : m);
    std::size_t lenB = (size_t)ldb*(opB == HIPBLAS_OP_N ? n : k);
    std::size_t lenC = (size_t)ldc*n;
    std::size_t sizeA = sizeof(Ta)*lenA;
    std::size_t sizeB = sizeof(Tb)*lenB;
    std::size_t sizeC = sizeof(Tc)*lenC;
    for (int device = 0; device < num_devices; ++device) {
        CALL_HIP(hipSetDevice(device));
        if (hostA) CALL_HIP(hipHostMalloc(&d_A[device], sizeA*batch_size));
        else       CALL_HIP(hipMalloc(    &d_A[device], sizeA*batch_size));

        if (hostB) CALL_HIP(hipHostMalloc(&d_B[device], sizeB*batch_size));
        else       CALL_HIP(hipMalloc(    &d_B[device], sizeB*batch_size));

        if (hostC) CALL_HIP(hipHostMalloc(&d_C[device], sizeC*batch_size));
        else       CALL_HIP(hipMalloc(    &d_C[device], sizeC*batch_size));
    }

    //------------------------
    // Initialize data arrays.
    for (int device = 0; device < num_devices; ++device) {
        CALL_HIP(hipSetDevice(device));
        if (testing) {
            matrix_set(opA == HIPBLAS_OP_N ? m : k,
                       opA == HIPBLAS_OP_N ? k : m,
                       d_A[device], lda, static_cast<Ta>(1));
            matrix_set(opB == HIPBLAS_OP_N ? k : n,
                       opB == HIPBLAS_OP_N ? n : k,
                       d_B[device], ldb, static_cast<Tb>(1));
            matrix_set(m, n, d_C[device], ldc, static_cast<Tc>(1));
        }
        else {
            hiprandGenUni(generator[device], d_A[device], lenA*batch_size);
            hiprandGenUni(generator[device], d_B[device], lenB*batch_size);
            hiprandGenUni(generator[device], d_C[device], lenC*batch_size);
        }
    }

    //-----------------------
    // Set up pointer arrays.
    std::vector<Ta**> h_A_array(num_devices);
    std::vector<Tb**> h_B_array(num_devices);
    std::vector<Tc**> h_C_array(num_devices);
    std::vector<Ta**> d_A_array(num_devices);
    std::vector<Tb**> d_B_array(num_devices);
    std::vector<Tc**> d_C_array(num_devices);
    for (int device = 0; device < num_devices; ++device) {
        CALL_HIP(hipSetDevice(device));
        if (batched) {
            h_A_array[device] = (Ta**)malloc(sizeof(Ta*)*batch_size);
            h_B_array[device] = (Tb**)malloc(sizeof(Tb*)*batch_size);
            h_C_array[device] = (Tc**)malloc(sizeof(Tc*)*batch_size);
            assert(h_A_array[device] != nullptr);
            assert(h_B_array[device] != nullptr);
            assert(h_C_array[device] != nullptr);
            for (int i = 0; i < batch_size; ++i) {
                h_A_array[device][i] = d_A[device] + lenA*i;
                h_B_array[device][i] = d_B[device] + lenB*i;
                h_C_array[device][i] = d_C[device] + lenC*i;
            }
            CALL_HIP(hipMalloc(&d_A_array[device], sizeof(Ta*)*batch_size));
            CALL_HIP(hipMalloc(&d_B_array[device], sizeof(Tb*)*batch_size));
            CALL_HIP(hipMalloc(&d_C_array[device], sizeof(Tc*)*batch_size));

            CALL_HIP(hipMemcpy(d_A_array[device], h_A_array[device],
                               sizeof(Ta*)*batch_size, hipMemcpyHostToDevice));

            CALL_HIP(hipMemcpy(d_B_array[device], h_B_array[device],
                               sizeof(Tb*)*batch_size, hipMemcpyHostToDevice));

            CALL_HIP(hipMemcpy(d_C_array[device], h_C_array[device],
                               sizeof(Tc*)*batch_size, hipMemcpyHostToDevice));
        }
    }

    Tc alpha;
    Tc beta;
    if (testing) {
        alpha = static_cast<Tc>(1);
        beta = static_cast<Tc>(1);
    }
    else {
        alpha = static_cast<Tc>(2.71828);
        beta = static_cast<Tc>(3.14159);
    }

    double operations = 2.0*m*n*k;
    if (std::is_same<Tc, std::complex<float>>::value ||
        std::is_same<Tc, std::complex<double>>::value  ) {
        operations *= 4;
    }

    for (int device = 0; device < num_devices; ++device)
        printf(" device_%d_[GFLOPS]", device);
    printf(" timestamp_[sec]\n");

    int count = 0;
    double timestamp;
    auto beginning = std::chrono::high_resolution_clock::now();
    do {
        if (batched) {
            //-------------
            // batched call
            for (int device = 0; device < num_devices; ++device) {
                CALL_HIP(hipSetDevice(device));
                CALL_HIP(hipEventRecord(start[device], stream[device]));
                if (!strided) {
                    // pointer arrays
                    hipblasGemmBatched(handle[device],
                                       opA, opB,
                                       m, n, k,
                                       &alpha, d_A_array[device], lda,
                                               d_B_array[device], ldb,
                                       &beta,  d_C_array[device], ldc,
                                       batch_size);
                }
                else {
                    // strided
                    hipblasGemmStridedBatched(handle[device],
                                              opA, opB,
                                              m, n, k,
                                              &alpha, d_A[device], lda, lenA,
                                                      d_B[device], ldb, lenB,
                                              &beta,  d_C[device], ldc, lenC,
                                              batch_size);
                }
                CALL_HIP(hipEventRecord(stop[device], stream[device]));
            }
            for (int device = 0; device < num_devices; ++device) {
                CALL_HIP(hipSetDevice(device));
                CALL_HIP(hipEventSynchronize(stop[device]));
                CALL_HIP(hipEventElapsedTime(&elapsed[device][0],
                                             start[device],
                                             stop[device]));
            }

            for (int device = 0; device < num_devices; ++device) {
                double miliseconds = elapsed[device][0];
                double time_in_sec = miliseconds/1e3;
                double gflops_perf = operations*batch_size/time_in_sec/1e9;
                if (count > 0)
                    printf("%18.2lf", gflops_perf);
            }
        }
        else {
            //-------------------------
            // standard calls in a loop
            for (int i = 0; i < batch_size; ++i) {
                for (int device = 0; device < num_devices; ++device) {
                    CALL_HIP(hipSetDevice(device));
                    CALL_HIP(hipEventRecord(start[device], stream[device]));
                    hipblasGemm(handle[device],
                                opA, opB,
                                m, n, k,
                                &alpha, d_A[device] + lenA*i, lda,
                                        d_B[device] + lenB*i, ldb,
                                &beta,  d_C[device] + lenC*i, ldc);
                    CALL_HIP(hipEventRecord(stop[device], stream[device]));
                }
                for (int device = 0; device < num_devices; ++device) {
                    CALL_HIP(hipSetDevice(device));
                    CALL_HIP(hipEventSynchronize(stop[device]));
                    CALL_HIP(hipEventElapsedTime(&elapsed[device][i],
                                                 start[device],
                                                 stop[device]));
                }
            }
            for (int device = 0; device < num_devices; ++device) {
                CALL_HIP(hipSetDevice(device));
                if (count == 0 && testing)
                    matrix_assert(m, n, d_C[device], ldc, static_cast<Tc>(k+1));

                std::sort(elapsed[device].begin(),
                          elapsed[device].end(),
                          std::greater<float>());
                double median_time = elapsed[device][batch_size/2];
                double time_in_sec = median_time/1e3;
                double gflops_perf = operations/time_in_sec/1e9;
                if (count > 0)
                    printf("%18.2lf", gflops_perf);
            }
        }

        timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now()-beginning).count();
        if (count > 0)
            printf("%16.2lf\n", timestamp/1e6);
        ++count;
    }
    while (timestamp/1e6 <= time_span || count < 2);

    //--------
    // cleanup
    for (int device = 0; device < num_devices; ++device) {
        CALL_HIP(hipEventDestroy(start[device]));
        CALL_HIP(hipEventDestroy(stop[device]));
        CALL_HIPRAND(hiprandDestroyGenerator(generator[device]));
        CALL_HIPBLAS(hipblasDestroy(handle[device]));
        CALL_HIP(hipStreamDestroy(stream[device]));
        CALL_HIP(hipFree(d_A[device]));
        CALL_HIP(hipFree(d_B[device]));
        CALL_HIP(hipFree(d_C[device]));
    }
}

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    assert(argc >= 12);
    char prec = argv[1][0];
    assert(prec == 'S' ||
           prec == 'D' ||
           prec == 'C' ||
           prec == 'Z' ||
           prec == 'I'   );

    hipblasOperation_t opA;
    switch (argv[2][0]) {
        case 'N': opA = HIPBLAS_OP_N; break;
        case 'T': opA = HIPBLAS_OP_T; break;
        case 'C': opA = HIPBLAS_OP_C; break;
        default : assert(false);
    }
    hipblasOperation_t opB;
    switch (argv[2][1]) {
        case 'N': opB = HIPBLAS_OP_N; break;
        case 'T': opB = HIPBLAS_OP_T; break;
        case 'C': opB = HIPBLAS_OP_C; break;
        default : assert(false);
    }

    bool hostA = false;
    switch (argv[3][0]) {
        case 'H': hostA = true; break;
        case 'D': hostA = false; break;
        default : assert(false);
    }
    bool hostB = false;
    switch (argv[3][1]) {
        case 'H': hostB = true; break;
        case 'D': hostB = false; break;
        default : assert(false);
    }
    bool hostC = false;
    switch (argv[3][2]) {
        case 'H': hostC = true; break;
        case 'D': hostC = false; break;
        default : assert(false);
    }

    int m = std::atoi(argv[4]);
    int n = std::atoi(argv[5]);
    int k = std::atoi(argv[6]);
    assert(m > 0 && n > 0 && k > 0);

    int lda = std::atoi(argv[7]);
    int ldb = std::atoi(argv[8]);
    int ldc = std::atoi(argv[9]);
    assert(lda >= (opA == HIPBLAS_OP_N ? m : k));
    assert(ldb >= (opB == HIPBLAS_OP_N ? k : n));
    assert(ldc >= m);

    int batch_size = std::atoi(argv[10]);
    int time_span = std::atoi(argv[11]);
    assert(batch_size > 0);

    bool batched = false;
    bool strided = false;
    bool testing = false;
    int arg = 12;
    while (arg < argc) {
        std::string str(argv[arg]);
        if (str == "batched")
            batched = true;
        if (str == "strided")
            strided = true;
        if (str == "testing")
            testing = true;
        ++arg;
    }
    if (strided)
        assert(batched);

    if (prec == 'S') {
        time_gemm<float>(
            opA, opB, hostA, hostB, hostC, m, n, k, lda, ldb, ldc,
            batch_size, time_span, batched, strided, testing);
    }
    else if (prec == 'D') {
        time_gemm<double>(
            opA, opB, hostA, hostB, hostC, m, n, k, lda, ldb, ldc,
            batch_size, time_span, batched, strided, testing);
    }
    else if (prec == 'C') {
        time_gemm<std::complex<float>>(
            opA, opB, hostA, hostB, hostC, m, n, k, lda, ldb, ldc,
            batch_size, time_span, batched, strided, testing);
    }
    else if (prec == 'Z') {
        time_gemm<std::complex<double>>(
            opA, opB, hostA, hostB, hostC, m, n, k, lda, ldb, ldc,
            batch_size, time_span, batched, strided, testing);
    }
    else if (prec == 'I') {
        time_gemm<int8_t, int32_t>(
            opA, opB, hostA, hostB, hostC, m, n, k, lda, ldb, ldc,
            batch_size, time_span, batched, strided, testing);
    }

    return (EXIT_SUCCESS);
}
