//------------------------------------------------------------------------------
/// \file
/// \brief      main CoralGemm driver routines
/// \date       2020-2021
/// \author     Jakub Kurzak
/// \copyright  Advanced Micro Devices, Inc.
///
#include "../DeviceBatchArray.h"
#include "../DeviceBatchedGemm.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <unistd.h>

#include <mpi.h>

//------------------------------------------------------------------------------
/// \brief
///
std::size_t type_size(std::string type_name)
{
    std::size_t size;
    if      (type_name == "R_16B") size =  2;
    else if (type_name == "R_16F") size =  2;
    else if (type_name == "R_32F") size =  4;
    else if (type_name == "R_64F") size =  8;
    else if (type_name == "C_32F") size =  8;
    else if (type_name == "C_64F") size =  16;
    else if (type_name == "R_8I" ) size =  1;
    else if (type_name == "R_32I") size =  4;
    return size;
}

//------------------------------------------------------------------------------
/// \brief
///
void round_up(int &n, std::string type_name, int ceil_bytes)
{
    int ceil_elements = ceil_bytes/type_size(type_name);
    if (n%ceil_elements != 0)
        n = n/ceil_elements*ceil_elements + ceil_elements;
}

//------------------------------------------------------------------------------
/// \brief
///
void step(std::string type_a_name,
          std::string type_b_name,
          std::string type_c_name,
          std::string compute_type_name,
          std::string op_a_name,
          std::string op_b_name,
          BatchedGemm::Mode mode,
          int m, int n, int k,
          int lda, int ldb, int ldc,
          int batch_count,
          std::vector<int> &n_array,
          std::vector<int> &k_array,
          std::vector<double> &gflops_array)
{
    float                alpha_r_32f;
    double               alpha_r_64f;
    std::complex<float>  alpha_c_32f;
    std::complex<double> alpha_c_64f;
    int32_t              alpha_r_32i;

    float                beta_r_32f;
    double               beta_r_64f;
    std::complex<float>  beta_c_32f;
    std::complex<double> beta_c_64f;
    int32_t              beta_r_32i;

    alpha_r_32f = 2.71828;
    alpha_r_64f = 2.71828;
    alpha_c_32f = 2.71828;
    alpha_c_64f = 2.71828;
    alpha_r_32i = 2;

    beta_r_32f = 3.14159;
    beta_r_64f = 3.14159;
    beta_c_32f = 3.14159;
    beta_c_64f = 3.14159;
    beta_r_32i = 3;

    void* alpha;
    void* beta;

    if      (type_c_name == "R_32F") alpha = &alpha_r_32f;
    else if (type_c_name == "R_64F") alpha = &alpha_r_64f;
    else if (type_c_name == "C_32F") alpha = &alpha_c_32f;
    else if (type_c_name == "C_64F") alpha = &alpha_c_64f;
    else if (type_c_name == "R_32I") alpha = &alpha_r_32i;

    if      (type_c_name == "R_32F") beta = &beta_r_32f;
    else if (type_c_name == "R_64F") beta = &beta_r_64f;
    else if (type_c_name == "C_32F") beta = &beta_c_32f;
    else if (type_c_name == "C_64F") beta = &beta_c_64f;
    else if (type_c_name == "R_32I") beta = &beta_r_32i;

    std::vector<BatchedGemm*> dev_gemms;
    BatchedGemm::makeDevices(type_a_name,
                             type_b_name,
                             type_c_name,
                             compute_type_name,
                             op_a_name,
                             op_b_name,
                             m, n, k,
                             lda, ldb, ldc,
                             batch_count,
                             alpha, beta,
                             false, false, false,
                             false, false, false,
                             false, false,
                             dev_gemms);

    dev_gemms[0]->generateUniform();
    dev_gemms[0]->run(mode);
    double gflops = dev_gemms[0]->getGflops(mode).first;
    delete dev_gemms[0];

    n_array.push_back(n);
    k_array.push_back(k);
    gflops_array.push_back(gflops);
}

//------------------------------------------------------------------------------
/// \brief
///
void sweep(std::string type_a_name,
           std::string type_b_name,
           std::string type_c_name,
           std::string compute_type_name,
           std::string op_a_name,
           std::string op_b_name,
           BatchedGemm::Mode mode,
           int max_n,
           int max_k,
           std::size_t max_size,
           int max_count,
           double duration,
           std::vector<int> &n_array,
           std::vector<int> &k_array,
           std::vector<double> &gflops_array)
{
    double timestamp;
    auto beginning = std::chrono::high_resolution_clock::now();
    do {
        int n;
        int k;
        if (compute_type_name != "R_32I") {
            do { n = rand()%max_n; } while (n == 0);
            do { k = rand()%max_k; } while (k == 0);
        }
        else {
            do { n = rand()%max_n; } while (n == 0 || n%4 != 0);
            do { k = rand()%max_k; } while (k == 0 || k%4 != 0);
        }
        int m = n;

        int lda = op_a_name == "OP_N" ? m : k;
        int ldb = op_b_name == "OP_N" ? k : n;
        int ldc = m;

        round_up(lda, type_a_name, 128);
        round_up(ldb, type_b_name, 128);
        round_up(ldc, type_c_name, 128);

        std::size_t size_a = type_size(type_a_name)*lda;
        std::size_t size_b = type_size(type_b_name)*ldb;
        size_a *= op_a_name == "OP_N" ? k : m;
        size_b *= op_b_name == "OP_N" ? n : k;
        std::size_t size_c = type_size(type_c_name)*ldc*n;
        std::size_t size = size_a+size_b+size_c;

        int batch_count = max_size/size;
        batch_count = std::min(batch_count, max_count);

        if (batch_count > 0)
            step(type_a_name,
                 type_b_name,
                 type_c_name,
                 compute_type_name,
                 op_a_name,
                 op_b_name,
                 mode,
                 m, n, k,
                 lda, ldb, ldc,
                 batch_count,
                 n_array,
                 k_array,
                 gflops_array);

        timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now()-beginning).count();
    }
    while (timestamp*1e-6 <= duration);
}

//------------------------------------------------------------------------------
/// \brief
///
void job(std::string type_a_name,
         std::string type_b_name,
         std::string type_c_name,
         std::string compute_type_name,
         std::string op_a_name,
         std::string op_b_name,
         BatchedGemm::Mode mode,
         int max_n,
         int max_k,
         std::size_t max_size,
         int max_count,
         double duration)
{
    int mpi_rank;
    int mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    std::vector<int> n_array;
    std::vector<int> k_array;
    std::vector<double> gflops_array;

    srand(mpi_rank);
    sweep(type_a_name,
          type_b_name,
          type_c_name,
          compute_type_name,
          op_a_name,
          op_b_name,
          mode,
          max_n,
          max_k,
          max_size,
          max_count,
          duration,
          n_array,
          k_array,
          gflops_array);

    for (int rank = 0; rank < mpi_size; ++rank) {
        if (rank == mpi_rank) {
            for (int i = 0; i < n_array.size(); ++i) {
                printf("%d,%d,%lf\n", n_array[i], k_array[i], gflops_array[i]);
            }
        }
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

//------------------------------------------------------------------------------
/// \brief
///     Launches the run inside a `try` block.
///     Caches and reports exceptions.
///
/// make -j -f Makefile.sweep
///
/// ./gemm PRECISION_A
///        PRECISION_B
///        PRECISION_C
///        COMPUTE_PRECISION
///        OP_A
///        OP_B
///        MAX_N
///        MAX_K
///        MAX_SIZE     max memory footprint
///        MAX_COUNT    max batch size
///        DURATION     runtime duration in seconds
///        [batched]    run batched GEMM
///        [strided]    run strided batched GEMM
///        [ex]         use the Ex API
///
int main(int argc, char** argv)
{
    ASSERT(argc >= 12);

    bool batched = false;
    bool strided = false;
    bool ex = false;

    int arg = 12;
    while (arg < argc) {
        std::string str(argv[arg]);
        if (str == "batched") batched = true;
        if (str == "strided") strided = true;
        if (str == "ex")      ex = true;
        ++arg;
    }

    BatchedGemm::Mode mode;
    if (strided) {
        if (ex) mode = BatchedGemm::Mode::StridedBatchedEx;
        else    mode = BatchedGemm::Mode::StridedBatched;
    }
    else if (batched) {
        if (ex) mode = BatchedGemm::Mode::BatchedEx;
        else    mode = BatchedGemm::Mode::Batched;
    }
    else {
        if (ex) mode = BatchedGemm::Mode::StandardEx;
        else    mode = BatchedGemm::Mode::Standard;
    }

    MPI_Init(&argc, &argv);
    try {
        job(std::string(argv[1]),
            std::string(argv[2]),
            std::string(argv[3]),
            std::string(argv[4]),
            std::string(argv[5]),
            std::string(argv[6]),
            mode,
            std::atoi(argv[7]),
            std::atoi(argv[8]),
            std::atol(argv[9]),
            std::atoi(argv[10]),
            std::atoi(argv[11]));
    }
    catch (Exception& e) {
        std::cerr << std::endl << e.what() << std::endl << std::endl;
        exit(EXIT_FAILURE);
    }
    MPI_Finalize();
}
