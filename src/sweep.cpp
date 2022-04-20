//------------------------------------------------------------------------------
/// \file
/// \brief      main CoralGemm driver routines
/// \date       2020-2021
/// \author     Jakub Kurzak
/// \copyright  Advanced Micro Devices, Inc.
///
#include "DeviceBatchArray.h"
#include "DeviceBatchedGemm.h"

#include <chrono>
#include <iostream>
#include <unistd.h>

//------------------------------------------------------------------------------
/// \brief
///
void run(std::string type_a_name,
         std::string type_b_name,
         std::string type_c_name,
         std::string compute_type_name,
         std::string op_a_name,
         std::string op_b_name,
         BatchedGemm::Mode mode,
         int m, int n, int k,
         int lda, int ldb, int ldc,
         int batch_count)
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
    printf("%8d%8d%12.2lf\n", m, k, gflops);
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
           BatchedGemm::Mode mode)
{
    int m = 8640;
    int n = 8640;
    int k = 8640;
    int lda = 8640;
    int ldb = 8640;
    int ldc = 8640;
    int batch_count = 3;

    run(type_a_name,
        type_b_name,
        type_c_name,
        compute_type_name,
        op_a_name,
        op_b_name,
        mode,
        m, n, k,
        lda, ldb, ldc,
        batch_count);
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
///        TIME_SPAN    runtime duration in seconds
///        [batched]    run batched GEMM
///        [strided]    run strided batched GEMM
///        [ex]         use the Ex API
///
int main(int argc, char** argv)
{
    ASSERT(argc >= 8);

    bool batched = false;
    bool strided = false;
    bool ex = false;

    int arg = 8;
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

    try {
        sweep(std::string(argv[1]),
            std::string(argv[2]),
            std::string(argv[3]),
            std::string(argv[4]),
            std::string(argv[5]),
            std::string(argv[6]),
            mode);
    }
    catch (Exception& e) {
        std::cerr << std::endl << e.what() << std::endl << std::endl;
        exit(EXIT_FAILURE);
    }
    exit(EXIT_SUCCESS);
}
