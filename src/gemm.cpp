
#include "DeviceBatchArray.h"
#include "DeviceBatchedGemm.h"

#include <iostream>

//------------------------------------------------------------------------------
/*
    ./gemm PRECISION_A
           PRECISION_B
           PRECISION_C
           COMPUTE_PRECISION
           OP_A
           OP_B
           M
           N
           K
           LDA
           LDB
           LDC
           BATCH_COUNT
           TIME_SPAN    runtime duration in seconds
           [batched]    run batched GEMM
           [strided]    run strided batched GEMM
           [ex]         use the Ex API
           [hostA]      A in host memory
           [hostB]      B in host memory
           [hostC]      C in host memory
           [coherentA]  if in host memory, A is coherent (not cached)
           [coherentB]  if in host memory, B is coherent (not cached)
           [coherentC]  if in host memory, C is coherent (not cached)
           [sharedA]    one A for all devices
           [sharedB]    one B for all devices

    supported precisions:
        - R_32F: float
        - R_64F: double
        - C_32F: float complex
        - C_64F: float double
        - R_8I:  8-bit int
        - R_32I: 32-bit int

    supported ops:
        - OP_N: non-transposed
        - OP_T: transposed
        - OP_C: conjugate-transposed
*/

//------------------------------------------------------------------------------
void run (int argc, char** argv)
{
    ASSERT(argc >= 15);

    void* alpha;
    void* beta;

    float                alpha_r_32f = 2.71828;
    double               alpha_r_64f = 2.71828;
    std::complex<float>  alpha_c_32f = 2.71828;
    std::complex<double> alpha_c_64f = 2.71828;
    int32_t              alpha_r_32i = 2;

    float                beta_r_32f = 3.14159;
    double               beta_r_64f = 3.14159;
    std::complex<float>  beta_c_32f = 3.14159;
    std::complex<double> beta_c_64f = 3.14159;
    int32_t              beta_r_32i = 3;

    std::string type_c(argv[3]);
    if      (type_c == "R_32F") alpha = &alpha_r_32f;
    else if (type_c == "R_64F") alpha = &alpha_r_64f;
    else if (type_c == "C_32F") alpha = &alpha_c_32f;
    else if (type_c == "C_64F") alpha = &alpha_c_64f;
    else if (type_c == "R_32I") alpha = &alpha_r_32i;

    if      (type_c == "R_32F") beta = &beta_r_32f;
    else if (type_c == "R_64F") beta = &beta_r_64f;
    else if (type_c == "C_32F") beta = &beta_c_32f;
    else if (type_c == "C_64F") beta = &beta_c_64f;
    else if (type_c == "R_32I") beta = &beta_r_32i;

    bool batched = false;
    bool strided = false;
    bool ex = false;
    bool host_a = false;
    bool host_b = false;
    bool host_c = false;
    bool coherent_a = false;
    bool coherent_b = false;
    bool coherent_c = false;
    bool shared_a = false;
    bool shared_b = false;

    int arg = 15;
    while (arg < argc) {
        std::string str(argv[arg]);
        if (str == "batched")   batched = true;
        if (str == "strided")   strided = true;
        if (str == "ex")        ex = true;
        if (str == "hostA")     host_a = true;
        if (str == "hostB")     host_b = true;
        if (str == "hostC")     host_c = true;
        if (str == "coherentA") coherent_a = true;
        if (str == "coherentB") coherent_b = true;
        if (str == "coherentC") coherent_c = true;
        if (str == "sharedA")   shared_a = true;
        if (str == "sharedB")   shared_b = true;
        ++arg;
    }

    std::vector<BatchedGemm*> dev_gemms;
    BatchedGemm::makeDevices(std::string(argv[1]), // type a
                             std::string(argv[2]), // type b
                             std::string(argv[3]), // type c
                             std::string(argv[4]), // compute type
                             std::string(argv[5]), // op a
                             std::string(argv[6]), // op b
                             std::atoi(argv[7]),   // m
                             std::atoi(argv[8]),   // n
                             std::atoi(argv[9]),   // k
                             std::atoi(argv[10]),  // lda
                             std::atoi(argv[11]),  // ldb
                             std::atoi(argv[12]),  // ldc
                             std::atoi(argv[13]),  // batch count
                             alpha, beta,
                             host_a, host_b, host_c,
                             coherent_a, coherent_b, coherent_c,
                             shared_a, shared_b,
                             dev_gemms);

    double time_span = std::atoi(argv[14]);

    // Assign the mode.
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

    // Initialize with random numbers.
    for (int dev = 0; dev < dev_gemms.size(); ++dev) {
//      dev_gemms[dev]->generateConstant(1.0);
        dev_gemms[dev]->generateUniform();
    }

    // Print column labels.
    for (int dev = 0; dev < dev_gemms.size(); ++dev)
        printf(" device_%d_[GFLOPS]", dev);
    printf(" timestamp_[sec]\n");

    int count = 0;
    double timestamp;
    auto beginning = std::chrono::high_resolution_clock::now();
    do {
        // Run on all devices.
        for (int dev = 0; dev < dev_gemms.size(); ++dev) {
            dev_gemms[dev]->run(mode);
        }

        // Report GFLOPS.
        for (int dev = 0; dev < dev_gemms.size(); ++dev) {
            double gflops;
            gflops = dev_gemms[dev]->getGflops(mode);
            if (count > 0)
                printf("%18.2lf", gflops);
        }

        // Collect the timestamp.
        // Print if not the first iteration.
        timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now()-beginning).count();
        if (count > 0)
            printf("%16.2lf\n", timestamp/1e6);

        ++count;
    }
    // Loop until time_span reached.
    while (timestamp/1e6 <= time_span || count < 2);
}

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    try {
        run(argc, argv);
    }
    catch (Exception& e) {
        std::cerr << std::endl << e.what() << std::endl << std::endl;
        exit(EXIT_FAILURE);
    }
    exit(EXIT_SUCCESS);
}
