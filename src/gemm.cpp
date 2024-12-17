//------------------------------------------------------------------------------
/// \file
/// \brief      main CoralGemm driver routines
/// \date       2020-2024
/// \author     Jakub Kurzak
/// \copyright  Advanced Micro Devices, Inc.
///
#include "CommandLine.h"
#include "DeviceBatchArray.h"
#include "DeviceBatchedGemm.h"
#include "TypeConstant.h"

#include <chrono>
#include <iostream>
#include <unistd.h>

//------------------------------------------------------------------------------
/// \brief
///     Scans the command line.
///     Creates BatchedGemm objects for all devices in the system.
///     Initializes the matrices with random numbers (uniform distribution).
///     Loops over the matrix multiplication routines until duration reached.
///     Prints GFLOPS numbers for all devices, followed by the timestamp.
///     Skips printing of the first row of results (considered a warmup call).
///
/// \param[in] argc, argv
///     command line arguments
///
void run(int argc, char** argv)
{
    ASSERT(argc >= 15, "Invalid command line.");
    CommandLine cmd(argc, argv);
    cmd.check(
        1, 4,
        std::regex(R"(^(?:R_8B|R_8F|R_16B|R_16F|R_32F|)"
                   R"(R_64F|C_32F|C_64F|R_8I|R_32I)$)"));
    cmd.check(
        {5, 6},
        std::regex(R"(^(?:|OP_N|OP_T|OP_C)$)"));
    cmd.check(
        7, 14,
        std::regex(R"(^(?:[0-9]+)$)"));
    if (argc > 15) {
        cmd.check(
            15, argc-1,
            std::regex(R"(^(?:batched|strided|ex|lt|)"
                       R"(hostA|hostB|hostC|)"
                       R"(coherentA|coherentB|sharedA|sharedB|)"
                       R"(zeroBeta|testing|times|hostname|threaded)$)"));
    }

    bool batched = false;
    bool strided = false;
    bool ex = false;
    bool lt = false;
    bool host_a = false;
    bool host_b = false;
    bool host_c = false;
    bool coherent_a = false;
    bool coherent_b = false;
    bool coherent_c = false;
    bool shared_a = false;
    bool shared_b = false;
    bool zero_beta = false;
    bool testing = false;
    bool times = false;
    bool hostname = false;
    [[maybe_unused]] bool threaded = false;

    int arg = 15;
    while (arg < argc) {
        std::string str(argv[arg]);
        if (str == "batched")   batched = true;
        if (str == "strided")   strided = true;
        if (str == "ex")        ex = true;
        if (str == "lt")        lt = true;
        if (str == "hostA")     host_a = true;
        if (str == "hostB")     host_b = true;
        if (str == "hostC")     host_c = true;
        if (str == "coherentA") coherent_a = true;
        if (str == "coherentB") coherent_b = true;
        if (str == "coherentC") coherent_c = true;
        if (str == "sharedA")   shared_a = true;
        if (str == "sharedB")   shared_b = true;
        if (str == "zeroBeta")  zero_beta = true;
        if (str == "testing")   testing = true;
        if (str == "times")     times = true;
        if (str == "hostname")  hostname = true;
        if (str == "threaded")  threaded = true;
        ++arg;
    }

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

    if (testing) {
        alpha_r_32f = 1.0;
        alpha_r_64f = 1.0;
        alpha_c_32f = 1.0;
        alpha_c_64f = 1.0;
        alpha_r_32i = 1;

        beta_r_32f = 1.0;
        beta_r_64f = 1.0;
        beta_c_32f = 1.0;
        beta_c_64f = 1.0;
        beta_r_32i = 1;
    }
    else {
        alpha_r_32f = 2.71828;
        alpha_r_64f = 2.71828;
        alpha_c_32f = 2.71828;
        alpha_c_64f = 2.71828;
        alpha_r_32i = 2;

        if (zero_beta) {
            beta_r_32f = 0.0;
            beta_r_64f = 0.0;
            beta_c_32f = 0.0;
            beta_c_64f = 0.0;
            beta_r_32i = 0;
        }
        else {
            beta_r_32f = 3.14159;
            beta_r_64f = 3.14159;
            beta_c_32f = 3.14159;
            beta_c_64f = 3.14159;
            beta_r_32i = 3;
        }
    }

    void* alpha;
    void* beta;

    std::string compute_type(argv[4]);
    if      (compute_type == "R_32F") alpha = &alpha_r_32f;
    else if (compute_type == "R_64F") alpha = &alpha_r_64f;
    else if (compute_type == "C_32F") alpha = &alpha_c_32f;
    else if (compute_type == "C_64F") alpha = &alpha_c_64f;
    else if (compute_type == "R_32I") alpha = &alpha_r_32i;

    if      (compute_type == "R_32F") beta = &beta_r_32f;
    else if (compute_type == "R_64F") beta = &beta_r_64f;
    else if (compute_type == "C_32F") beta = &beta_c_32f;
    else if (compute_type == "C_64F") beta = &beta_c_64f;
    else if (compute_type == "R_32I") beta = &beta_r_32i;

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
                     mode = BatchedGemm::Mode::Standard;
        if (batched) mode = BatchedGemm::Mode::Batched;
        if (strided) mode = BatchedGemm::Mode::StridedBatched;
    if (ex) {
                     mode = BatchedGemm::Mode::StandardEx;
        if (batched) mode = BatchedGemm::Mode::BatchedEx;
        if (strided) mode = BatchedGemm::Mode::StridedBatchedEx;
    }
    if (lt) {
                     mode = BatchedGemm::Mode::StandardLt;
        if (batched) mode = BatchedGemm::Mode::BatchedLt;
    }

    // Initialize with a constant or random numbers.
    for (int dev = 0; dev < dev_gemms.size(); ++dev) {
        if (testing)
            dev_gemms[dev]->generateConstant(1.0);
        else
            dev_gemms[dev]->generateUniform();
    }

    // Print column labels.
    if (time_span > 0.0) {
        for (int dev = 0; dev < dev_gemms.size(); ++dev)
            printf(" device_%d_[GFLOPS]", dev);
        printf(" timestamp_[sec]");
        if (times)
            for (int dev = 0; dev < dev_gemms.size(); ++dev)
                printf(" device_%d_[us]", dev);
        printf("\n");
    }

    int count = 0;
    double timestamp;
    auto beginning = std::chrono::high_resolution_clock::now();
    do {
        // Run on all devices.
        #pragma omp parallel for num_threads(dev_gemms.size()) if(threaded)
        for (int dev = 0; dev < dev_gemms.size(); ++dev) {
            dev_gemms[dev]->run(mode);
        }

        // Test the first pass.
        if (count == 0 && testing) {
            for (int dev = 0; dev < dev_gemms.size(); ++dev) {
                dev_gemms[dev]->validateConstant(std::atoi(argv[9])+1);
            }
        }

        // Collect GFLOPS numbers and execution times.
        // Print GFLOPS numbers.
        std::vector<double> gflops(dev_gemms.size());
        std::vector<double> time_in_sec(dev_gemms.size());
        for (int dev = 0; dev < dev_gemms.size(); ++dev) {
            auto retval = dev_gemms[dev]->getGflops(mode);
            gflops[dev] = retval.first;
            time_in_sec[dev] = retval.second;
            if (count > 0) {
                printf("%18.2lf", gflops[dev]);
            }
        }

        // Collect the timestamp.
        // Print if not the first iteration.
        timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now()-beginning).count();
        if (count > 0)
            printf("%16.2lf", timestamp/1e6);

        // Print execution times.
        if (times) {
            for (int dev = 0; dev < dev_gemms.size(); ++dev) {
                if (count > 0) {
                    printf("%14.0lf", time_in_sec[dev]*1e6);
                }
            }
        }

        if (count > 0 && hostname) {
            int const max_host_name = 64;
            char hostname[max_host_name];
            gethostname(hostname, max_host_name);
            printf("\t%s", hostname);
        }

        if (count > 0)
            printf("\n");

        ++count;
    }
    // Loop until time_span reached.
    while (timestamp/1e6 <= time_span || count < 2);
}

//------------------------------------------------------------------------------
/// \brief
///     Launches the run inside a `try` block.
///     Caches and reports exceptions.
///
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
