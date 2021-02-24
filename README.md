```
 ______                    __ _______
|      |.-----.----.---.-.|  |     __|.-----.--------.--------.
|   ---||  _  |   _|  _  ||  |    |  ||  -__|        |        |
|______||_____|__| |___._||__|_______||_____|__|__|__|__|__|__|
```
# Matrix Multiply Stress Test

## Prerequisites

* [ROCm][]
* [rocBLAS][]
* [hipBLAS][]
* [rocRAND][]
* hipRAND

`sudo apt install rocm-dkms rocm-libs` to install all prerequisites.

## Installing

* `cd src`
* `make rocm` to build with HIP (ROCm target),
* `make hip_cuda` to build with HIP (CUDA target),
* `make cuda` to build with CUDA (HIP not required).

## Running DGEMM

* 16 GB devices (Radeon VII): `./gemm D NT DDD 8640 8640 8640 8640 8640 8640 9 300`

* 32 GB devices (MI60, MI100): `./gemm D NT DDD 8640 8640 8640 8640 8640 8640 18 300`

* 64 GB devices (one MI200 GCD): `./gemm D NT DDD 8640 8640 8640 8640 8640 8640 36 300`

## Running SGEMM

* 16 GB devices (Radeon VII): `./gemm S NT DDD 8640 8640 8640 8640 8640 8640 18 300`

* 32 GB devices (MI60, MI100): `./gemm S NT DDD 8640 8640 8640 8640 8640 8640 36 300`

* 64 GB devices (one MI200 GCD): `./gemm S NT DDD 8640 8640 8640 8640 8640 8640 72 300`

## Command-Line Details

```
./gemm <S|C|D|Z|I>        precision
       <NN|NT|TN|...>     transposition of A and B
       <DDD|DDH|DHH|...>  location of A, B, and C (host or device)
       <M> <N> <K>        dimensions
       <LDA> <LDB> <LDC>  leading dimensions
       <BATCH_SIZE>       number of matrices to sweep
       <TIME_SPAN>        runtime duration in seconds
       [batched]          call the batched routine
       [batched strided]  call the strided batched routine
       [testing]          perform a basic sanity check
```

* benchmarks `hipblas?gemm[Batched|StridedBatched]`
* allocates `BATCH_SIZE` number of matrices A, B, and C
* initializes with hipRAND (random uniform, 0.0 to 1.0)
* calls hipBLAS and collects execution times using `std::chrono`
* sets *alpha* to 2.71828 and *beta* to *3.14159*
* for `hipblas?gemm` launches a sequence of calls and takes the median time
* for `hipblas?gemm[Strided]Batched` launches one call and takes the overall time
* reports the corresponding GFLOPS
* repeats until `TIME_SPAN` exceeded

If `testing` is set, a primitive sanity test is ran.
Entries of A, B, and C matrices are set to 1,
*alpha* and *beta* factors are set to 1,
and then, after calling GEMM,
all entries of the C matrix are checked to contain k+1.
Only the first A, B, and C in the batch are used.
**Performance obtained when testing does not reflect performance of GEMM with random data.**
**Either time or test, but not both.**

## Help

Jakub Kurzak (<jakurzak@amd.com>)

[ROCm]: https://github.com/RadeonOpenCompute/ROCm
[rocBLAS]: https://github.com/ROCmSoftwarePlatform/rocBLAS
[hipBLAS]: https://github.com/ROCmSoftwarePlatform/hipBLAS
[rocRAND]: https://github.com/ROCmSoftwarePlatform/rocRAND
