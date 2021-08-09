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
* `make`

## Running DGEMM

* 16 GB devices (Radeon VII): `./gemm R_64F R_64F R_64F R_64F OP_N OP_T 8640 8640 8640 8640 8640 8640 9 300`

* 32 GB devices (MI60, MI100): `./gemm R_64F R_64F R_64F R_64F OP_N OP_T 8640 8640 8640 8640 8640 8640 18 300`

## Running SGEMM

* 16 GB devices (Radeon VII): `./gemm R_32F R_32F R_32F R_32F OP_N OP_T 8640 8640 8640 8640 8640 8640 18 300`

* 32 GB devices (MI60, MI100): `./gemm R_32F R_32F R_32F R_32F OP_N OP_T 8640 8640 8640 8640 8640 8640 36 300`

## Command-Line Details

```
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
           [zeroBeta]   set beta to zero
           [testing]    perform a basic sanity check
           [times]      print time in microseconds in addition to GFLOPS 
```

### Supported Precisions:

* `R_32F`: float
* `R_64F`: double
* `C_32F`: float complex
* `C_64F`: float double
* `R_8I`:  8-bit int
* `R_32I`: 32-bit int

### Supported Ops:

* `OP_N`: non-transposed
* `OP_T`: transposed
* `OP_C`: conjugate-transposed

## Details

* benchmarks `hipblas?gemm[Batched|StridedBatched][Ex]`
* allocates `BATCH_SIZE` number of matrices A, B, and C
* initializes with hipRAND (random uniform, 0.0 to 1.0)
* calls hipBLAS and collects execution times using `std::chrono`
* sets `alpha` to 2.71828 and `beta` to 3.14159
* for `hipblas?gemm[Ex]` launches a sequence of calls and takes the median time
* for `hipblas?gemm[Strided]Batched[Ex]` launches one call and takes the overall time
* reports the corresponding GFLOPS
* repeats until `TIME_SPAN` exceeded
* executes simulteneously on all devices

If `testing` is set, a primitive sanity test is ran.
Entries of A, B, and C are set to 1, and so are the factors `alpha` and `beta`.
Then, after GEMM is ran, all entries of C are checked to contain k+1.
Note that performance is usually much higher when using integer initialization
then when using random data.

## Help

Jakub Kurzak (<jakurzak@amd.com>)

[ROCm]: https://github.com/RadeonOpenCompute/ROCm
[rocBLAS]: https://github.com/ROCmSoftwarePlatform/rocBLAS
[hipBLAS]: https://github.com/ROCmSoftwarePlatform/hipBLAS
[rocRAND]: https://github.com/ROCmSoftwarePlatform/rocRAND
