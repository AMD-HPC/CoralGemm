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

## Building

```
git clone git@github.com:AMD-HPC/CoralGemm.git
cd CoralGemm
mkdir build
cd build
cmake ..
make -j
```

Need be, set `CMAKE_MODULE_PATH` and `CMAKE_PREFIX_PATH`, e.g.:

```
export CMAKE_MODULE_PATH=/opt/rocm/share/rocm/cmake:${CMAKE_MODULE_PATH}
export CMAKE_PREFIX_PATH=/opt/rocm/share/rocm/cmake:${CMAKE_PREFIX_PATH}
```

By default CoralGemm is built for AMD GPUs using ROCm.\
However, it can also be built for NVIDIA GPUs using CUDA.\
To do so, set `USE_HIP=OFF`, `USE_CUDA=ON`, and set `CMAKE_CUDA_ARCHITECTURES`, e.g.:

```
cmake -DUSE_HIP=OFF -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 ..
```

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
           [lt]         use hipBLASLt
           [hostA]      A in host memory
           [hostB]      B in host memory
           [hostC]      C in host memory
           [coherentA]  if in host memory, A is coherent (not cached)
           [coherentB]  if in host memory, B is coherent (not cached)
           [coherentC]  if in host memory, C is coherent (not cached)
           [sharedA]    one A for all devices
           [sharedB]    one B for all devices
           [zeroBeta]   set beta to zero
           [testing]    perform a basic sanity check (requires -DCMAKE_BUILD_TYPE=DEBUG)
           [times]      print time in microseconds in addition to GFLOPS
           [hostname]   print the hostname
           [threaded]   launch to each device from a different thread
```

When `TIME_SPAN` is set to 0, one warmup run is done, followed by one timing run, and printing of column labels is disabled.

### Supported Precisions:

* `R_8B`: FP8 E5M2
* `R_8F`: FP8 E4M3
* `R_16B`: BF16
* `R_16F`: FP16
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

If `testing` is set, a primitive sanity test is ran.\
The test uses `assert()` in device code and requires `-DCMAKE_BUILD_TYPE=DEBUG`.\
Entries of A, B, and C are set to 1, and so are the factors `alpha` and `beta`.\
Then, after GEMM is ran, all entries of C are checked to contain k+1.\
Note that performance is usually much higher when using integer initialization.

## Examples

### DGEMM/SGEMM

```
./gemm R_64F R_64F R_64F R_64F OP_N OP_T ...
./gemm R_32F R_32F R_32F R_32F OP_N OP_T ...
```

### FP16/BF16

```
./gemm R_16B R_16B R_32F R_32F OP_N OP_T ... ex
./gemm R_16F R_16F R_32F R_32F OP_N OP_T ... ex
```

### FP8 (E4M3/E5M2)

```
./gemm R_8B R_8B R_32F R_32F OP_N OP_T ... lt
./gemm R_8F R_8F R_32F R_32F OP_N OP_T ... lt
```

## Tips

* Aim for a dozen or so matrices in the batch to keep the median stable.
* Beyond that, extra memory is usually better spent on larger matrices.
* Grow the matrices and/or batch size until the GPU is effectively full.
* Choose dimensions divisible by a small power of two, but avoid exact powers of two:
  - Multiples of 64, 128, or 256 align well with tiling, wavefront sizes, and memory layout.
  - Exact powers of two (e.g. 16 384, 32 768) can trigger unfavorable strides (cache/TLB conflict misses).
  - Example: use 12 800 instead of 131 072; 25 600 instead of 262 144; 512 000 instead of 524 288.

## Help

Jakub Kurzak (<jakurzak@amd.com>)

[ROCm]: https://github.com/RadeonOpenCompute/ROCm
[rocBLAS]: https://github.com/ROCmSoftwarePlatform/rocBLAS
[hipBLAS]: https://github.com/ROCmSoftwarePlatform/hipBLAS
[rocRAND]: https://github.com/ROCmSoftwarePlatform/rocRAND
