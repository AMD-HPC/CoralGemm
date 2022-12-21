
SRC  = $(wildcard *.cpp)
OBJ = $(SRC:.cpp=.o)
EXE = gemm

ROCM_PATH ?= /opt/rocm
HIPCC      = ${ROCM_PATH}/bin/hipcc

HIP_INC  = -I${ROCM_PATH}/include
HIP_INC += -I${ROCM_PATH}/include/hiprand
HIP_INC += -I${ROCM_PATH}/include/rocrand

HIP_LIB  = -L${ROCM_PATH}/lib
HIP_LIB += -lhipblas -lhiprand
HIP_LIB += -lrocblas -lrocrand
HIP_LIB += -lamdhip64 -fopenmp

OFFLOAD_ARCH ?= $(shell ${ROCM_PATH}/bin/rocminfo | grep -m 1 -o -P 'gfx.{0,4}')
OFFLOAD_ARCH := $(if $(OFFLOAD_ARCH),$(OFFLOAD_ARCH),gfx906,gfx908,gfx90a)

HIPCC_FLAGS  = -fPIC
HIPCC_FLAGS += --offload-arch=${OFFLOAD_ARCH}

CXXFLAGS  = -O3 -std=c++11 -fopenmp
CXXFLAGS += -D__HIP_PLATFORM_AMD__
CXXFLAGS += -D__HIP_PLATFORM_HCC__

CUDA_PATH ?= /usr/local/cuda
NVCC       = ${CUDA_PATH}/bin/nvcc

CUDA_LIB   = -lcublas -lcurand
CUDA_FLAGS = -x cu --expt-relaxed-constexpr

rocm: $(OBJ)
	$(CXX) $(OBJ) $(HIP_LIB) -o $(EXE)

%.hip.o: %.hip.cpp
	$(HIPCC) $(CXXFLAGS) $(HIPCC_FLAGS) $(HIP_INC) -c $< -o $@

.cpp.o:
	$(CXX) $(CXXFLAGS) $(HIP_INC) -c $< -o $@

cuda:
	$(NVCC) $(CUDA_FLAGS) $(SRC) $(CUDA_LIB) -o $(EXE)

clean:
	rm -rf $(OBJ) $(EXE) core.*
