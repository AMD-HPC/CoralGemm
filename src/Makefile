
SRC = $(wildcard *.cpp)
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

CXXFLAGS       = -O3 -std=c++11
AMDGPU_TARGET ?= $(shell ${ROCM_PATH}/bin/rocminfo | grep -m 1 -o -P 'gfx.{0,4}')
AMDGPU_TARGET := $(if $(AMDGPU_TARGET),$(AMDGPU_TARGET),gfx906,gfx908,gfx90a)
CXXFLAGS      += --amdgpu-target=${AMDGPU_TARGET}

CUDA_PATH ?= /usr/local/cuda
NVCC       = ${CUDA_PATH}/bin/nvcc

CUDA_LIB   = -lcublas -lcurand
CUDA_FLAGS = -x cu --expt-relaxed-constexpr

rocm: $(OBJ)
	$(HIPCC) $(CXXFLAGS) $(HIP_INC) $(OBJ) $(HIP_LIB) -o $(EXE)

.cpp.o:
	$(HIPCC) $(CXXFLAGS) $(HIP_INC) -c $< -o $@

cuda:
	$(NVCC) $(CUDA_FLAGS) $(SRC) $(CUDA_LIB) -o $(EXE)

clean:
	rm -rf $(OBJ) $(EXE) core.*
