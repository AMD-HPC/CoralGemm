
SRC = $(wildcard *.cpp)
OBJ = $(SRC:.cpp=.o)
EXE = gemm

ROCM_PATH ?= /opt/rocm
HIPCC      = ${ROCM_PATH}/bin/hipcc

HIP_INC  = -I${ROCM_PATH}/include
HIP_INC += -I${ROCM_PATH}/include/hiprand
ROC_INC  = -I${ROCM_PATH}/include/rocrand

HIP_LIB  = -L${ROCM_PATH}/lib
HIP_LIBS = -lhipblas -lhiprand
ROC_LIBS = -lrocblas -lrocrand

CXXFLAGS      = -O3 -std=c++11
AMDGPU_TARGET = $(shell ${ROCM_PATH}/bin/rocminfo | grep -m 1 -o -P 'gfx.{0,4}')
ROC_FLAGS     = --amdgpu-target=${AMDGPU_TARGET}

# CUDA_PATH ?= /usr/local/cuda
# NVCC       = ${CUDA_PATH}/bin/nvcc
# NV_LIBS    = -lcublas -lcurand
# NV_FLAGS   = -x cu --expt-relaxed-constexpr

INC = $(HIP_INC) $(ROC_INC)
LIB = $(HIP_LIB) $(HIP_LIBS) $(ROC_LIBS)

rocm: $(OBJ)
	$(HIPCC) $(CXXFLAGS) $(ROC_FLAGS) $(INC) $(LIB) $(OBJ) -o $(EXE)

.cpp.o:
	$(HIPCC) $(CXXFLAGS) $(ROC_FLAGS) $(INC) -c $< -o $@

# hip_cuda:
# 	$(NVCC) -D__HIP_PLATFORM__=NVCC \
# 			$(HIP_INC) \
# 			$(HIP_LIB) $(HIP_LIBS) $(NV_LIBS) \
# 			$(CXXFLAGS) $(NV_FLAGS) $(SRC) -o $(EXE)

# cuda:
# 	$(NVCC) $(NV_LIBS) \
# 			$(CXXFLAGS) $(NV_FLAGS) $(SRC) -o $(EXE)

clean:
	rm -rf $(OBJ) $(EXE)
