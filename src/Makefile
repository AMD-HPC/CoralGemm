
ROCM_PATH ?= /opt/rocm
CUDA_PATH ?= /usr/local/cuda
HIPCC      = ${ROCM_PATH}/bin/hipcc
NVCC       = ${CUDA_PATH}/bin/nvcc
HIP_INC    = -I${ROCM_PATH}/include -I${ROCM_PATH}/include/hiprand
ROC_INC    = -I${ROCM_PATH}/include/rocrand
HIP_LIB    = -L${ROCM_PATH}/lib
HIP_LIBS   = -lhipblas -lhiprand
ROC_LIBS   = -lrocblas -lrocrand
NV_LIBS    = -lcublas -lcurand
CXXFLAGS   = -O3 -std=c++11
ROC_FLAGS  = --amdgpu-target=gfx906,gfx908
NV_FLAGS   = -x cu --expt-relaxed-constexpr
SRC        = gemm.cpp
EXE        = gemm

rocm:
	$(HIPCC) $(HIP_INC) $(ROC_INC) \
			 $(HIP_LIB) $(HIP_LIBS) $(ROC_LIBS) \
			 $(CXXFLAGS) $(ROC_FLAGS) $(SRC) -o $(EXE)

hip_cuda:
	$(NVCC) -D__HIP_PLATFORM__=NVCC \
			$(HIP_INC) \
			$(HIP_LIB) $(HIP_LIBS) $(NV_LIBS) \
			$(CXXFLAGS) $(NV_FLAGS) $(SRC) -o $(EXE)

cuda:
	$(NVCC) $(NV_LIBS) \
			$(CXXFLAGS) $(NV_FLAGS) $(SRC) -o $(EXE)

clean:
	rm -rf $(EXE)
