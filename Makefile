.PHONY: all clean

CUDA_BASE:=/usr/local/cuda-10.1
CUDA_INCLUDE_PATH=$(CUDA_BASE)/include
CUDA_LIBRARY_PATH=$(CUDA_BASE)/lib64

CUB_BASE:=$(CUDA_BASE)/targets/x86_64-linux/include/thrust/system/cuda/detail
CUB_INCLUDE_PATH=$(CUB_BASE)

CXX=g++-8
CXX_FLAGS=-std=c++17 -O2 -g -I$(CUDA_INCLUDE_PATH) -I$(CUB_INCLUDE_PATH)
LD_FLAGS=-L$(CUDA_LIBRARY_PATH) -lcudart -lcuda

NVCC=$(CUDA_BASE)/bin/nvcc -ccbin $(CXX)
NVCC_FLAGS=-std=c++14 -O2 -g --generate-code arch=compute_50,code=sm_50 -I$(CUB_INCLUDE_PATH)


all: empty grid n_array tri_matrix

clean:
	rm -f empty grid n_array tri_matrix *.o

main.o: main.cc
	$(CXX) $(CXX_FLAGS) -c $< -o $@

empty: empty.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

tri_matrix.o: tri_matrix.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

tri_matrix: main.o tri_matrix.o
	$(CXX) $(CXX_FLAGS) $^ $(LD_FLAGS) -o $@

grid.o: grid.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

grid: main.o grid.o
	$(CXX) $(CXX_FLAGS) $^ $(LD_FLAGS) -o $@

n_array: n_array.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@
