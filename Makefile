.PHONY: all clean

CUDA_BASE:=$(shell scram tool tag cuda CUDA_BASE)
CUB_BASE:=$(shell scram tool tag cub CUB_BASE)

NVCC=nvcc
NVCC_FLAGS=-std=c++14 -O2 --generate-code arch=compute_75,code=sm_75 -I$(CUB_BASE)/include

all: empty grid n_array tri_matrix

main.o: main.cc
	g++ -c -std=c++17 -I$(CUDA_BASE)/include -I$(CUB_BASE)/include -g -O2 main.cc -o main.o

empty: empty.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

grid.o: grid.cu
	nvcc -c -std=c++14 -g -O2 -arch=sm_75 grid.cu -o grid.o

grid: grid.o main.o
	g++ -std=c++17 -L$(CUDA_BASE)/lib64 -g -O2 main.o grid.o -lcudart -lcuda -o grid

n_array: n_array.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

tri_matrix: tri_matrix.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@
