.PHONY: all clean

CUDA_BASE:=$(shell scram tool tag cuda CUDA_BASE)
CUB_BASE:=$(shell scram tool tag cub CUB_BASE)

NVCC=nvcc
NVCC_FLAGS=-std=c++14 -O2 --generate-code arch=compute_75,code=sm_75 -I$(CUB_BASE)/include

all: empty grid n_array tri_matrix

clean:
	rm -f empty grid n_array tri_matrix *.o

main.o: main.cc
	g++ -c -std=c++17 -I$(CUDA_BASE)/include -I$(CUB_BASE)/include -g -O2 main.cc -o main.o

empty: empty.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

tri_matrix.o: tri_matrix.cu
	nvcc -c -std=c++14 -I$(CUDA_BASE)/include -I$(CUB_BASE)/include -g -O2 -arch=sm_75 tri_matrix.cu -o $@

tri_matrix: main.o tri_matrix.o
	g++ -std=c++17 -L$(CUDA_BASE)/lib64 -g -O2 $^ -lcudart -lcuda -o $@

grid.o: grid.cu
	nvcc -c -std=c++14 -I$(CUDA_BASE)/include -I$(CUB_BASE)/include -g -O2 -arch=sm_75 grid.cu -o $@

grid: main.o grid.o
	g++ -std=c++17 -L$(CUDA_BASE)/lib64 -g -O2 $^ -lcudart -lcuda -o $@

n_array: n_array.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@
