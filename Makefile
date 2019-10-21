.PHONY: all clean

CUDA_BASE:=/usr/local/cuda-10.1
CUDA_INCLUDE_PATH=$(CUDA_BASE)/include
CUDA_LIBRARY_PATH=$(CUDA_BASE)/lib64

CUB_BASE:=$(CUDA_BASE)/targets/x86_64-linux/include/thrust/system/cuda/detail
CUB_INCLUDE_PATH=$(CUB_BASE)

FASTJET_BASE:=

CXX=g++-8
CXX_FLAGS=-std=c++17 -O2 -g -I$(CUDA_INCLUDE_PATH) -I$(CUB_INCLUDE_PATH)
LD_FLAGS=-L$(CUDA_LIBRARY_PATH) -lcudart -lcuda

NVCC=$(CUDA_BASE)/bin/nvcc -ccbin $(CXX)
NVCC_FLAGS=-std=c++14 -O2 --expt-relaxed-constexpr --expt-extended-lambda -DBOOST_PP_VARIADICS=1 -g --generate-line-info --source-in-ptx --generate-code arch=compute_50,code=sm_50 -I$(CUB_INCLUDE_PATH)


all: grid n_array tri_matrix tri_matrix_empty

clean:
	rm -f grid n_array tri_matrix tri_matrix_empty *.o

main.o: main.cc
	$(CXX) $(CXX_FLAGS) -c $< -o $@

tri_matrix.o: tri_matrix.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

tri_matrix: main.o tri_matrix.o
	$(CXX) $(CXX_FLAGS) $^ $(LD_FLAGS) -o $@

tri_matrix_empty.o: tri_matrix_empty.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

tri_matrix_empty: main.o tri_matrix_empty.o
	$(CXX) $(CXX_FLAGS) $^ $(LD_FLAGS) -o $@

grid.o: grid.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

grid: main.o grid.o
	$(CXX) $(CXX_FLAGS) $^ $(LD_FLAGS) -o $@

n_array.o: n_array.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

n_array: main.o n_array.o
	$(CXX) $(CXX_FLAGS) $^ $(LD_FLAGS) -o $@


.PHONY: run run_fastjet run_tri_matrix run_grid

run: run_fastjet run_tri_matrix run_grid run_n_array

# run fastjet to get reference results
ifdef FASTJET_BASE
run_fastjet:
	mkdir -p output/fastjet
	$(FASTJET_BASE)/example/fastjet_timing_plugins -antikt -r 0.4 -incl 1.0 < data/808_cms.dat > output/fastjet/ak_0.4.log
	$(FASTJET_BASE)/example/fastjet_timing_plugins -cam    -r 0.4 -incl 1.0 < data/808_cms.dat > output/fastjet/ca_0.4.log
	$(FASTJET_BASE)/example/fastjet_timing_plugins -kt     -r 0.4 -incl 1.0 < data/808_cms.dat > output/fastjet/kt_0.4.log
else
run_fastjet:
endif

# run the tri_matrix implementation
run_tri_matrix: tri_matrix
	mkdir -p output/tri_matrix
	./tri_matrix -antikt -r 0.4 -p 1.0 < data/808_cms.dat > output/tri_matrix/ak_0.4.log
	./tri_matrix -cam    -r 0.4 -p 1.0 < data/808_cms.dat > output/tri_matrix/ca_0.4.log
	./tri_matrix -kt     -r 0.4 -p 1.0 < data/808_cms.dat > output/tri_matrix/kt_0.4.log

# run the grid implementation
run_grid: grid
	mkdir -p output/grid
	./grid -antikt -r 0.4 -p 1.0 < data/808_cms.dat > output/grid/ak_0.4.log
	./grid -cam    -r 0.4 -p 1.0 < data/808_cms.dat > output/grid/ca_0.4.log
	./grid -kt     -r 0.4 -p 1.0 < data/808_cms.dat > output/grid/kt_0.4.log

# run the n_array implementation
run_n_array: n_array
	mkdir -p output/n_array
	./n_array -antikt -r 0.4 -p 1.0 < data/808_cms.dat > output/n_array/ak_0.4.log
	./n_array -cam    -r 0.4 -p 1.0 < data/808_cms.dat > output/n_array/ca_0.4.log
	./n_array -kt     -r 0.4 -p 1.0 < data/808_cms.dat > output/n_array/kt_0.4.log
