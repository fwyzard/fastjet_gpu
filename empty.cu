#include <iostream>
#include <numeric>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <vector>

using namespace std;

__global__ void empty0() { return; }
__global__ void empty1() { return; }
__global__ void empty2() { return; }

struct Dist {
  double distance;
  int i;
  int j;
};

__global__ void reduce(Dist *ddd) {
  extern __shared__ Dist sdata[];
  int tid = threadIdx.x;
  Dist d;
  d.distance = threadIdx.x * 1.0;
  d.i = tid;
  d.j = tid;

  if (tid >= 354)
    return;

  for (int i = 0; i < blockDim.x; i++) {
    sdata[tid] = d;

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (tid < s && (tid + s) < blockDim.x) {
        if (sdata[tid + s].distance < sdata[tid].distance) {
          sdata[tid] = sdata[tid + s];
        }
      }
      __syncthreads();
    }

    if (tid == 0) {
      ddd[0] = sdata[0];
    }
  }
}

int main() {
  cudaSetDevice(0);
  thrust::device_vector<Dist> d_d(1);
  Dist *d_d_ptr = thrust::raw_pointer_cast(d_d.data());

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  vector<double> acc;
  float milliseconds;
  for (int s = 0; s < 1000; s++) {
    cudaEventRecord(start);
    for (int i = 354; i > 0; i--) {
      empty0<<<1, i>>>();
    //   empty1<<<1, 65>>>();
    //   empty0<<<1, 1024>>>();
    }

    // for (int i = 354; i > 0; i--) {
    // reduce<<<1, 354, sizeof(Dist) * 354>>>(d_d_ptr);
    // }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("run %d\t%.3fms\n", s, milliseconds);
    acc.push_back(milliseconds);
  }

  double sum = std::accumulate(acc.begin(), acc.end(), 0.0);
  double mean = sum / acc.size();

  double sq_sum = std::inner_product(acc.begin(), acc.end(), acc.begin(), 0.0);
  double stdev = std::sqrt(sq_sum / acc.size() - mean * mean);
  printf("mean %.3fms\n", mean);
  printf("std %.3fms\n", stdev);

  return 0;
}