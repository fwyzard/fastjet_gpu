#include <assert.h>
#include <cmath>
#include <cub/block/block_reduce.cuh>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdio.h>
#include <vector>
// Here you can set the device ID that was assigned to you
#define MYDEVICE 0
#define OUTPUT_JETS false
#define BENCH !OUTPUT_JETS

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char *msg);

unsigned int upper_power_of_two(int v) {

  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;

  return v;
}

using namespace std;
using namespace cub;

struct PseudoJet {
  double px;
  double py;
  double pz;
  double E;
  double diB;
  double phi;
  double rap;
  bool isJet;

  __host__ __device__ double get_diB() const {
    return diB > 1e-300 ? 1.0 / diB : 1e300;
  }
};

const double pi = 3.141592653589793238462643383279502884197;
const double twopi = 6.283185307179586476925286766559005768394;
const double MaxRap = 1e5;
const double R = 0.4;
const double R2 = R * R;
const double invR2 = 1.0 / R2;
const double MAX_DOUBLE = 1.79769e+308;
#if OUTPUT_JETS
const double ptmin = 1.0;
const double dcut = ptmin * ptmin;
#endif

__device__ void _set_jet(PseudoJet &jet) {
  jet.diB = jet.px * jet.px + jet.py * jet.py;
  jet.isJet = false;

  if (jet.diB == 0.0) {
    jet.phi = 0.0;
  } else {
    jet.phi = atan2(jet.py, jet.px);
  }
  if (jet.phi < 0.0) {
    jet.phi += twopi;
  }
  if (jet.phi >= twopi) {
    jet.phi -= twopi;
  } // can happen if phi=-|eps<1e-15|?
  if (jet.E == abs(jet.pz) && jet.diB == 0) {
    // Point has infinite rapidity -- convert that into a very large
    // number, but in such a way that different 0-pt momenta will have
    // different rapidities (so as to lift the degeneracy between
    // them) [this can be relevant at parton-level]
    double MaxRapHere = MaxRap + abs(jet.pz);
    if (jet.pz >= 0.0) {
      jet.rap = MaxRapHere;
    } else {
      jet.rap = -MaxRapHere;
    }
  } else {
    // get the rapidity in a way that's modestly insensitive to roundoff
    // error when things pz,E are large (actually the best we can do without
    // explicit knowledge of mass)
    double effective_m2 = max(0.0, (jet.E + jet.pz) * (jet.E - jet.pz) -
                                       jet.diB); // force non tachyonic mass
    double E_plus_pz = jet.E + abs(jet.pz);      // the safer of p+, p-
    // p+/p- = (p+ p-) / (p-)^2 = (kt^2+m^2)/(p-)^2
    jet.rap = 0.5 * log((jet.diB + effective_m2) / (E_plus_pz * E_plus_pz));
    if (jet.pz > 0) {
      jet.rap = -jet.rap;
    }
  }
}

__device__ double plain_distance(PseudoJet &jet1, PseudoJet &jet2) {
  double dphi = abs(jet1.phi - jet2.phi);
  if (dphi > pi) {
    dphi = twopi - dphi;
  }
  double drap = jet1.rap - jet2.rap;
  return (dphi * dphi + drap * drap);
}

__device__ double yij_distance(PseudoJet &jet1, PseudoJet &jet2) {
  return min(jet1.get_diB(), jet2.get_diB()) * plain_distance(jet1, jet2) *
         invR2;
}

__device__ void tid_to_ij(int &i, int &j, int tid) {
  tid += 1;
  j = std::ceil(std::sqrt(2 * tid + 0.25) - 0.5);
  i = trunc(tid - (j - 1) * j / 2.0);
  j -= 1;
  i -= 1;
}

__global__ void set_jets(PseudoJet *jets) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  _set_jet(jets[tid]);
}

struct Dist {
  double d;
  int i;
  int j;
};

struct dist_compare {
  __host__ __device__ Dist operator()(Dist &first, Dist &second) {
    return first.d < second.d ? first : second;
  }
};

__global__ void reduction_min(PseudoJet *jets, Dist *distances_out,
                              int const distances_array_size,
                              int const num_particles) {
  // Specialize BlockReduce type for our thread block
  typedef BlockReduce<Dist, 1024> BlockReduceT;
  // Shared memory
  __shared__ typename BlockReduceT::TempStorage sdata;

  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  Dist dst;
  tid_to_ij(dst.i, dst.j, tid);

  if (tid >= distances_array_size || dst.j >= num_particles ||
      dst.i >= num_particles) {
    dst.d = MAX_DOUBLE;
  } else if (dst.i == dst.j) {
    dst.d = jets[dst.i].get_diB();
  } else {
    dst.d = yij_distance(jets[dst.i], jets[dst.j]);
  }

  Dist min = BlockReduceT(sdata).Reduce(dst, dist_compare());

  if (threadIdx.x == 0) {
    distances_out[blockIdx.x] = min;
    // printf("%4d%4d%4d%20.8e\n", num_particles, min.i, min.j, min.d);
  }
}

__global__ void reduction_blocks(PseudoJet *jets, Dist *distances_out,
                                 int const distances_array_size,
                                 int const num_particles) {
  // Specialize BlockReduce type for our thread block
  typedef BlockReduce<Dist, 512> BlockReduceT;
  // Shared memory
  __shared__ typename BlockReduceT::TempStorage sdata;

  unsigned int tid = threadIdx.x;

  Dist dst;

  if (tid >= distances_array_size) {
    dst.d = MAX_DOUBLE;
  } else {
    dst = distances_out[tid];
  }

  Dist min = BlockReduceT(sdata).Reduce(dst, dist_compare());

  if (tid == 0) {
    int i, j;
    i = min.i;
    j = min.j;

    // printf("block %4d%4d%4d%20.8e\n", num_particles, i, j, min.d);

    // int f, e;
    // tid_to_ij(f, e, 58101);
    // printf("%4d%4d%4d%20.8e\n", num_particles, f, e, distances[58101]);

    if (i == j) {
      PseudoJet temp;
      temp = jets[j];
      jets[j] = jets[num_particles - 1];
      temp.isJet = true;
      jets[num_particles - 1] = temp;
    } else {
      jets[i].px += jets[j].px;
      jets[i].py += jets[j].py;
      jets[i].pz += jets[j].pz;
      jets[i].E += jets[j].E;
      _set_jet(jets[i]);

      jets[j] = jets[num_particles - 1];
    }
  }
}

int main() {
  cudaSetDevice(0);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  printf("Device Name: %s\n", prop.name);

  int num_particles = 0;
  // Increase the number to process more events
  int num_events = 1;

  // Loop events
  for (int event = 0; event < num_events; event++) {
    PseudoJet *h_jets = NULL;
    PseudoJet *h_more_jets = NULL;
    PseudoJet temp;

    // Read particles
    num_particles = 0;
    while (true) {
      cin >> temp.px >> temp.py >> temp.pz >> temp.E;

      if (cin.fail())
        break;

      num_particles++;

      h_more_jets =
          (PseudoJet *)realloc(h_jets, num_particles * sizeof(PseudoJet));

      if (h_more_jets != NULL) {
        h_jets = h_more_jets;
        h_jets[num_particles - 1] = temp;
      } else {
        free(h_jets);
        puts("Error (re)allocating memory");
        exit(1);
      }
    }

    cin.clear();
    cin.ignore(numeric_limits<streamsize>::max(), '\n');

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

#pragma regoin CudaMalloc
    PseudoJet *d_jets = 0;
    cudaMalloc((void **)&d_jets, num_particles * sizeof(PseudoJet));
    cudaMemcpy(d_jets, h_jets, num_particles * sizeof(PseudoJet),
               cudaMemcpyHostToDevice);

    int num_threads = num_particles;
    int num_blocks = (num_particles + num_threads) / (num_threads + 1);

    Dist *d_out = 0;
    cudaMalloc((void **)&d_out, num_threads * sizeof(Dist));
#pragma endregoin

// Benchmarking
#if BENCH
    float milliseconds;
    vector<double> acc;
    for (int s = 0; s < 1000; s++) {
      cudaEventRecord(start);
#endif
      // Compute dIB, eta, phi for each jet
      set_jets<<<num_blocks, num_threads>>>(d_jets);

      // Loop n times reduce + recombine
      for (int n = num_particles; n > 0; n--) {
        num_threads = (n * (n + 1) / 2);
        num_blocks = (num_threads / 1024) + 1;

        // Find the minimum in each block for the distances array
        reduction_min<<<num_blocks, 1024, 1024 * sizeof(Dist)>>>(
            d_jets, d_out, num_threads, n);

        // // Find the minimum of all blocks
        int b = upper_power_of_two(num_blocks - 1) + 1;
        // cout << num_blocks << "\t" << b + 1 << endl;
        reduction_blocks<<<1, 512, 512 * sizeof(Dist)>>>(d_jets, d_out, num_blocks,
                                                     n);
      }
#if BENCH
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);

      cudaEventElapsedTime(&milliseconds, start, stop);
      printf("run %d\t%.3fms\n", s, milliseconds);
      acc.push_back(milliseconds);
    }

    double sum = std::accumulate(acc.begin(), acc.end(), 0.0);
    double mean = sum / acc.size();

    double sq_sum =
        std::inner_product(acc.begin(), acc.end(), acc.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / acc.size() - mean * mean);
    printf("n =  %d\n", num_particles);
    printf("mean %.3fms\n", mean);
    printf("std %.3fms\n", stdev);
#endif

    checkCUDAError("kernal launch");

#if OUTPUT_JETS
    cudaMemcpy(h_jets, d_jets, num_particles * sizeof(PseudoJet),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_particles; i++)
      if (h_jets[i].diB >= dcut && h_jets[i].isJet)
        printf("%15.8f %15.8f %15.8f %15.8f\n",
               h_jets[i].px, h_jets[i].py, h_jets[i].pz, h_jets[i].E
               );
#endif

    // free device memory
    cudaFree(d_jets);
    // cudaFree(d_distances);
    // cudaFree(d_indices);
    // cudaFree(d_indices_ii);
    // cudaFree(d_indices_jj);
    cudaFree(d_out);

    free(h_jets);
  }

  return 0;
}

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(-1);
  }
}