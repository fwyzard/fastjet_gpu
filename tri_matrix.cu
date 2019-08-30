#include <assert.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdio.h>
#include <vector>
// Here you can set the device ID that was assigned to you
#define MYDEVICE 0
#define OUTPUT_JETS false
#define BENCH true

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

struct PseudoJet {
  double px;
  double py;
  double pz;
  double E;
  double diB;
  double phi;
  double rap;
  bool isJet;
};

const double pi = 3.141592653589793238462643383279502884197;
const double twopi = 6.283185307179586476925286766559005768394;
const double MaxRap = 1e5;
const double R = 0.6;
const double R2 = R * R;
const double invR2 = 1.0 / R2;
#if OUTPUT_JETS
const double ptmin = 5.0;
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
  return min(jet1.diB, jet2.diB) * plain_distance(jet1, jet2) * invR2;
}

__device__ void tid_to_ij(int &i, int &j, int tid) {
  tid += 1;
  j = std::ceil(std::sqrt(2 * tid + 0.25) - 0.5);
  i = trunc(tid - (j - 1) * j / 2.0);
  j -= 1;
  i -= 1;
}

__global__ void set_jets(PseudoJet *jets, int *min) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  _set_jet(jets[tid]);
  if (tid == 0)
    min[0] = -1;
}

__global__ void reduction_min(PseudoJet *jets, double *distances,
                              double *distances_out, int *indices,
                              int *indices_ii, int *indices_jj,
                              int const distances_array_size,
                              int const num_particels, int *min) {
  extern __shared__ double sdata[];
  double *s_distances = sdata;
  int *s_indices = (int *)&s_distances[blockDim.x];

  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= distances_array_size)
    return;

  indices[tid] = tid;

  int i, j;
  tid_to_ij(i, j, tid);
  indices_ii[tid] = i;
  indices_jj[tid] = j;

  double d;

  if (i == j) {
    d = jets[i].diB;
  } else {
    d = yij_distance(jets[i], jets[j]);
  }

  i = threadIdx.x;
  s_distances[i] = d;
  s_indices[i] = tid;
  __syncthreads();

  int jj;
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (i < s && (tid + s) < distances_array_size) {
      jj = indices_jj[s_indices[i + s]];
      if (s_distances[i] > s_distances[i + s] && jj < num_particels) {
        s_distances[i] = s_distances[i + s];

        s_indices[i] = s_indices[i + s];
      }
    }
    __syncthreads();
  }

  if (i == 0) {
    distances_out[blockIdx.x] = s_distances[0];
    int min_tid = s_indices[0];
    indices[blockIdx.x] = min_tid;
  }
}

__global__ void reduction_blocks(PseudoJet *jets, double *distances,
                                 double *distances_out, int *indices,
                                 int *indices_ii, int *indices_jj,
                                 int const distances_array_size,
                                 int const num_particles, int *min) {
  extern __shared__ double sdata[];
  double *s_distances = sdata;
  int *s_indices = (int *)&s_distances[blockDim.x];

  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int i = threadIdx.x;

  if (tid >= distances_array_size)
    return;

  s_distances[i] = distances[tid];
  s_indices[i] = indices[tid];
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (i < s && (tid + s) < distances_array_size) {
      if (s_distances[i] > s_distances[i + s]) {
        s_distances[i] = s_distances[i + s];

        s_indices[i] = s_indices[i + s];
      }
    }
    __syncthreads();
  }

  if (i == 0) {
    distances_out[blockIdx.x] = s_distances[0];
    int min_tid = s_indices[0];
    indices[blockIdx.x] = min_tid;

    int i, j;
    i = indices_ii[min_tid];
    j = indices_jj[min_tid];

    min[0] = i;
    min[0] = j;

    // printf("%4d%4d%4d%20.8e\n", num_particles, i, j, s_distances[0]);

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

  int num_particels = 0;
  // Increase the number to process more events
  int num_events = 1;

  // Loop events
  for (int event = 0; event < num_events; event++) {
    PseudoJet *h_jets = NULL;
    PseudoJet *h_more_jets = NULL;
    PseudoJet temp;

    // Read particles
    num_particels = 0;
    while (true) {
      cin >> temp.px >> temp.py >> temp.pz >> temp.E;

      if (cin.fail())
        break;

      num_particels++;

      h_more_jets =
          (PseudoJet *)realloc(h_jets, num_particels * sizeof(PseudoJet));

      if (h_more_jets != NULL) {
        h_jets = h_more_jets;
        h_jets[num_particels - 1] = temp;
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
    cudaMalloc((void **)&d_jets, num_particels * sizeof(PseudoJet));
    cudaMemcpy(d_jets, h_jets, num_particels * sizeof(PseudoJet),
               cudaMemcpyHostToDevice);

    double *d_distances = 0;
    cudaMalloc((void **)&d_distances,
               (num_particels * (num_particels + 1) / 2) * sizeof(double));

    int *d_indices = 0;
    cudaMalloc((void **)&d_indices,
               (num_particels * (num_particels + 1) / 2) * sizeof(int));
    int *d_indices_ii = 0;
    cudaMalloc((void **)&d_indices_ii,
               (num_particels * (num_particels + 1) / 2) * sizeof(int));
    int *d_indices_jj = 0;
    cudaMalloc((void **)&d_indices_jj,
               (num_particels * (num_particels + 1) / 2) * sizeof(int));

    int *d_min = 0;
    cudaMalloc((void **)&d_min, 2 * sizeof(int));

    int num_threads = num_particels;
    int num_blocks = (num_particels + num_threads) / (num_threads + 1);

    double *d_out = 0;
    cudaMalloc((void **)&d_out, num_threads * sizeof(double));
#pragma endregoin

// Benchmarking
#if BENCH
    float milliseconds;
    vector<double> acc;
    for (int s = 0; s < 1000; s++) {
      cudaEventRecord(start);
#endif
      // Compute dIB, eta, phi for each jet
      set_jets<<<num_blocks, num_threads>>>(d_jets, d_min);

      // Loop n times reduce + recombine
      for (int n = num_particels; n > 0; n--) {
        num_threads = (n * (n + 1) / 2);
        num_blocks = (num_threads / 1024) + 1;

        // Find the minimum in each block for the distances array
        reduction_min<<<num_blocks, 1024,
                        1024 * sizeof(double) + 1024 * sizeof(int)>>>(
            d_jets, d_distances, d_out, d_indices, d_indices_ii,
            d_indices_jj,
            num_threads, n, d_min);

        // // Find the minimum of all blocks
        int b = upper_power_of_two(num_blocks - 1);
        // cout << num_blocks << "\t" << b + 1 << endl;
        reduction_blocks<<<1, b + 1, (b + 1) * sizeof(double) +
                                         num_blocks * sizeof(int)>>>(
            d_jets, d_out, d_out, d_indices, d_indices_ii, d_indices_jj,
            num_blocks, n, d_min);
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
    printf("n =  %d\n", num_particels);
    printf("mean %.3fms\n", mean);
    printf("std %.3fms\n", stdev);
#endif

    checkCUDAError("kernal launch");

#if OUTPUT_JETS
    cudaMemcpy(h_jets, d_jets, num_particels * sizeof(PseudoJet),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_particels; i++)
      if (h_jets[i].diB >= dcut && h_jets[i].isJet)
        printf("%15.8f %15.8f %15.8f\n", h_jets[i].rap, h_jets[i].phi,
               sqrt(h_jets[i].diB));
#endif

    // free device memory
    cudaFree(d_jets);
    cudaFree(d_distances);
    cudaFree(d_indices);
    cudaFree(d_indices_ii);
    cudaFree(d_indices_jj);
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