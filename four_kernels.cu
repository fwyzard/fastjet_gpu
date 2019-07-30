#include <assert.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdio.h>

// Here you can set the device ID that was assigned to you
#define MYDEVICE 5

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char *msg);
bool double_equals(double a, double b, double epsilon = 1e-6);

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
const double ptmin = 5.0;
const double dcut = ptmin * ptmin;
int const NUM_PARTICLES = 354;

__device__ static double atomicMin(double *address, double val) {
  unsigned long long int *address_as_i = (unsigned long long int *)address;
  unsigned long long int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = ::atomicCAS(
        address_as_i, assumed,
        __double_as_longlong(fmin(val, __longlong_as_double(assumed))));
  } while (assumed != old);
  return __longlong_as_double(old);
}

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

void _set_jet_h(PseudoJet &jet) {
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

__global__ void set_jets(PseudoJet *jets) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  _set_jet(jets[tid]);
}

__global__ void set_distances(PseudoJet *jets, double *distances, int *indices,
                              int const num_particles) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int N = num_particles * (num_particles + 1) / 2;

  if (tid >= N)
    return;

  indices[tid] = tid;

  int i, j;
  tid_to_ij(i, j, tid);

  if (i == j) {
    distances[tid] = jets[i].diB;
  } else {
    distances[tid] = yij_distance(jets[i], jets[j]);
  }
}

__global__ void reduction_min_second(PseudoJet *jets, double *distances,
                                     double *distances_out, int *indices,
                                     int *indices_out, int const num_particles,
                                     bool debug) {
  extern __shared__ double sdata[];
  double *s_distances = sdata;
  int *s_indices = (int *)&s_distances[blockDim.x];

  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int i = threadIdx.x;

  if (tid >= num_particles)
    return;

  s_distances[i] = distances[tid];
  s_indices[i] = indices[tid];
  __syncthreads();

  int ii, jj;
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (i < s && (tid + s) < num_particles) {
      tid_to_ij(ii, jj, s_indices[i + s]);
      if (s_distances[i] > s_distances[i + s] && jj < num_particles) {
        s_distances[i] = s_distances[i + s];

        s_indices[i] = s_indices[i + s];
      }
    }
    __syncthreads();
  }
  if (debug)
    printf("%15.8e\n", s_distances[i]);

  if (i == 0) {
    // if (debug) {
    //   printf("%15.8e ", distances_out[0]);
    //   printf("%d\n", indices_out[0]);
    // }
    // printf("end\n");
    distances_out[blockIdx.x] = s_distances[0];
    indices_out[blockIdx.x] = s_indices[0];
    // printf("%15.8e\n", distances_out[1]);
    // printf("%d\n", indices_out[1]);
  }
}

__global__ void reco_and_recal(PseudoJet *jets, double *distances,
                               double *min_distances, int *min_indices,
                               int *min_index, int const n) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= n)
    return;

  int i, j;
  int index = *min_index; // indices[0];
  tid_to_ij(i, j, index);

  if (j >= n) {
    tid_to_ij(i, j, index - n);
  }

  if (tid == 0) {
    if (i == j) {
      PseudoJet temp;
      temp = jets[j];
      jets[j] = jets[n];
      temp.isJet = true;
      jets[n] = temp;
    } else {
      jets[i].px += jets[j].px;
      jets[i].py += jets[j].py;
      jets[i].pz += jets[j].pz;
      jets[i].E += jets[j].E;
      _set_jet(jets[i]);

      jets[j] = jets[n];
    }
  }
  __syncthreads();

  int tid_j = tid + ((j) * (j + 1) / 2);

  double dj = 0;
  // atomicMin(&minimum, ymin);
  if (tid == j) {
    dj = jets[tid].diB;
  } else if (tid < j) {
    dj = yij_distance(jets[tid], jets[j]);
  } else {
    tid_j = j + ((tid) * (tid + 1) / 2);
    dj = yij_distance(jets[tid], jets[j]);
  }
  distances[tid_j] = dj;

  if (dj < min_distances[j]) {
    atomicMin(&min_distances[j], dj);
    atomicExch(&min_indices[j], tid_j);
  }

  int tid_i = tid_j;
  double di = 0;
  if (i != j) {
    tid_i = tid + ((i) * (i + 1) / 2);
    if (tid == i)
      di = jets[tid].diB;
    else if (tid < i)
      di = yij_distance(jets[tid], jets[i]);
  } else {
    tid_i = i + ((tid) * (tid + 1) / 2);
    di = yij_distance(jets[tid], jets[i]);
  }

  distances[tid_i] = di;

  if (di < min_distances[i]) {
    atomicMin(&min_distances[i], di);
    atomicExch(&min_indices[i], tid_i);
  }
}

int main() {
  int d_id;
  cudaDeviceProp d_prop;

  cudaChooseDevice(&d_id, &d_prop);
  cudaSetDevice(MYDEVICE);

  PseudoJet *h_jets = 0;
  h_jets = (PseudoJet *)malloc(NUM_PARTICLES * sizeof(PseudoJet));

  int i;
  for (i = 0; i < NUM_PARTICLES; i++) {
    cin >> h_jets[i].px >> h_jets[i].py >> h_jets[i].pz >> h_jets[i].E;
    _set_jet_h(h_jets[i]);
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  PseudoJet *d_jets = 0;
  cudaMalloc((void **)&d_jets, NUM_PARTICLES * sizeof(PseudoJet));
  cudaMemcpy(d_jets, h_jets, NUM_PARTICLES * sizeof(PseudoJet),
             cudaMemcpyHostToDevice);

  double *d_distances = 0;
  cudaMalloc((void **)&d_distances,
             (NUM_PARTICLES * (NUM_PARTICLES + 1) / 2) * sizeof(double));

  int *d_indices = 0;
  cudaMalloc((void **)&d_indices,
             (NUM_PARTICLES * (NUM_PARTICLES + 1) / 2) * sizeof(int));

  int num_threads = 354;
  int num_blocks = (NUM_PARTICLES + num_threads) / (num_threads + 1);

  double *d_distances_out = 0;
  cudaMalloc((void **)&d_distances_out, num_blocks * sizeof(double));

  int *d_indices_out = 0;
  cudaMalloc((void **)&d_indices_out, num_blocks * sizeof(int));

  double *d_distances_min = 0;
  cudaMalloc((void **)&d_distances_min, NUM_PARTICLES * sizeof(double));

  int *d_indices_min = 0;
  cudaMalloc((void **)&d_indices_min, NUM_PARTICLES * sizeof(int));

  cudaEventRecord(start);
  set_jets<<<num_blocks, num_threads>>>(d_jets);

  num_threads = (NUM_PARTICLES * (NUM_PARTICLES + 1) / 2);
  num_blocks = (num_threads / 1024) + 1;
  set_distances<<<num_blocks, 1024>>>(d_jets, d_distances, d_indices,
                                      NUM_PARTICLES);

  int N = 0;
  for (int n = NUM_PARTICLES; n > 0; n--) {
    N = (n * (n + 1) / 2);
    num_threads = (NUM_PARTICLES / 512 + 1) * 512;
    num_blocks = (num_threads / (512 + 1)) + 1;
    // cout << num_blocks << " " << num_threads << endl;

    reduction_min_second<<<num_blocks, num_threads,
                           num_threads * sizeof(double) +
                               num_threads * sizeof(int)>>>(
        d_jets, &d_distances[N - n], d_distances_out, &d_indices[N - n],
        d_indices_out, n, false);

    reduction_min_second<<<1, num_blocks, num_blocks * sizeof(double) +
                                              num_blocks * sizeof(int)>>>(
        d_jets, d_distances_out, &d_distances_min[n - 1], d_indices_out,
        &d_indices_min[n - 1], n, false);

    // recalculate_distances<<<(NUM_PARTICLES / 1024) + 1, 1024>>>(
    //     d_jets, d_distances, d_indices, n - 1);
  }

  for (int n = NUM_PARTICLES; n > 0; n--) {
    num_threads = (n / 512 + 1) * 512;
    num_blocks = (num_threads / (512 + 1)) + 1;
    reduction_min_second<<<num_blocks, num_threads,
                           num_threads * sizeof(double) +
                               num_threads * sizeof(int)>>>(
        d_jets, d_distances_min, d_distances_out, d_indices_min, d_indices_out,
        n, false);

    reduction_min_second<<<1, num_blocks, num_blocks * sizeof(double) +
                                              num_blocks * sizeof(int)>>>(
        d_jets, d_distances_out, d_distances_out, d_indices_out, d_indices_out,
        n, false);

    reco_and_recal<<<num_blocks, num_threads>>>(d_jets, d_distances,
                                                d_distances_min, d_indices_min,
                                                &d_indices_out[0], n - 1);

    double *h_distances_out = 0;
    h_distances_out = (double *)malloc(num_blocks * sizeof(double));
    cudaMemcpy(h_distances_out, d_distances_out, num_blocks * sizeof(double),
               cudaMemcpyDeviceToHost);
    int *h_indices_out = 0;
    h_indices_out = (int *)malloc(num_blocks * sizeof(int));
    cudaMemcpy(h_indices_out, d_indices_out, num_blocks * sizeof(int),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < NUM_PARTICLES; i++)
      printf("i = %d, d = %15.8e\n", h_indices_out[i], h_distances_out[i]);
    printf("end\n");
  }
  cudaEventRecord(stop);
  cudaMemcpy(h_jets, d_jets, NUM_PARTICLES * sizeof(PseudoJet),
             cudaMemcpyDeviceToHost);

  // Check for any CUDA errors
  checkCUDAError("kernal launch");

  // double *h_out = 0;
  // h_out = (double *)malloc(num_blocks * sizeof(double));
  // cudaMemcpy(h_out, d_out, num_blocks * sizeof(double),
  //            cudaMemcpyDeviceToHost);
  // double *h_distances_min = 0;
  // h_distances_min = (double *)malloc(NUM_PARTICLES * sizeof(double));
  // cudaMemcpy(h_distances_min, d_distances_min, NUM_PARTICLES *
  // sizeof(double),
  //            cudaMemcpyDeviceToHost);
  // int *h_indices_min = 0;
  // h_indices_min = (int *)malloc(NUM_PARTICLES * sizeof(int));
  // cudaMemcpy(h_indices_min, d_indices_min, NUM_PARTICLES * sizeof(int),
  //            cudaMemcpyDeviceToHost);
  double *h_distances_out = 0;
  h_distances_out = (double *)malloc(num_blocks * sizeof(double));
  cudaMemcpy(h_distances_out, d_distances_out, num_blocks * sizeof(double),
             cudaMemcpyDeviceToHost);
  int *h_indices_out = 0;
  h_indices_out = (int *)malloc(num_blocks * sizeof(int));
  cudaMemcpy(h_indices_out, d_indices_out, num_blocks * sizeof(int),
             cudaMemcpyDeviceToHost);

  for (int i = 0; i < NUM_PARTICLES; i++)
    //   printf("i = %d, d = %15.8e\n", h_indices_out[i], h_distances_out[i]);
    // printf("i = %d, d = %15.8e\n", h_indices_out[0], h_distances_out[0]);
    if (h_jets[i].diB >= dcut && h_jets[i].isJet)
      printf("%15.8f %15.8f %15.8f\n", h_jets[i].rap, h_jets[i].phi,
             sqrt(h_jets[i].diB));

  // Check for any CUDA errors
  checkCUDAError("cudaMemcpy calls");

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("time = %10.8f\n", milliseconds);

  // free device memory
  cudaFree(d_jets);
  cudaFree(d_distances);
  cudaFree(d_distances_out);
  cudaFree(d_distances_min);
  cudaFree(d_indices);
  cudaFree(d_indices_out);
  cudaFree(d_indices_min);
  // cudaFree(d_out);

  // free host memory
  free(h_jets);
  // free(h_out);

  return 0;
}

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(-1);
  }
}

bool double_equals(double a, double b, double epsilon) {
  return std::abs(a - b) < epsilon;
}
