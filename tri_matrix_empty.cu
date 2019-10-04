#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdio.h>
#include <vector>

#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

#include "PseudoJet.h"
#include "cluster.h"
#include "cudaCheck.h"

using namespace std;
using namespace cub;

unsigned int upper_power_of_two(int v) {
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;

  return v;
}

struct PseudoJetExt {
  double px;
  double py;
  double pz;
  double E;
  double diB;
  double phi;
  double rap;
  bool isJet;

  __host__ __device__ double get_diB(Algorithm algo) const {
    switch (algo) {
      case Algorithm::Kt:
        return diB;

      case Algorithm::CambridgeAachen:
        return 1.;

      case Algorithm::AntiKt:
        return diB > 1e-300 ? 1.0 / diB : 1e300;
    }
    // never reached
    return diB;
  }
};

const double pi = 3.141592653589793238462643383279502884197;
const double twopi = 6.283185307179586476925286766559005768394;
const double MaxRap = 1e5;
#if 0
const double MAX_DOUBLE = 1.79769e+308;
#endif

__device__ void _set_jet(PseudoJetExt &jet) {
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
  }  // can happen if phi=-|eps<1e-15|?
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
    double effective_m2 = max(0.0, (jet.E + jet.pz) * (jet.E - jet.pz) - jet.diB);  // force non tachyonic mass
    double E_plus_pz = jet.E + abs(jet.pz);                                         // the safer of p+, p-
    // p+/p- = (p+ p-) / (p-)^2 = (kt^2+m^2)/(p-)^2
    jet.rap = 0.5 * log((jet.diB + effective_m2) / (E_plus_pz * E_plus_pz));
    if (jet.pz > 0) {
      jet.rap = -jet.rap;
    }
  }
}

__device__ double plain_distance(PseudoJetExt &jet1, PseudoJetExt &jet2) {
  double dphi = abs(jet1.phi - jet2.phi);
  if (dphi > pi) {
    dphi = twopi - dphi;
  }
  double drap = jet1.rap - jet2.rap;
  return (dphi * dphi + drap * drap);
}

__device__ double yij_distance(PseudoJetExt &jet1, PseudoJetExt &jet2, Algorithm algo, double one_over_r2) {
  return min(jet1.get_diB(algo), jet2.get_diB(algo)) * plain_distance(jet1, jet2) * one_over_r2;
}

__device__ void tid_to_ij(int &i, int &j, int tid) {
  tid += 1;
  j = std::ceil(std::sqrt(2 * tid + 0.25) - 0.5);
  i = trunc(tid - (j - 1) * j / 2.0);
  j -= 1;
  i -= 1;
}

__global__ void set_jets(PseudoJetExt *jets) {
#if 0
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  _set_jet(jets[tid]);
#endif
}

struct Dist {
  double d;
  int i;
  int j;
};

struct dist_compare {
  __host__ __device__ Dist operator()(Dist &first, Dist &second) { return first.d < second.d ? first : second; }
};

__global__ void reduction_min(PseudoJetExt *jets,
                              Dist *distances_out,
                              int const distances_array_size,
                              int const num_particles,
                              Algorithm algo,
                              double one_over_r2) {
#if 0
  // Specialize BlockReduce type for our thread block
  typedef BlockReduce<Dist, 1024> BlockReduceT;
  // Shared memory
  __shared__ typename BlockReduceT::TempStorage sdata;

  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  Dist dst;
  tid_to_ij(dst.i, dst.j, tid);

  if (tid >= distances_array_size || dst.j >= num_particles || dst.i >= num_particles) {
    dst.d = MAX_DOUBLE;
  } else if (dst.i == dst.j) {
    dst.d = jets[dst.i].get_diB(algo);
  } else {
    dst.d = yij_distance(jets[dst.i], jets[dst.j], algo, one_over_r2);
  }

  Dist min = BlockReduceT(sdata).Reduce(dst, dist_compare());

  if (threadIdx.x == 0) {
    distances_out[blockIdx.x] = min;
    // printf("%4d%4d%4d%20.8e\n", num_particles, min.i, min.j, min.d);
  }
#endif
}

__global__ void reduction_blocks(PseudoJetExt *jets,
                                 Dist *distances_out,
                                 int const distances_array_size,
                                 int const num_particles) {
#if 0
  // Specialize BlockReduce type for our thread block
  typedef BlockReduce<Dist, 1024> BlockReduceT;
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
      PseudoJetExt temp;
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
#endif
}

__global__ void init(const PseudoJet *particles, PseudoJetExt *jets, int size) {
#if 0
  int first = threadIdx.x + blockIdx.x * blockDim.x;
  int grid = blockDim.x * gridDim.x;

  for (int i = first; i < size; i += grid) {
    jets[i].px = particles[i].px;
    jets[i].py = particles[i].py;
    jets[i].pz = particles[i].pz;
    jets[i].E = particles[i].E;
    _set_jet(jets[i]);
  }
#endif
}

__global__ void output(const PseudoJetExt *jets, PseudoJet *particles, int size) {
#if 0
  int first = threadIdx.x + blockIdx.x * blockDim.x;
  int grid = blockDim.x * gridDim.x;

  for (int i = first; i < size; i += grid) {
    particles[i].px = jets[i].px;
    particles[i].py = jets[i].py;
    particles[i].pz = jets[i].pz;
    particles[i].E = jets[i].E;
    particles[i].index = i;
    particles[i].isJet = jets[i].isJet;
  }
#endif
}

void cluster(PseudoJet *particles, int size, Algorithm algo, double r) {
#pragma regoin CudaMalloc
  PseudoJetExt *d_jets;
  cudaCheck(cudaMalloc(&d_jets, size * sizeof(PseudoJetExt)));
  init<<<8, 512>>>(particles, d_jets, size);

  Dist *d_out = 0;
  cudaCheck(cudaMalloc((void **)&d_out, size * sizeof(Dist)));
#pragma endregoin

  int num_threads = size;
  int num_blocks = (size + num_threads) / (num_threads + 1);
  double one_over_r2 = 1. / (r * r);

  // Compute dIB, eta, phi for each jet
  set_jets<<<num_blocks, num_threads>>>(d_jets);

  // Loop n times reduce + recombine
  for (int n = size; n > 0; n--) {
    num_threads = (n * (n + 1) / 2);
    num_blocks = (num_threads / 1024) + 1;

    // Find the minimum in each block for the distances array
    reduction_min<<<num_blocks, 1024, 1024 * sizeof(Dist)>>>(d_jets, d_out, num_threads, n, algo, one_over_r2);

    // // Find the minimum of all blocks
    int b = upper_power_of_two(num_blocks - 1) + 1;
    // cout << num_blocks << "\t" << b + 1 << endl;
    reduction_blocks<<<1, 1024, 1024 * sizeof(Dist)>>>(d_jets, d_out, num_blocks, n);
  }

  output<<<8, 512>>>(d_jets, particles, size);

  cudaCheck(cudaFree(d_jets));
  cudaCheck(cudaFree(d_out));
}
