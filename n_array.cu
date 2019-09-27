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

struct PseudoJetExt {
  double px;
  double py;
  double pz;
  double E;
  double diB;
  double inv_diB;
  double phi;
  double rap;
  bool isJet;
};

const double pi = 3.141592653589793238462643383279502884197;
const double twopi = 6.283185307179586476925286766559005768394;
const double MaxRap = 1e5;
const double MAX_DOUBLE = 1.79769e+308;

__device__ void _set_jet(PseudoJetExt &jet) {
  jet.diB = jet.px * jet.px + jet.py * jet.py;
  jet.inv_diB = jet.diB > 1e-300 ? 1.0 / jet.diB : 1e300;
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

__device__ double yij_distance(PseudoJetExt &jet1, PseudoJetExt &jet2, Scheme scheme, double one_over_r2) {
  switch (scheme) {
    case Scheme::Kt:
      return min(jet1.diB, jet2.diB) * plain_distance(jet1, jet2) * one_over_r2;
      break;

    case Scheme::CambridgeAachen:
      return plain_distance(jet1, jet2) * one_over_r2;
      break;

    // TODO: store 1/diB instead of diB
    case Scheme::AntiKt:
      return min(jet1.inv_diB, jet2.inv_diB) * plain_distance(jet1, jet2) * one_over_r2;
      break;
  }

  // never reached
  return 0.;
}

__device__ void tid_to_ij(int &i, int &j, int tid) {
  tid += 1;
  j = std::ceil(std::sqrt(2 * tid + 0.25) - 0.5);
  i = trunc(tid - (j - 1) * j / 2.0);
  j -= 1;
  i -= 1;
}

__global__ void set_jets(PseudoJetExt *jets) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  _set_jet(jets[tid]);
}

struct Dist {
  double d;
  int i;
  int j;
};

struct dist_compare {
  __host__ __device__ Dist operator()(Dist &first, Dist &second) { return first.d < second.d ? first : second; }
};

__global__ void fastjet(PseudoJetExt *jets, int n, const Scheme scheme, const float one_over_r2) {
  // Specialize BlockReduce type for our thread block
  typedef BlockReduce<Dist, 1024> BlockReduceT;
  // Shared memory
  __shared__ typename BlockReduceT::TempStorage sdata;

  while (n > 0) {
    for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
      _set_jet(jets[tid]);
    }

    __syncthreads();

    Dist local_min;
    local_min.d = MAX_DOUBLE;
    int N = n * (n + 1) / 2;
    for (int tid = threadIdx.x; tid < N; tid += blockDim.x) {
      Dist dst;
      tid_to_ij(dst.i, dst.j, tid);

      if (dst.i == dst.j) {
        switch (scheme) {
          case Scheme::Kt:
            dst.d = jets[dst.i].diB;
            break;
          case Scheme::CambridgeAachen:
            dst.d = 1.;
            break;
          case Scheme::AntiKt:
            dst.d = jets[dst.i].inv_diB;
            break;
        }
      } else {
        dst.d = yij_distance(jets[dst.i], jets[dst.j], scheme, one_over_r2);
      }

      if (dst.d < local_min.d)
        local_min = dst;
    }

    if (threadIdx.x >= N) {
      local_min.d = MAX_DOUBLE;
    }

    Dist min = BlockReduceT(sdata).Reduce(local_min, dist_compare());

    if (threadIdx.x == 0) {
      int i, j;
      i = min.i;
      j = min.j;

      if (i == j) {
        PseudoJetExt temp;
        temp = jets[j];
        jets[j] = jets[n - 1];
        temp.isJet = true;
        jets[n - 1] = temp;
      } else {
        jets[i].px += jets[j].px;
        jets[i].py += jets[j].py;
        jets[i].pz += jets[j].pz;
        jets[i].E += jets[j].E;
        _set_jet(jets[i]);

        jets[j] = jets[n - 1];
      }
    }

    n--;
    __syncthreads();
  }
}

__global__ void init(const PseudoJet *particles, PseudoJetExt *jets, int size) {
  int first = threadIdx.x + blockIdx.x * blockDim.x;
  int grid = blockDim.x * gridDim.x;

  for (int i = first; i < size; i += grid) {
    jets[i].px = particles[i].px;
    jets[i].py = particles[i].py;
    jets[i].pz = particles[i].pz;
    jets[i].E = particles[i].E;
    _set_jet(jets[i]);
  }
}

__global__ void output(const PseudoJetExt *jets, PseudoJet *particles, int size) {
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
}

void cluster(PseudoJet *particles, int size, Scheme scheme, double r) {
#pragma regoin CudaMalloc
  PseudoJetExt *d_jets;
  cudaCheck(cudaMalloc(&d_jets, size * sizeof(PseudoJetExt)));
  init<<<8, 512>>>(particles, d_jets, size);
#pragma endregoin

  double one_over_r2 = 1. / (r * r);

  fastjet<<<1, 1024>>>(d_jets, size, scheme, one_over_r2);

  output<<<8, 512>>>(d_jets, particles, size);

  cudaCheck(cudaFree(d_jets));
}
