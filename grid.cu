#include <cmath>

#include <cuda_runtime.h>

#include "PseudoJet.h"
#include "cluster.h"
#include "cudaCheck.h"

#pragma region consts
const double MaxRap = 1e5;
#pragma endregion

#pragma region struct
template <typename T>
__host__ __device__ inline void swap(T &a, T &b) {
  auto t = std::move(a);
  a = std::move(b);
  b = std::move(t);
}

using GridIndexType = int;
using ParticleIndexType = int;

struct PseudoJetExt {
  int index;
  bool isJet;
  double px;
  double py;
  double pz;
  double E;
  double rap;
  double phi;
  double diB;
  GridIndexType i;
  GridIndexType j;
};

struct Dist {
  double distance;
  ParticleIndexType i;
  ParticleIndexType j;
};

struct Grid {
  const double min_rap;
  const double max_rap;
  const double min_phi;
  const double max_phi;
  const double r;
  const GridIndexType max_i;
  const GridIndexType max_j;
  const ParticleIndexType n;

  ParticleIndexType *jets;

  // TODO use a smaller grid size (esimate from distributions in data/mc)
  // TODO usa a SoA
  __host__ Grid(double min_rap, double max_rap, double min_phi, double max_phi, double r, ParticleIndexType n)
      : min_rap(min_rap),
        max_rap(max_rap),
        min_phi(min_phi),
        max_phi(min_phi),
        r((2 * M_PI) / (int)((2 * M_PI) / r)),  // round up the grid size to have an integer number of cells in phi
        max_i((GridIndexType)(((max_rap - min_rap) / r))),
        max_j((GridIndexType)(((max_phi - min_phi) / r))),
        n(n),
        jets(nullptr) {}

  __host__ __device__ constexpr inline GridIndexType i(double rap) const {
    return (GridIndexType)((rap - min_rap) / r);
  }

  __host__ __device__ constexpr inline GridIndexType j(double phi) const {
    return (GridIndexType)((phi - min_phi) / r);
  }

  __host__ __device__ constexpr inline double rap_min(GridIndexType i) const { return min_rap + r * i; }

  __host__ __device__ constexpr inline double rap_max(GridIndexType i) const { return min_rap + r * (i + 1); }

  __host__ __device__ constexpr inline double phi_min(GridIndexType j) const { return min_phi + r * j; }

  __host__ __device__ constexpr inline double phi_max(GridIndexType j) const { return min_phi + r * (j + 1); }

  __host__ __device__ constexpr inline int index(GridIndexType i, GridIndexType j) const { return (int)max_j * i + j; }

  __host__ __device__ constexpr inline int offset(GridIndexType i, GridIndexType j) const { return index(i, j) * n; }
};
#pragma endregion

#pragma region device_functions
__host__ __device__ constexpr inline double safe_inverse(double x) { return (x > 1e-300) ? (1.0 / x) : 1e300; }

__host__ __device__ void _set_jet(Grid const &grid, PseudoJetExt &jet, Algorithm algo) {
  auto pt2 = jet.px * jet.px + jet.py * jet.py;
  jet.isJet = false;

  if (pt2 == 0.0) {
    jet.phi = 0.0;
  } else {
    jet.phi = std::atan2(jet.py, jet.px);
    if (jet.phi < 0.0) {
      jet.phi += (2 * M_PI);
    }
    // this should never happen !
    // can happen if phi=-|eps<1e-15| ?
    if (jet.phi >= (2 * M_PI)) {
      jet.phi -= (2 * M_PI);
    }
  }
  if (jet.E == std::abs(jet.pz) and pt2 == 0) {
    // Point has infinite rapidity -- convert that into a very large
    // number, but in such a way that different 0-pt momenta will have
    // different rapidities (so as to lift the degeneracy between
    // them) [this can be relevant at parton-level]
    double MaxRapHere = MaxRap + std::abs(jet.pz);
    if (jet.pz >= 0.0) {
      jet.rap = MaxRapHere;
    } else {
      jet.rap = -MaxRapHere;
    }
  } else {
    // get the rapidity in a way that's modestly insensitive to roundoff
    // error when things pz,E are large (actually the best we can do without
    // explicit knowledge of mass)
    double effective_m2 = ::max(0.0, (jet.E + jet.pz) * (jet.E - jet.pz) - pt2);  // force non tachyonic mass
    double E_plus_pz = jet.E + std::abs(jet.pz);                                  // the safer of p+, p-
    // p+/p- = (p+ p-) / (p-)^2 = (kt^2+m^2)/(p-)^2
    jet.rap = 0.5 * std::log((pt2 + effective_m2) / (E_plus_pz * E_plus_pz));
    if (jet.pz > 0) {
      jet.rap = -jet.rap;
    }
  }

  // set the "weight" used depending on the jet algorithm
  switch (algo) {
    case Algorithm::Kt:
      jet.diB = pt2;
      break;

    case Algorithm::CambridgeAachen:
      jet.diB = 1.;
      break;

    case Algorithm::AntiKt:
      jet.diB = safe_inverse(pt2);
      break;
  }

  // set the grid coordinates
  jet.i = grid.i(jet.rap);
  jet.j = grid.j(jet.phi);
}

__device__ double plain_distance(const PseudoJetExt &p1, const PseudoJetExt &p2) {
  double dphi = std::abs(p1.phi - p2.phi);
  if (dphi > M_PI) {
    dphi = (2 * M_PI) - dphi;
  }
  double drap = p1.rap - p2.rap;
  return (dphi * dphi + drap * drap);
}

__device__ Dist yij_distance(const PseudoJetExt *pseudojets,
                             ParticleIndexType i,
                             ParticleIndexType j,
                             double one_over_r2) {
  if (i > j) {
    ::swap(i, j);
  }

  Dist d;
  d.i = i;
  d.j = j;

  if (i == j) {
    d.distance = pseudojets[i].diB;
  } else {
    d.distance = min(pseudojets[i].diB, pseudojets[j].diB) * plain_distance(pseudojets[i], pseudojets[j]) * one_over_r2;
  }

  return d;
}

__device__ Dist minimum_in_cell(Grid const &grid,
                                const PseudoJetExt *pseudojets,
                                Dist min,
                                const ParticleIndexType tid,  // jet index
                                const GridIndexType i,        // cell coordinates
                                const GridIndexType j,
                                double one_over_r2) {
  int k = 0;
  int index = grid.index(i, j);
  ParticleIndexType num = grid.jets[index * grid.n + k];

  Dist temp;
  while (num >= 0) {
    if (tid != num) {
      temp = yij_distance(pseudojets, tid, num, one_over_r2);

      if (temp.distance < min.distance)
        min = temp;
    }

    k++;
    num = grid.jets[index * grid.n + k];
  }

  return min;
}

__device__ void remove_from_grid(Grid const &grid, ParticleIndexType jet, const PseudoJetExt &p) {
  // Remove an element from a grid cell, and shift all following elements to fill the gap
  int index = grid.index(p.i, p.j);
  int first, last;
  for (int k = 0; k < grid.n; ++k) {
    ParticleIndexType num = grid.jets[index * grid.n + k];
    if (num == jet) {
      first = k;
    } else if (num == -1) {
      last = k;
      break;
    }
    // FIXME handle the case where the jet is not found
    // FIXME handle the case where the cell is full
  }
  if (first != last - 1) {
    grid.jets[index * grid.n + first] = grid.jets[index * grid.n + last - 1];
  }
  // set the last entry to invalid
  grid.jets[index * grid.n + last - 1] = -1;
}

__device__ void add_to_grid(Grid const &grid, ParticleIndexType jet, const PseudoJetExt &p) {
  // Add a jet as the last element of a grid cell
  int index = grid.index(p.i, p.j);
  for (int k = 0; k < grid.n; ++k) {
    // if the k-th element is -1, replace it with the jet id
    if (atomicCAS(&grid.jets[index * grid.n + k], -1, jet) == -1) {
      break;
    }
    // FIXME handle the case where the cell is full
  }
}

__device__ ParticleIndexType &jet_in_grid(Grid const &grid, ParticleIndexType jet, const PseudoJetExt &p) {
  // Return a reference to the element that identifies a jet in a grid cell
  int index = grid.index(p.i, p.j);
  for (int k = 0; k < grid.n; ++k) {
    ParticleIndexType num = grid.jets[index * grid.n + k];
    if (num == jet) {
      return grid.jets[index * grid.n + k];
    }
  }
  // handle the case where the jet is not found
  return grid.jets[grid.max_i * grid.max_j * grid.n];
}
#pragma endregion

#pragma region kernels
__global__ void set_points(Grid grid, PseudoJetExt *particles, const ParticleIndexType n, Algorithm algo) {
  int start = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;

  for (int tid = start; tid < n; tid += stride) {
    _set_jet(grid, particles[tid], algo);
    //printf("particle %3d has (rap,phi,pT) = (%f,%f,%f) and cell (i,j) = (%d,%d)\n", tid, p.rap, p.phi, sqrt(p.diB), p.i, p.j);
    add_to_grid(grid, tid, particles[tid]);
  }
}

__global__ void reduce_recombine(
    Grid grid, PseudoJetExt *pseudojets, Dist *min_dists, ParticleIndexType n, Algorithm algo, const float r) {
  extern __shared__ Dist sdata[];

  const double one_over_r2 = 1. / (r * r);

  for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
    min_dists[tid].i = -3;
    min_dists[tid].j = -1;
  }
  Dist min;
  min.i = -4;
  min.j = -4;
  while (n > 0) {
    for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
      auto p = pseudojets[tid];
      Dist local_min = min_dists[tid];
      if (local_min.i == -3 or local_min.j == min.i or local_min.j == min.j or local_min.i == min.i or
          local_min.i == min.j or local_min.i >= n or local_min.j >= n) {
        local_min = yij_distance(pseudojets, tid, tid, one_over_r2);
        local_min = minimum_in_cell(grid, pseudojets, local_min, tid, p.i, p.j, one_over_r2);

        bool right = p.i + 1 < grid.max_i;
        bool left = p.i > 0;

        // Right
        if (right) {
          local_min = minimum_in_cell(grid, pseudojets, local_min, tid, p.i + 1, p.j, one_over_r2);
        }

        // Left
        if (left) {
          local_min = minimum_in_cell(grid, pseudojets, local_min, tid, p.i - 1, p.j, one_over_r2);
        }

        // check if (p.j + 1) would overflow grid.max_j
        GridIndexType j = (p.j + 1 < grid.max_j) ? p.j + 1 : 0;

        // Up
        local_min = minimum_in_cell(grid, pseudojets, local_min, tid, p.i, j, one_over_r2);

        // Up Right
        if (right) {
          local_min = minimum_in_cell(grid, pseudojets, local_min, tid, p.i + 1, j, one_over_r2);
        }

        // Up Left
        if (left) {
          local_min = minimum_in_cell(grid, pseudojets, local_min, tid, p.i - 1, j, one_over_r2);
        }

        // check if (p.j - 1) would underflow below 0
        j = p.j - 1 >= 0 ? p.j - 1 : grid.max_j - 1;

        // Down
        local_min = minimum_in_cell(grid, pseudojets, local_min, tid, p.i, j, one_over_r2);

        // Down Right
        if (right) {
          local_min = minimum_in_cell(grid, pseudojets, local_min, tid, p.i + 1, j, one_over_r2);
        }

        // Down Left
        if (left) {
          local_min = minimum_in_cell(grid, pseudojets, local_min, tid, p.i - 1, j, one_over_r2);
        }

        min_dists[tid] = local_min;
      }

      sdata[tid] = local_min;
    }
    __syncthreads();

    // find the largest power of 2 smaller than n
    unsigned int width = (1u << 31) >> __clz(n - 1);

    for (unsigned int s = width; s > 0; s >>= 1) {
      for (int tid = threadIdx.x; tid < s and tid < n - s; tid += blockDim.x) {
        if (sdata[tid + s].distance < sdata[tid].distance) {
          sdata[tid] = sdata[tid + s];
        }
      }
      __syncthreads();
    }

    min = sdata[0];
    if (threadIdx.x == 0) {
      //printf("will recombine pseudojets %d and %d with distance %f\n", min.i, min.j, min.distance);
      if (min.i == min.j) {
        // remove the pseudojet at position min.j from the grid and promote it to jet status
        auto jet = pseudojets[min.j];
        remove_from_grid(grid, min.j, jet);
        jet.isJet = true;

        // move the last pseudojet to position min.j
        if (min.j != n - 1) {
          jet_in_grid(grid, n - 1, pseudojets[n - 1]) = min.j;
          pseudojets[min.j] = pseudojets[n - 1];
        }

        // move the jet to the end of the list
        pseudojets[n - 1] = jet;

      } else {
        auto ith = pseudojets[min.i];
        auto jth = pseudojets[min.j];
        remove_from_grid(grid, min.i, ith);
        remove_from_grid(grid, min.j, jth);

        // recombine the two pseudojets
        PseudoJetExt pseudojet;
        pseudojet.px = ith.px + jth.px;
        pseudojet.py = ith.py + jth.py;
        pseudojet.pz = ith.pz + jth.pz;
        pseudojet.E = ith.E + jth.E;
        _set_jet(grid, pseudojet, algo);
        add_to_grid(grid, min.i, pseudojet);
        pseudojets[min.i] = pseudojet;

        // move the last pseudojet to position min.j
        if (min.j != n - 1) {
          jet_in_grid(grid, n - 1, pseudojets[n - 1]) = min.j;
          pseudojets[min.j] = pseudojets[n - 1];
        }
      }
    }
    n--;
    __syncthreads();
  }
}
#pragma endregion

__global__ void init(const PseudoJet *particles, PseudoJetExt *jets, int size) {
  int first = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = first; i < size; i += stride) {
    jets[i].px = particles[i].px;
    jets[i].py = particles[i].py;
    jets[i].pz = particles[i].pz;
    jets[i].E = particles[i].E;
    jets[i].index = particles[i].index;
    jets[i].isJet = particles[i].isJet;
  }
}

__global__ void output(const PseudoJetExt *jets, PseudoJet *particles, int size) {
  int first = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = first; i < size; i += stride) {
    particles[i].px = jets[i].px;
    particles[i].py = jets[i].py;
    particles[i].pz = jets[i].pz;
    particles[i].E = jets[i].E;
    particles[i].index = jets[i].index;
    particles[i].isJet = jets[i].isJet;
  }
}

void cluster(PseudoJet *particles, int size, Algorithm algo, double r) {
#pragma region vectors
  // examples from FastJet span |rap| < 10
  // TODO: make the rap range dynamic, based on the data themselves
  // TODO: make the cell size dynamic, based on the data themselves
  // TODO: try to use __constant__ memory for config
  Grid grid(-10., +10., 0, 2 * M_PI, r, size);
  cudaCheck(cudaMalloc(&grid.jets, sizeof(ParticleIndexType) * grid.max_i * grid.max_j * grid.n));
  cudaCheck(cudaMemset(grid.jets, 0xff, sizeof(ParticleIndexType) * grid.max_i * grid.max_j * grid.n));

  PseudoJetExt *pseudojets;
  cudaCheck(cudaMalloc(&pseudojets, size * sizeof(PseudoJetExt)));

  Dist *d_min_dists_ptr;
  cudaCheck(cudaMalloc(&d_min_dists_ptr, sizeof(Dist) * size));
#pragma endregion

#pragma region kernel_launches
  cudaCheck(cudaDeviceSynchronize());
  // copy the particles from the input buffer to the pseudojet structures
  init<<<8, 512>>>(particles, pseudojets, size);
  cudaCheck(cudaGetLastError());

  // TODO: move to helper function
  int blockSize;
  int minGridSize;
  cudaCheck(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, set_points, 0, 0));
  int gridSize = std::min((size + blockSize - 1) / blockSize, minGridSize);
  // set jets into points
  set_points<<<gridSize, blockSize>>>(grid, pseudojets, size, algo);
  cudaCheck(cudaDeviceSynchronize());

  {
    cudaFuncAttributes attr;
    cudaCheck(cudaFuncGetAttributes(&attr, reduce_recombine));
    /*
    std::cout << "binaryVersion:             " << attr.binaryVersion << std::endl;
    std::cout << "cacheModeCA:               " << attr.cacheModeCA << std::endl;
    std::cout << "constSizeBytes:            " << attr.constSizeBytes << std::endl;
    std::cout << "localSizeBytes:            " << attr.localSizeBytes << std::endl;
    std::cout << "maxDynamicSharedSizeBytes: " << attr.maxDynamicSharedSizeBytes << std::endl;
    std::cout << "maxThreadsPerBlock:        " << attr.maxThreadsPerBlock << std::endl;
    std::cout << "numRegs:                   " << attr.numRegs << std::endl;
    std::cout << "preferredShmemCarveout:    " << attr.preferredShmemCarveout << std::endl;
    std::cout << "ptxVersion:                " << attr.ptxVersion << std::endl;
    std::cout << "sharedSizeBytes:           " << attr.sharedSizeBytes << std::endl;
    */
    int gridSize = 1;
    //int blockSize = 1;
    int blockSize = std::min(size, attr.maxThreadsPerBlock);
    int sharedMemory = sizeof(Dist) * size;

    reduce_recombine<<<gridSize, blockSize, sharedMemory>>>(grid, pseudojets, d_min_dists_ptr, size, algo, r);
    cudaCheck(cudaGetLastError());
  }
  cudaCheck(cudaDeviceSynchronize());

  // copy the clustered jets back to the input buffer
  output<<<8, 512>>>(pseudojets, particles, size);
  cudaCheck(cudaDeviceSynchronize());
#pragma endregion

  cudaCheck(cudaFree(pseudojets));
  cudaCheck(cudaFree(grid.jets));
  cudaCheck(cudaFree(d_min_dists_ptr));
}
