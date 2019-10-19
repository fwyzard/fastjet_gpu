#include <cmath>
#include <limits>

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
namespace cg = cooperative_groups;

#include "PseudoJet.h"
#include "cluster.h"
#include "cudaCheck.h"
#include "launch.h"

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

struct Cell {
  GridIndexType i;
  GridIndexType j;
};

struct Grid {
  const double r;
  const double min_rap;
  const double max_rap;
  const double min_phi;
  const double max_phi;
  const GridIndexType max_i;
  const GridIndexType max_j;
  const ParticleIndexType n;

  ParticleIndexType *jets;
  Dist *minimum;
  Dist *neighbours;

  // TODO use a smaller grid size (esimate from distributions in data/mc)
  // TODO usa a SoA
  __host__ Grid(double min_rap_, double max_rap_, double min_phi_, double max_phi_, double r_, ParticleIndexType n_)
      : r((2 * M_PI) / (int)((2 * M_PI) / r_)),  // round up the grid size to have an integer number of cells in phi
        min_rap(min_rap_),
        max_rap(max_rap_),
        min_phi(min_phi_),
        max_phi(max_phi_),
        max_i((GridIndexType)(((max_rap - min_rap) / r))),
        max_j((GridIndexType)(((max_phi - min_phi) / r))),
        n(n_),
        jets(nullptr),
        minimum(nullptr),
        neighbours(nullptr)
  {}

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

  __host__ __device__ constexpr inline int size() const { return (int) max_i * max_j; }

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

__device__ Dist minimum_pair_in_cell(Grid const &grid,
                                const PseudoJetExt *pseudojets,
                                const GridIndexType i,        // cell coordinates
                                const GridIndexType j,
                                double one_over_r2) {
  int index = grid.index(i, j);
  Dist min{ std::numeric_limits<double>::infinity(), -1, -1 };

  int k = 0;
  GridIndexType first = grid.jets[index * grid.n + k];
  while (first >= 0) {
    for (int l = 0; l <= k; ++l) {
      GridIndexType second = grid.jets[index * grid.n + l];
      auto temp = yij_distance(pseudojets, first, second, one_over_r2);
      if (temp.distance < min.distance)
        min = temp;
    }
    ++k;
    first = grid.jets[index * grid.n + k];
  }

  return min;
}

__device__ Dist minimum_pair_in_cells(Grid const &grid,
                                const PseudoJetExt *pseudojets,
                                const GridIndexType first_i,
                                const GridIndexType first_j,
                                const GridIndexType second_i,
                                const GridIndexType second_j,
                                double one_over_r2) {
  int first_index = grid.index(first_i, first_j);
  int second_index = grid.index(second_i, second_j);
  Dist min{ std::numeric_limits<double>::infinity(), -1, -1 };

  int k = 0;
  GridIndexType first = grid.jets[first_index * grid.n + k];
  while (first >= 0) {
    int l = 0;
    GridIndexType second = grid.jets[second_index * grid.n + l];
    while (second >= 0) {
      auto temp = yij_distance(pseudojets, first, second, one_over_r2);
      if (temp.distance < min.distance)
        min = temp;
      ++l;
      second = grid.jets[second_index * grid.n + l];
    }
    ++k;
    first = grid.jets[first_index * grid.n + k];
  }

  return min;
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
    temp = yij_distance(pseudojets, tid, num, one_over_r2);
    if (temp.distance < min.distance)
      min = temp;

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
#pragma endregion

#pragma region kernels
__global__ void set_jets_coordiinates(Grid grid, PseudoJetExt *particles, const ParticleIndexType n, Algorithm algo) {
  int start = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;

  for (int tid = start; tid < n; tid += stride) {
    _set_jet(grid, particles[tid], algo);
    //printf("particle %3d has (rap,phi,pT) = (%f,%f,%f) and cell (i,j) = (%d,%d)\n", tid, p.rap, p.phi, sqrt(p.diB), p.i, p.j);
  }
}

__global__ void set_jets_to_grid(Grid grid, PseudoJetExt *particles, const ParticleIndexType n, Algorithm algo) {
  int start = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;

  for (int tid = start; tid < n; tid += stride) {
    add_to_grid(grid, tid, particles[tid]);
  }
}

__global__ void compute_initial_distances(Grid grid, PseudoJetExt *pseudojets, const ParticleIndexType n, double r) {
  const double one_over_r2 = 1. / (r * r);
  const Dist none { std::numeric_limits<double>::infinity(), -1, -1 };

  int start = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;

  for (int index = start; index < grid.max_i * grid.max_j; index += stride) {
    GridIndexType i = index / grid.max_j;
    GridIndexType j = index % grid.max_j;
    auto jet = grid.jets[index * grid.n];

    // check if the cell is empty
    if (jet == -1) {
      for (int k = 0; k < 9; ++k)
        grid.neighbours[index * 9 + k] = none;
      grid.minimum[index] = none;
    } else {
      // FIXME use 9 threads ?
      GridIndexType j_plus = (j + 1 < grid.max_j) ? j + 1 : 0;
      GridIndexType j_minus = (j - 1 >= 0) ? j - 1 : grid.max_j - 1;
      auto min = none;
      auto tmp = none;
      min = minimum_pair_in_cell(grid, pseudojets, i, j, one_over_r2);
      grid.neighbours[index * 9 + 4] = min;
      tmp = minimum_pair_in_cells(grid, pseudojets, i, j, i, j_minus, one_over_r2);
      grid.neighbours[index * 9 + 3] = tmp;
      if (tmp.distance < min.distance) min = tmp;
      tmp = minimum_pair_in_cells(grid, pseudojets, i, j, i, j_plus, one_over_r2);
      grid.neighbours[index * 9 + 5] = tmp;
      if (tmp.distance < min.distance) min = tmp;
      if (i - 1 >= 0) {
        tmp = minimum_pair_in_cells(grid, pseudojets, i, j, i-1, j_minus, one_over_r2);
        grid.neighbours[index * 9 + 0] = tmp;
        if (tmp.distance < min.distance) min = tmp;
        tmp = minimum_pair_in_cells(grid, pseudojets, i, j, i-1, j, one_over_r2);
        grid.neighbours[index * 9 + 1] = tmp;
        if (tmp.distance < min.distance) min = tmp;
        tmp = minimum_pair_in_cells(grid, pseudojets, i, j, i-1, j_plus, one_over_r2);
        grid.neighbours[index * 9 + 2] = tmp;
        if (tmp.distance < min.distance) min = tmp;
      } else {
        grid.neighbours[index * 9 + 0] = none;
        grid.neighbours[index * 9 + 1] = none;
        grid.neighbours[index * 9 + 2] = none;
      }
      if (i + 1 < grid.max_i) {
        tmp = minimum_pair_in_cells(grid, pseudojets, i, j, i+1, j_minus, one_over_r2);
        grid.neighbours[index * 9 + 6] = tmp;
        if (tmp.distance < min.distance) min = tmp;
        tmp = minimum_pair_in_cells(grid, pseudojets, i, j, i+1, j, one_over_r2);
        grid.neighbours[index * 9 + 7] = tmp;
        if (tmp.distance < min.distance) min = tmp;
        tmp = minimum_pair_in_cells(grid, pseudojets, i, j, i+1, j_plus, one_over_r2);
        grid.neighbours[index * 9 + 8] = tmp;
        if (tmp.distance < min.distance) min = tmp;
      } else {
        grid.neighbours[index * 9 + 6] = none;
        grid.neighbours[index * 9 + 7] = none;
        grid.neighbours[index * 9 + 8] = none;
      }
      grid.minimum[index] = min;
    }
  }
}

constexpr const int n_neighbours = 9;                            // self, plus 8 neighbours
constexpr const int n_affected = 3;                              // 3 possibly affected cells
constexpr const int active_threads = n_neighbours * n_affected;  // 1 cell + 8 neighbours, times 3 possibly affected cells

// reduce_recombine(...) must be called with at least active_threads (27) threads
__global__ void reduce_recombine(
    Grid grid, PseudoJetExt *pseudojets, ParticleIndexType n, Algorithm algo, const float r) {
  extern __shared__ Dist minima[];

  int start = threadIdx.x;
  int stride = blockDim.x;

  const double one_over_r2 = 1. / (r * r);
  const Dist none { std::numeric_limits<double>::infinity(), -1, -1 };

  while (true) {
    // copy the minimum distances into shared memory
    for (int index = start; index < grid.max_i * grid.max_j; index += stride) {
      minima[index] = grid.minimum[index];
    }
    __syncthreads();

    // find the largest power of 2 smaller than the grid size
    unsigned int width = (1u << 31) >> __clz(grid.max_i * grid.max_j - 1);

    // find to global minimum
    for (unsigned int s = width; s > 0; s >>= 1) {
      for (int tid = threadIdx.x; tid < s and tid + s < grid.max_i * grid.max_j; tid += blockDim.x) {
        if (minima[tid + s].distance < minima[tid].distance) {
          minima[tid] = minima[tid + s];
        }
      }
      __syncthreads();
    }

    // promote or recombine the minimum pseudojet(s)
    Dist min = minima[0];
    __shared__ Cell affected[n_affected];

    if (threadIdx.x == 0) {
      if (min.i == min.j) {
        // remove the pseudojet at position min.j from the grid and promote it to jet status
        //printf("will promote pseudojet %d (%d,%d) with distance %f\n", min.j, pseudojets[min.j].i, pseudojets[min.j].j, min.distance);
        pseudojets[min.j].isJet = true;
        auto jet = pseudojets[min.j];
        remove_from_grid(grid, min.j, jet);
        affected[0] = { jet.i, jet.j };
        affected[1] = { -1, -1 };
        affected[2] = { -1, -1 };
      } else {
        //printf("will recombine pseudojets %d (%d,%d) and %d (%d,%d) with distance %f\n", min.i, pseudojets[min.i].i, pseudojets[min.i].j, min.j, pseudojets[min.j].i, pseudojets[min.j].j, min.distance);
        auto ith = pseudojets[min.i];
        auto jth = pseudojets[min.j];
        remove_from_grid(grid, min.i, ith);
        remove_from_grid(grid, min.j, jth);
        affected[0] = { ith.i, ith.j };
        if (jth.i != ith.i or jth.j != ith.j) {
          affected[1] = { jth.i, jth.j };
        } else {
          affected[1] = { -1, -1 };
        }

        // recombine the two pseudojets
        PseudoJetExt pseudojet;
        pseudojet.px = ith.px + jth.px;
        pseudojet.py = ith.py + jth.py;
        pseudojet.pz = ith.pz + jth.pz;
        pseudojet.E = ith.E + jth.E;
        _set_jet(grid, pseudojet, algo);
        add_to_grid(grid, min.i, pseudojet);
        pseudojets[min.i] = pseudojet;
        if ((pseudojet.i != ith.i or pseudojet.j != ith.j) and
            (pseudojet.i != jth.i or pseudojet.j != jth.j)) {
          affected[2] = { pseudojet.i, pseudojet.j };
        } else {
          affected[2] = { -1, -1 };
        }
      }
    }
    __syncthreads();

    if (--n == 0)
      break;

    int tid = start;
    if (tid < active_threads) {
      int self = tid / n_neighbours;   // potentially affected cell (0..2)
      int cell = tid % n_neighbours;   // neighbour id (0..8)
      GridIndexType i = affected[self].i;
      GridIndexType j = affected[self].j;

      // consider only the affected cells
      if (i >= 0 and j >= 0) {

        auto g = cg::coalesced_threads();
        const int index = grid.index(i, j);

        // check if the cell is empty
        bool empty = (grid.jets[index * grid.n] == -1);

        // evaluate the neighbouring cells
        const int delta_i = cell / 3 - 1;
        const int delta_j = cell % 3 - 1;
        const GridIndexType other_i = i + delta_i;
        const GridIndexType other_j = (j + delta_j + grid.max_j) % grid.max_j;
        const bool central = (cell == 4);
        const bool outside = other_i < 0 or other_i >= grid.max_i;

        // update the local minima
        if (central) {
          grid.neighbours[index * n_neighbours + cell] = empty ? none : minimum_pair_in_cell(grid, pseudojets, i, j, one_over_r2);
        } else if (outside) {
          grid.neighbours[index * n_neighbours + cell] = none;
        } else {
          auto tmp = empty ? none : minimum_pair_in_cells(grid, pseudojets, i, j, other_i, other_j, one_over_r2);
          grid.neighbours[index * n_neighbours + cell] = tmp;
          grid.neighbours[grid.index(other_i, other_j) * n_neighbours + (n_neighbours - 1) - cell] = tmp;
        }

        // synchronise the active threads
        g.sync();

        // update the minimum in neighbouring cells
        if (not outside) {
          const int other = grid.index(other_i, other_j);
          auto min = none;
          for (int k = 0; k < n_neighbours; ++k) {
            auto tmp = grid.neighbours[other * n_neighbours + k];
            if (tmp.distance < min.distance) min = tmp;
          }
          grid.minimum[other] = min;
        }
      }
    }

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
  cudaCheck(cudaMalloc(&grid.minimum, sizeof(Dist) * grid.max_i * grid.max_j));
  cudaCheck(cudaMalloc(&grid.neighbours, sizeof(Dist) * grid.max_i * grid.max_j * 9));

  PseudoJetExt *pseudojets;
  cudaCheck(cudaMalloc(&pseudojets, size * sizeof(PseudoJetExt)));
#pragma endregion

#pragma region kernel_launches
  LaunchParameters l;

  // copy the particles from the input buffer to the pseudojet structures
  l = estimateMinimalGrid(init, size);
  init<<<l.gridSize, l.blockSize>>>(particles, pseudojets, size);
  cudaCheck(cudaGetLastError());

  // compute the jets cilindrical coordinates and grid indices
  l = estimateMinimalGrid(set_jets_coordiinates, size);
  set_jets_coordiinates<<<l.gridSize, l.blockSize>>>(grid, pseudojets, size, algo);
  cudaCheck(cudaGetLastError());

  // sort the inputs according to their grid coordinates and "beam" clustering distance
  thrust::sort(thrust::device, pseudojets, pseudojets + size, [] __device__(auto const &a, auto const &b) {
    return (a.i < b.i) or (a.i == b.i and a.j < b.j) or (a.i == b.i and a.j == b.j and a.diB < b.diB);
  });

  // organise the jets in the grid
  l = estimateMinimalGrid(set_jets_to_grid, size);
  set_jets_to_grid<<<l.gridSize, l.blockSize>>>(grid, pseudojets, size, algo);
  cudaCheck(cudaGetLastError());

  // compute the minimum distances in all grid cells
  l = estimateMinimalGrid(compute_initial_distances, grid.size());
  compute_initial_distances<<<l.gridSize, l.blockSize>>>(grid, pseudojets, size, r);
  cudaCheck(cudaGetLastError());

  // recombine the particles into jets
  l = estimateSingleBlock(reduce_recombine, grid.size());
  int sharedMemory = sizeof(Dist) * grid.size();
  reduce_recombine<<<l.gridSize, l.blockSize, sharedMemory>>>(grid, pseudojets, size, algo, r);
  cudaCheck(cudaGetLastError());

  // copy the clustered jets back to the input buffer
  l = estimateMinimalGrid(output, size);
  output<<<l.gridSize, l.blockSize>>>(pseudojets, particles, size);
  cudaCheck(cudaGetLastError());
#pragma endregion

  cudaCheck(cudaFree(pseudojets));
  cudaCheck(cudaFree(grid.jets));
  cudaCheck(cudaFree(grid.minimum));
  cudaCheck(cudaFree(grid.neighbours));
}
