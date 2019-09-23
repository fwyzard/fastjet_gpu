#include <cmath>

#include <cuda_runtime.h>

#include "PseudoJet.h"
#include "cluster.h"
#include "cudaCheck.h"

using namespace std;

#pragma region consts
const double MaxRap = 1e5;
#pragma endregion

#pragma region struct

using GridIndexType     = int;
using ParticleIndexType = int;

struct EtaPhi {
  double eta;
  double phi;
  double diB;
  GridIndexType box_i;
  GridIndexType box_j;
};

struct Dist {
  double distance;
  ParticleIndexType i;
  ParticleIndexType j;
};

struct Grid {
  double min_eta;
  double max_eta;
  double min_phi;
  double max_phi;
  double r;
  GridIndexType max_i;
  GridIndexType max_j;
  int n;

  // TODO use a smaller grid size (esimate from distributions in data/mc)
  // TODO usa a SoA
  __host__ __device__ Grid(double min_eta, double max_eta, double min_phi, double max_phi, double r, int n)
      : min_eta(min_eta),
        max_eta(max_eta),
        min_phi(min_phi),
        max_phi(min_phi),
        r(r),
        max_i((GridIndexType) (((max_eta - min_eta) / r) + 1)),
        max_j((GridIndexType) (((max_phi - min_phi) / r) + 1)),
        n(n) {}

  __host__ __device__ constexpr inline GridIndexType i(double eta) const { return (GridIndexType) ((eta - min_eta) / r); }

  __host__ __device__ constexpr inline GridIndexType j(double phi) const { return (GridIndexType) ((phi - min_phi) / r); }

  __host__ __device__ constexpr inline double eta_min(GridIndexType i) const { return min_eta + r * i; }

  __host__ __device__ constexpr inline double eta_max(GridIndexType i) const { return min_eta + r * (i + 1); }

  __host__ __device__ constexpr inline double phi_min(GridIndexType j) const { return min_phi + r * j; }

  __host__ __device__ constexpr inline double phi_max(GridIndexType j) const { return min_phi + r * (j + 1); }

  __host__ __device__ constexpr inline int index(GridIndexType i, GridIndexType j) const { return (int) max_j * i + j; }

  __host__ __device__ constexpr inline int offset(GridIndexType i, GridIndexType j) const { return index(i, j) * n; }
};
#pragma endregion

#pragma region device_functions
__host__ __device__ EtaPhi _set_jet(PseudoJet &jet) {
  EtaPhi point;

  point.diB = jet.px * jet.px + jet.py * jet.py;
  jet.isJet = false;

  if (point.diB == 0.0) {
    point.phi = 0.0;
  } else {
    point.phi = std::atan2(jet.py, jet.px);
    if (point.phi < 0.0) {
      point.phi += (2 * M_PI);
    }
    // this should never happen !
    // can happen if phi=-|eps<1e-15| ?
    if (point.phi >= (2 * M_PI)) {
      point.phi -= (2 * M_PI);
    }
  }
  if (jet.E == std::abs(jet.pz) and point.diB == 0) {
    // Point has infinite rapidity -- convert that into a very large
    // number, but in such a way that different 0-pt momenta will have
    // different rapidities (so as to lift the degeneracy between
    // them) [this can be relevant at parton-level]
    double MaxRapHere = MaxRap + std::abs(jet.pz);
    if (jet.pz >= 0.0) {
      point.eta = MaxRapHere;
    } else {
      point.eta = -MaxRapHere;
    }
  } else {
    // get the rapidity in a way that's modestly insensitive to roundoff
    // error when things pz,E are large (actually the best we can do without
    // explicit knowledge of mass)
    double effective_m2 = ::max(0.0, (jet.E + jet.pz) * (jet.E - jet.pz) - point.diB);  // force non tachyonic mass
    double E_plus_pz = jet.E + std::abs(jet.pz);                                        // the safer of p+, p-
    // p+/p- = (p+ p-) / (p-)^2 = (kt^2+m^2)/(p-)^2
    point.eta = 0.5 * std::log((point.diB + effective_m2) / (E_plus_pz * E_plus_pz));
    if (jet.pz > 0) {
      point.eta = -point.eta;
    }
  }

  return point;
}

__device__ double plain_distance(const EtaPhi &p1, const EtaPhi &p2) {
  double dphi = std::abs(p1.phi - p2.phi);
  if (dphi > M_PI) {
    dphi = (2 * M_PI) - dphi;
  }
  double drap = p1.eta - p2.eta;
  return (dphi * dphi + drap * drap);
}

__device__ double safe_inverse(double x) {
  return (x > 1e-300) ? (1.0 / x) : 1e300;
}

__device__ Dist yij_distance(const EtaPhi *points, ParticleIndexType i, ParticleIndexType j, Scheme scheme, double one_over_r2) {
  if (i > j) {
    auto t = i;
    i = j;
    j = t;
  }

  Dist d;
  d.i = i;
  d.j = j;

  switch (scheme) {
    case Scheme::Kt:
      if (i == j) {
        d.distance = points[i].diB;
      } else {
        d.distance = min(points[i].diB, points[j].diB) * plain_distance(points[i], points[j]) * one_over_r2;
      }
      break;

    case Scheme::CambridgeAachen:
      if (i == j) {
        d.distance = 1.;
      } else {
        d.distance = plain_distance(points[i], points[j]) * one_over_r2;
      }
      break;

    // TODO: store 1/diB instead of diB
    case Scheme::AntiKt:
      if (i == j) {
        d.distance = safe_inverse(points[i].diB);
      } else {
        d.distance = min(
            safe_inverse(points[i].diB),
            safe_inverse(points[j].diB)) * plain_distance(points[i], points[j]) * one_over_r2;
      }
      break;
  }

  return d;
}

__device__ Dist minimum_in_cell(Grid const &config,
                                const ParticleIndexType *grid,
                                const EtaPhi *points,
                                const PseudoJet *jets,
                                Dist min,
                                const ParticleIndexType tid,    // jet index
                                const GridIndexType i,          // cell coordinates
                                const GridIndexType j,
                                Scheme scheme,                  // recombination scheme (Kt, CambridgeAachen,  AntiKt)
                                double one_over_r2) {
  int k = 0;
  int offset = config.offset(i, j);
  ParticleIndexType num = grid[offset + k];

  Dist temp;
  while (num >= 0) {
    if (tid != num) {
      temp = yij_distance(points, tid, num, scheme, one_over_r2);

      if (temp.distance < min.distance)
        min = temp;
    }

    k++;
    num = grid[offset + k];
  }

  return min;
}

__device__ void remove_from_grid(Grid const &config, ParticleIndexType *grid, PseudoJet &jet, const EtaPhi &p) {
  // Remove an element from a grid cell, and shift all following elements to fill the gap
  int k = 0;
  int offset = config.offset(p.box_i, p.box_j);
  ParticleIndexType num = grid[offset + k];
  bool shift = false;

  while (num != -1) {
    if (jet.index == num)
      shift = true;
    if (shift) {
      grid[offset + k] = grid[offset + k + 1];
    }
    k++;

    num = grid[offset + k];
  }
}

__device__ void add_to_grid(Grid const &config, ParticleIndexType *grid, const PseudoJet &jet, const EtaPhi &p) {
  // Add a jet as the last element of a grid cell
  int k = 0;
  int offset = config.offset(p.box_i, p.box_j);
  int num = grid[offset + k];

  while (true) {
    num = grid[offset + k];
    if (num == -1) {
      grid[offset + k] = jet.index;         // FIXME add a check that jet.index fits in ParticleIndexType
      grid[offset + k + 1] = -1;
      break;
    }
    k++;
  }
}
#pragma endregion

#pragma region kernels
__global__ void set_points(Grid config, PseudoJet *jets, EtaPhi *points, const int n) {
  int start = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;

  for (int tid = start; tid < n; tid += stride) {
    EtaPhi p = _set_jet(jets[tid]);
    p.box_i = config.i(p.eta);
    p.box_j = config.j(p.phi);
    points[tid] = p;
    //printf("particle %3d has (eta,phi,pT) = (%f,%f,%f) and cell (i,j) = (%d,%d)\n", tid, p.eta, p.phi, sqrt(p.diB), p.box_i, p.box_j);
  }
}

__global__ void set_grid(Grid config, ParticleIndexType *grid, const EtaPhi *points, const PseudoJet *jets, const int n) {
  GridIndexType tid = threadIdx.x;
  GridIndexType bid = blockIdx.x;

  int k = 0;
  EtaPhi p;

  int offset = config.offset(bid, tid);

  for (int i = 0; i < n; i++) {
    p = points[i];

    if (p.box_i == bid and p.box_j == tid) {
      grid[offset + k] = jets[i].index;     // FIXME add a check that jet.index fits in ParticleIndexType
      k++;
    }
  }

  grid[offset + k] = -1;
  //printf("cell (%d,%d) has %d elements\n", tid, bid, k);
}

__global__ void reduce_recombine(
    Grid config, ParticleIndexType *grid, EtaPhi *points, PseudoJet *jets, Dist *min_dists, int n, Scheme scheme, const float r) {
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
      EtaPhi p = points[tid];
      Dist local_min = min_dists[tid];
      if (local_min.i == -3 or local_min.j == min.i or local_min.j == min.j or local_min.i == min.i or
          local_min.i == min.j or local_min.i >= n or local_min.j >= n) {

        min = yij_distance(points, tid, tid, scheme, one_over_r2);
        min = minimum_in_cell(config, grid, points, jets, min, tid, p.box_i, p.box_j, scheme, one_over_r2);

        bool right = true;
        bool left = true;
        bool up = true;
        bool down = true;

        /*
        EtaPhi bp;
        bp.eta = config.eta_max(p.box_i);
        bp.phi = p.phi;
        if (min.distance < plain_distance(p, bp)) {
          right = false;
        }

        bp.eta = config.eta_min(p.box_i);
        bp.phi = p.phi;
        if (min.distance < plain_distance(p, bp)) {
          left = false;
        }

        bp.eta = p.eta;
        bp.phi = p.box_j + 1 <= config.max_j ? (p.box_j + 1) * r : 0;
        if (min.distance < plain_distance(p, bp)) {
          up = false;
        }

        bp.eta = p.eta;
        bp.phi = p.box_j - 1 >= 0 ? p.box_j * r : (config.max_j - 1) * r;
        if (min.distance < plain_distance(p, bp) and p.box_j - 1 >= 0) {
          down = false;
        }
        */

        // Right
        if (p.box_i + 1 < config.max_i and right) {
          min = minimum_in_cell(config, grid, points, jets, min, tid, p.box_i + 1, p.box_j, scheme, one_over_r2);
        }

        // Left
        if (p.box_i - 1 >= 0 and left) {
          min = minimum_in_cell(config, grid, points, jets, min, tid, p.box_i - 1, p.box_j, scheme, one_over_r2);
        }

        // check if above config.max_j
        GridIndexType j = (p.box_j + 1 <= config.max_j) ? p.box_j + 1 : 0;

        // Up
        if (up) {
          min = minimum_in_cell(config, grid, points, jets, min, tid, p.box_i, j, scheme, one_over_r2);

          // Up Right
          if (p.box_i + 1 < config.max_i and right) {
            min = minimum_in_cell(config, grid, points, jets, min, tid, p.box_i + 1, j, scheme, one_over_r2);
          }

          // Up Left
          if (p.box_i - 1 >= 0 and left) {
            min = minimum_in_cell(config, grid, points, jets, min, tid, p.box_i - 1, j, scheme, one_over_r2);
          }
        }

        // check if bellow 0
        j = p.box_j - 1 >= 0 ? p.box_j - 1 : config.max_j - 1;

        // Down
        if (down) {
          min = minimum_in_cell(config, grid, points, jets, min, tid, p.box_i, j, scheme, one_over_r2);

          // Down Right
          if (p.box_i + 1 < config.max_i and right) {
            min = minimum_in_cell(config, grid, points, jets, min, tid, p.box_i + 1, j, scheme, one_over_r2);
          }

          // Down Left
          if (p.box_i - 1 >= 0 and left) {
            min = minimum_in_cell(config, grid, points, jets, min, tid, p.box_i - 1, j, scheme, one_over_r2);
          }

          if (p.box_j - 1 < 0) {
            // Down Down
            min = minimum_in_cell(config, grid, points, jets, min, tid, p.box_i, j - 1, scheme, one_over_r2);

            // Down Down Right
            if (p.box_i + 1 < config.max_i and right) {
              min = minimum_in_cell(config, grid, points, jets, min, tid, p.box_i + 1, j - 1, scheme, one_over_r2);
            }

            // Down Down Left
            if (p.box_i - 1 >= 0 and left) {
              min = minimum_in_cell(config, grid, points, jets, min, tid, p.box_i - 1, j - 1, scheme, one_over_r2);
            }
          }
        }

        if (min.i > min.j) {
          auto t = min.i;
          min.i = min.j;
          min.j = t;
        }

        min_dists[tid] = min;
      }

      // FIXME: why an extra copy ?
      sdata[tid] = min_dists[tid];
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

    // Minimum of the row
    // if (tid == 0) {
    // min_dists[k] = sdata[0];
    // }

    min = sdata[0];
    if (threadIdx.x == 0) {
      //printf("will recombine pseudojets %d and %d with distance %f\n", min.i, min.j, min.distance);
      PseudoJet jet_i, jet_j;

      EtaPhi p1, p2;
      if (min.i == min.j) {
        jet_j = jets[min.j];
        p1 = points[min.j];
        remove_from_grid(config, grid, jet_j, p1);
        if (min.j != n - 1)
          remove_from_grid(config, grid, jets[n - 1], points[n - 1]);

        jets[min.j] = jets[n - 1];
        points[min.j] = points[n - 1];

        jets[min.j].index = min.j;

        jet_j.isJet = true;
        jet_j.index = n - 1;
        jets[n - 1] = jet_j;
        points[n - 1] = p1;

        if (min.j != n - 1) {
          add_to_grid(config, grid, jets[min.j], points[min.j]);
        }

      } else {
        jet_i = jets[min.i];
        jet_j = jets[min.j];

        remove_from_grid(config, grid, jet_i, points[min.i]);
        remove_from_grid(config, grid, jet_j, points[min.j]);
        if (min.j != n - 1) {
          remove_from_grid(config, grid, jets[n - 1], points[n - 1]);
        }

        jet_i.px += jet_j.px;
        jet_i.py += jet_j.py;
        jet_i.pz += jet_j.pz;
        jet_i.E += jet_j.E;
        p2 = _set_jet(jet_i);

        p2.box_i = config.i(p2.eta);
        p2.box_j = config.j(p2.phi);

        jet_i.index = min.i;

        jets[min.i] = jet_i;
        points[min.i] = p2;

        jets[min.j] = jets[n - 1];
        points[min.j] = points[n - 1];
        jets[min.j].index = min.j;

        add_to_grid(config, grid, jet_i, p2);
        if (min.j != n - 1) {
          add_to_grid(config, grid, jets[min.j], points[min.j]);
        }
      }
    }
    n--;
    __syncthreads();
  }
}
#pragma endregion

void cluster(PseudoJet *particles, int size, Scheme scheme, double r) {
  // examples from FastJet span |eta| < 10
  // TODO: make the eta range dynamic, based on the data themselves
  // TODO: try to use __constant__ memory for config
  const Grid config(-10., +10., 0, 2 * M_PI, r, size);

#pragma region vectors
  EtaPhi *d_points_ptr;
  cudaCheck(cudaMalloc(&d_points_ptr, sizeof(EtaPhi) * size));

  // TODO: use `short` instead of `int` if there are less than 32k particles
  ParticleIndexType *d_grid_ptr;
  cudaCheck(cudaMalloc(&d_grid_ptr, sizeof(ParticleIndexType) * config.max_i * config.max_j * config.n));

  Dist *d_min_dists_ptr;
  cudaCheck(cudaMalloc(&d_min_dists_ptr, sizeof(Dist) * size));
#pragma endregion

#pragma region kernel_launches
  cudaCheck(cudaDeviceSynchronize());
  // TODO: move to helper function
  int blockSize;
  int minGridSize;
  cudaCheck(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, set_points, 0, 0));
  int gridSize = std::min((size + blockSize - 1) / blockSize, minGridSize);
  // set jets into points
  set_points<<<gridSize, blockSize>>>(config, particles, d_points_ptr, size);
  cudaCheck(cudaDeviceSynchronize());

  // create grid
  set_grid<<<config.max_i, config.max_j>>>(config, d_grid_ptr, d_points_ptr, particles, size);
  cudaCheck(cudaDeviceSynchronize());

  // compute dist_min
  // for (int i = n; i > 0; i--) {
  // compute_nn<<<1, n>>>(d_grid_ptr, d_points_ptr, particles,
  //                      d_min_dists_ptr, i, N);

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

    reduce_recombine<<<gridSize, blockSize, sharedMemory>>>(
        config, d_grid_ptr, d_points_ptr, particles, d_min_dists_ptr, size, scheme, r);
    cudaCheck(cudaGetLastError());
  }
#pragma endregion

  cudaCheck(cudaDeviceSynchronize());
  cudaCheck(cudaFree(d_points_ptr));
  cudaCheck(cudaFree(d_grid_ptr));
  cudaCheck(cudaFree(d_min_dists_ptr));
}
