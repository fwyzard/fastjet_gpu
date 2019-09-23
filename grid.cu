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
struct EtaPhi {
  double eta;
  double phi;
  double diB;
  int box_i;
  int box_j;
};

struct Dist {
  double distance;
  int i;
  int j;
};

struct Grid {
  double min_eta;
  double max_eta;
  double min_phi;
  double max_phi;
  double r;
  int max_i;
  int max_j;
  int n;

  // TODO use a smaller grid size (esimate from distributions in data/mc)
  // TODO usa a SoA
  __host__ __device__ Grid(double min_eta, double max_eta, double min_phi, double max_phi, double r, int n)
      : min_eta(min_eta),
        max_eta(max_eta),
        min_phi(min_phi),
        max_phi(min_phi),
        r(r),
        max_i(int((max_eta - min_eta) / r) + 1),
        max_j(int((max_phi - min_phi) / r) + 1),
        n(n) {}

  __host__ __device__ constexpr inline int i(double eta) const { return static_cast<int>((max_eta - min_eta) / r); }

  __host__ __device__ constexpr inline int j(double phi) const { return static_cast<int>((max_phi - min_phi) / r); }

  __host__ __device__ constexpr inline double eta_min(int i) const { return min_eta + r * i; }

  __host__ __device__ constexpr inline double eta_max(int i) const { return min_eta + r * (i + 1); }

  __host__ __device__ constexpr inline double phi_min(int j) const { return min_phi + r * j; }

  __host__ __device__ constexpr inline double phi_max(int j) const { return min_phi + r * (j + 1); }

  __host__ __device__ constexpr inline int index(int i, int j) const { return max_j * i + j; }

  __host__ __device__ constexpr inline int offset(int i, int j) const { return index(i, j) * n; }
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

__device__ Dist yij_distance(const EtaPhi *points, int i, int j, double one_over_r2) {
  // k is the one in qusetion
  // d k tid
  if (i > j) {
    int t = i;
    i = j;
    j = t;
  }

  Dist d;
  d.i = i;
  d.j = j;
  // if k == tid return diB
  if (i == j)
    d.distance = points[i].diB;
  else
    d.distance = ::min(points[i].diB, points[j].diB) * plain_distance(points[i], points[j]) * one_over_r2;

  return d;
}

__device__ Dist minimum_in_cell(Grid const &config,
                                const int *grid,
                                const EtaPhi *points,
                                const PseudoJet *jets,
                                Dist min,
                                const int tid,          // jet index
                                const int i,            // cell coordinates
                                const int j,
                                double one_over_r2) {
  int k = 0;
  int offset = config.offset(i, j);
  int num = grid[offset + k];

  // PseudoJet jet1 = jets[tid];
  // PseudoJet jet2;
  Dist temp;
  while (num > 0) {
    if (tid != num) {
      temp = yij_distance(points, tid, num, one_over_r2);

      if (temp.distance < min.distance)
        min = temp;
    }

    k++;
    num = grid[offset + k];
  }

  return min;
}

__device__ void remove_from_grid(Grid const &config, int *grid, PseudoJet &jet, const EtaPhi &p) {
  // Remove from grid
  int k = 0;
  int offset = config.offset(p.box_i, p.box_j);
  int num = grid[offset + k];
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

__device__ void add_to_grid(Grid const &config, int *grid, const PseudoJet &jet, const EtaPhi &p) {
  // Remove from grid
  int k = 0;
  int offset = config.offset(p.box_i, p.box_j);
  int num = grid[offset + k];

  while (true) {
    num = grid[offset + k];
    if (num == -1) {
      grid[offset + k] = jet.index;
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
  }
}

__global__ void set_grid(Grid config, int *grid, const EtaPhi *points, const PseudoJet *jets, const int n) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  int k = 0;
  EtaPhi p;

  int offset = config.offset(bid, tid);

  for (int i = 0; i < n; i++) {
    p = points[i];

    if (p.box_i == bid and p.box_j == tid) {
      grid[offset + k] = jets[i].index;
      k++;
    }
  }

  grid[offset + k] = -1;
}

__global__ void reduce_recombine(
    Grid config, int *grid, EtaPhi *points, PseudoJet *jets, Dist *min_dists, int n, const float r) {
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
        EtaPhi bp;

        min = yij_distance(points, tid, tid, one_over_r2);

        min = minimum_in_cell(config, grid, points, jets, min, tid, p.box_i, p.box_j, one_over_r2);

        bool right = true;
        bool left = true;
        bool up = true;
        bool down = true;

        bp.eta = config.eta_max(p.box_i);
        bp.phi = p.phi;
        if (min.distance < plain_distance(p, bp)) {
          // printf("saved right!\n");
          right = false;
        }

        bp.eta = config.eta_min(p.box_i);
        bp.phi = p.phi;
        if (min.distance < plain_distance(p, bp)) {
          // printf("saved left!\n");
          // printf("%20.8e\n", bp.eta);
          // printf("%20.8e\n", points[min.j].eta);
          left = false;
        }

        bp.eta = p.eta;
        bp.phi = p.box_j + 1 <= config.max_j ? (p.box_j + 1) * r : 0;
        if (min.distance < plain_distance(p, bp)) {
          // printf("saved up!\n");
          up = false;
        }

        bp.eta = p.eta;
        bp.phi = p.box_j - 1 >= 0 ? p.box_j * r : (config.max_j - 1) * r;
        if (min.distance < plain_distance(p, bp) and p.box_j - 1 >= 0) {
          // printf("saved down!\n");
          down = false;
        }

        // Right
        if (p.box_i + 1 < config.max_i and right) {
          min = minimum_in_cell(config, grid, points, jets, min, tid, p.box_i + 1, p.box_j, one_over_r2);
        }

        // Left
        if (p.box_i - 1 >= 0 and left) {
          min = minimum_in_cell(config, grid, points, jets, min, tid, p.box_i - 1, p.box_j, one_over_r2);
        }

        // check if above config.max_j
        int j = p.box_j + 1 <= config.max_j ? p.box_j + 1 : 0;

        // Up
        if (up) {
          min = minimum_in_cell(config, grid, points, jets, min, tid, p.box_i, j, one_over_r2);

          // Up Right
          if (p.box_i + 1 < config.max_i and right) {
            min = minimum_in_cell(config, grid, points, jets, min, tid, p.box_i + 1, j, one_over_r2);
          }

          // Up Left
          if (p.box_i - 1 >= 0 and left) {
            min = minimum_in_cell(config, grid, points, jets, min, tid, p.box_i - 1, j, one_over_r2);
          }
        }

        // check if bellow 0
        j = p.box_j - 1 >= 0 ? p.box_j - 1 : config.max_j - 1;

        // Down
        if (down) {
          min = minimum_in_cell(config, grid, points, jets, min, tid, p.box_i, j, one_over_r2);

          // Down Right
          if (p.box_i + 1 < config.max_i and right) {
            min = minimum_in_cell(config, grid, points, jets, min, tid, p.box_i + 1, j, one_over_r2);
          }

          // Down Left
          if (p.box_i - 1 >= 0 and left) {
            min = minimum_in_cell(config, grid, points, jets, min, tid, p.box_i - 1, j, one_over_r2);
          }

          if (p.box_j - 1 < 0) {
            // Down Down
            min = minimum_in_cell(config, grid, points, jets, min, tid, p.box_i, j - 1, one_over_r2);

            // Down Down Right
            if (p.box_i + 1 < config.max_i and right) {
              min = minimum_in_cell(config, grid, points, jets, min, tid, p.box_i + 1, j - 1, one_over_r2);
            }

            // Down Down Left
            if (p.box_i - 1 >= 0 and left) {
              min = minimum_in_cell(config, grid, points, jets, min, tid, p.box_i - 1, j - 1, one_over_r2);
            }
          }
        }

        int t;
        if (min.i > min.j) {
          t = min.i;
          min.i = min.j;
          min.j = t;
        }

        min_dists[tid] = min;
      }

      sdata[tid] = min_dists[tid];
    }
    __syncthreads();

    // FIXME: why 256 ?
    //for (unsigned int s = 256; s > 0; s >>= 1)
    unsigned int width = 1;
    while (width * 2 < n) {
      width *= 2;
    }
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

void cluster(PseudoJet *particles, int size, double r) {
  const Grid config(-5., +5., 0, 2 * M_PI, r, size);
  // TODO: try to use __constant__ memory for config

#pragma region vectors
  EtaPhi *d_points_ptr;
  cudaCheck(cudaMalloc(&d_points_ptr, sizeof(EtaPhi) * size));

  int *d_grid_ptr;
  cudaCheck(cudaMalloc(&d_grid_ptr, sizeof(int) * size * config.max_i * config.max_j));

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
        config, d_grid_ptr, d_points_ptr, particles, d_min_dists_ptr, size, r);
    cudaCheck(cudaGetLastError());
  }
#pragma endregion

  cudaCheck(cudaDeviceSynchronize());
  cudaCheck(cudaFree(d_points_ptr));
  cudaCheck(cudaFree(d_grid_ptr));
  cudaCheck(cudaFree(d_min_dists_ptr));
}
