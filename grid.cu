#include <cmath>

#include <cuda_runtime.h>

#include "PseudoJet.h"
#include "cluster.h"
#include "cudaCheck.h"

using namespace std;

#pragma region consts
const double pi = 3.141592653589793238462643383279502884197;
const double twopi = 6.283185307179586476925286766559005768394;
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

  __host__ __device__ Grid(double min_eta, double max_eta, double min_phi, double max_phi, double r)
      : min_eta(min_eta),
        max_eta(max_eta),
        min_phi(min_phi),
        max_phi(min_phi),
        r(r),
        max_i(int((max_eta - min_eta) / r) + 1),
        max_j(int((max_phi - min_phi) / r) + 1) {}

  __host__ __device__ constexpr inline int i(double eta) const { return static_cast<int>((max_eta - min_eta) / r); }

  __host__ __device__ constexpr inline int j(double phi) const { return static_cast<int>((max_phi - min_phi) / r); }

  __host__ __device__ constexpr inline double eta_min(int i) const { return min_eta + r * i; }

  __host__ __device__ constexpr inline double eta_max(int i) const { return min_eta + r * (i + 1); }

  __host__ __device__ constexpr inline double phi_min(int j) const { return min_phi + r * j; }

  __host__ __device__ constexpr inline double phi_max(int j) const { return min_phi + r * (j + 1); }

  __host__ __device__ constexpr inline int index(int i, int j) const { return max_j * i + j; }
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
    point.phi = atan2(jet.py, jet.px);
  }
  if (point.phi < 0.0) {
    point.phi += twopi;
  }
  if (point.phi >= twopi) {
    point.phi -= twopi;
  }  // can happen if phi=-|eps<1e-15|?
  if (jet.E == abs(jet.pz) && point.diB == 0) {
    // Point has infinite rapidity -- convert that into a very large
    // number, but in such a way that different 0-pt momenta will have
    // different rapidities (so as to lift the degeneracy between
    // them) [this can be relevant at parton-level]
    double MaxRapHere = MaxRap + abs(jet.pz);
    if (jet.pz >= 0.0) {
      point.eta = MaxRapHere;
    } else {
      point.eta = -MaxRapHere;
    }
  } else {
    // get the rapidity in a way that's modestly insensitive to roundoff
    // error when things pz,E are large (actually the best we can do without
    // explicit knowledge of mass)
    double effective_m2 = max(0.0, (jet.E + jet.pz) * (jet.E - jet.pz) - point.diB);  // force non tachyonic mass
    double E_plus_pz = jet.E + abs(jet.pz);                                           // the safer of p+, p-
    // p+/p- = (p+ p-) / (p-)^2 = (kt^2+m^2)/(p-)^2
    point.eta = 0.5 * log((point.diB + effective_m2) / (E_plus_pz * E_plus_pz));
    if (jet.pz > 0) {
      point.eta = -point.eta;
    }
  }

  return point;
}

__device__ double plain_distance(const EtaPhi &p1, const EtaPhi &p2) {
  double dphi = abs(p1.phi - p2.phi);
  if (dphi > pi) {
    dphi = twopi - dphi;
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
    d.distance = min(points[i].diB, points[j].diB) * plain_distance(points[i], points[j]) * one_over_r2;

  return d;
}

__device__ Dist minimum_in_cell(Grid const &grid_config,
                                const int *grid,
                                const EtaPhi *points,
                                const PseudoJet *jets,
                                Dist min,
                                const int tid,
                                const int i,
                                const int j,
                                const int n,
                                double one_over_r2) {
  int k = 0;
  int offset = grid_config.index(i, j) * n;
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

__device__ void remove_from_grid(Grid const &grid_config, int *grid, PseudoJet &jet, const EtaPhi &p, const int n) {
  // Remove from grid
  int k = 0;
  int offset = grid_config.index(p.box_i, p.box_j) * n;
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

__device__ void add_to_grid(Grid const &grid_config, int *grid, const PseudoJet &jet, const EtaPhi &p, const int n) {
  // Remove from grid
  int k = 0;
  int offset = grid_config.index(p.box_i, p.box_j) * n;
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
__global__ void set_points(Grid const &grid_config, PseudoJet *jets, EtaPhi *points, const int n) {
  int tid = threadIdx.x;

  if (tid >= n)
    return;

  EtaPhi p = _set_jet(jets[tid]);
  p.box_i = grid_config.i(p.eta);
  p.box_j = grid_config.j(p.phi);

  points[tid] = p;
}

__global__ void set_grid(Grid const &grid_config, int *grid, const EtaPhi *points, const PseudoJet *jets, const int n) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  int k = 0;
  EtaPhi p;

  int offset = grid_config.index(bid, tid) * n;

  for (int i = 0; i < n; i++) {
    p = points[i];

    if (p.box_i == bid && p.box_j == tid) {
      grid[offset + k] = jets[i].index;
      k++;
    }
  }

  grid[offset + k] = -1;
}

__global__ void reduce_recombine(Grid const &grid_config,
                                 int *grid,
                                 EtaPhi *points,
                                 PseudoJet *jets,
                                 Dist *min_dists,
                                 int n,
                                 const float r,
                                 const int N) {
  extern __shared__ Dist sdata[];

  const double one_over_r2 = 1. / (r * r);
  int tid = threadIdx.x;

  if (tid >= n)
    return;

  min_dists[tid].i = -3;
  min_dists[tid].j = -1;
  Dist min;
  min.i = -4;
  min.j = -4;
  while (n > 0) {
    if (tid >= n)
      return;

    EtaPhi p = points[tid];
    Dist local_min = min_dists[tid];
    if (local_min.i == -3 || local_min.j == min.i || local_min.j == min.j || local_min.i == min.i ||
        local_min.i == min.j || local_min.i >= n || local_min.j >= n) {
      EtaPhi bp;

      min = yij_distance(points, tid, tid, one_over_r2);
#define NODY false
#if defined NODY
      // printf("not dynamic!\n");

      min = minimum_in_cell(grid_config, grid, points, jets, min, tid, p.box_i, p.box_j, N, one_over_r2);

      bool right = true;
      bool left = true;
      bool up = true;
      bool down = true;

      bp.eta = grid_config.eta_max(p.box_i);
      bp.phi = p.phi;
      if (min.distance < plain_distance(p, bp)) {
        // printf("saved right!\n");
        right = false;
      }

      bp.eta = grid_config.eta_min(p.box_i);
      bp.phi = p.phi;
      if (min.distance < plain_distance(p, bp)) {
        // printf("saved left!\n");
        // printf("%20.8e\n", bp.eta);
        // printf("%20.8e\n", points[min.j].eta);
        left = false;
      }

      bp.eta = p.eta;
      bp.phi = p.box_j + 1 <= grid_config.max_j ? (p.box_j + 1) * r : 0;
      if (min.distance < plain_distance(p, bp)) {
        // printf("saved up!\n");
        up = false;
      }

      bp.eta = p.eta;
      bp.phi = p.box_j - 1 >= 0 ? p.box_j * r : (grid_config.max_j - 1) * r;
      if (min.distance < plain_distance(p, bp) && p.box_j - 1 >= 0) {
        // printf("saved down!\n");
        down = false;
      }

      // Right
      if (p.box_i + 1 < grid_config.max_i + 1 && right) {
        min = minimum_in_cell(grid_config, grid, points, jets, min, tid, p.box_i + 1, p.box_j, N, one_over_r2);
      }

      // Left
      if (p.box_i - 1 >= 0 && left) {
        min = minimum_in_cell(grid_config, grid, points, jets, min, tid, p.box_i - 1, p.box_j, N, one_over_r2);
      }

      // check if above grid_config.max_j
      int j = p.box_j + 1 <= grid_config.max_j ? p.box_j + 1 : 0;

      // Up
      if (up) {
        min = minimum_in_cell(grid_config, grid, points, jets, min, tid, p.box_i, j, N, one_over_r2);

        // Up Right
        if (p.box_i + 1 < grid_config.max_i + 1 && right) {
          min = minimum_in_cell(grid_config, grid, points, jets, min, tid, p.box_i + 1, j, N, one_over_r2);
        }

        // Up Left
        if (p.box_i - 1 >= 0 && left) {
          min = minimum_in_cell(grid_config, grid, points, jets, min, tid, p.box_i - 1, j, N, one_over_r2);
        }
      }

      // check if bellow 0
      j = p.box_j - 1 >= 0 ? p.box_j - 1 : grid_config.max_j - 1;

      if (down) {
        // Down
        min = minimum_in_cell(grid_config, grid, points, jets, min, tid, p.box_i, j, N, one_over_r2);

        // Down Right
        if (p.box_i + 1 < grid_config.max_i + 1 && right) {
          min = minimum_in_cell(grid_config, grid, points, jets, min, tid, p.box_i + 1, j, N, one_over_r2);
        }

        // Down Left
        if (p.box_i - 1 >= 0 && left) {
          min = minimum_in_cell(grid_config, grid, points, jets, min, tid, p.box_i - 1, j, N, one_over_r2);
        }

        if (p.box_j - 1 < 0) {
          // Down Down
          min = minimum_in_cell(grid_config, grid, points, jets, min, tid, p.box_i, j - 1, N, one_over_r2);

          // Down Down Right
          if (p.box_i + 1 < grid_config.max_i + 1 && right) {
            min = minimum_in_cell(grid_config, grid, points, jets, min, tid, p.box_i + 1, j - 1, N, one_over_r2);
          }

          // Down Down Left
          if (p.box_i - 1 >= 0 && left) {
            min = minimum_in_cell(grid_config, grid, points, jets, min, tid, p.box_i - 1, j - 1, N, one_over_r2);
          }
        }
      }
#endif

#if !defined NODY
      minimum_in_cell_kernel<<<1, 12, 12 * sizeof(Dist)>>>(grid, points, jets, min_dists, min, tid, p, N);
      cudaDeviceSynchronize();
#endif

      int t;
      if (min.i > min.j) {
        t = min.i;
        min.i = min.j;
        min.j = t;
      }

#if defined NODY
      min_dists[tid] = min;
#endif
    }

    sdata[tid] = min_dists[tid];
    __syncthreads();

    for (unsigned int s = 256; s > 0; s >>= 1) {
      if (tid < s && (tid + s) < n) {
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
    if (tid == 0) {
      PseudoJet jet_i, jet_j;

      EtaPhi p1, p2;
      if (min.i == min.j) {
        jet_j = jets[min.j];
        p1 = points[min.j];
        remove_from_grid(grid_config, grid, jet_j, p1, N);
        if (min.j != n - 1)
          remove_from_grid(grid_config, grid, jets[n - 1], points[n - 1], N);

        jets[min.j] = jets[n - 1];
        points[min.j] = points[n - 1];

        jets[min.j].index = min.j;

        jet_j.isJet = true;
        jet_j.index = n - 1;
        jets[n - 1] = jet_j;
        points[n - 1] = p1;

        if (min.j != n - 1)
          add_to_grid(grid_config, grid, jets[min.j], points[min.j], N);

      } else {
        jet_i = jets[min.i];
        jet_j = jets[min.j];

        remove_from_grid(grid_config, grid, jet_i, points[min.i], N);
        remove_from_grid(grid_config, grid, jet_j, points[min.j], N);
        if (min.j != n - 1) {
          // printf("removing: %4d\n", n - 1);
          remove_from_grid(grid_config, grid, jets[n - 1], points[n - 1], N);
        }

        jet_i.px += jet_j.px;
        jet_i.py += jet_j.py;
        jet_i.pz += jet_j.pz;
        jet_i.E += jet_j.E;
        p2 = _set_jet(jet_i);

        p2.box_i = grid_config.i(p2.eta);
        p2.box_j = grid_config.j(p2.phi);

        jet_i.index = min.i;

        jets[min.i] = jet_i;
        points[min.i] = p2;

        jets[min.j] = jets[n - 1];
        points[min.j] = points[n - 1];
        jets[min.j].index = min.j;

        add_to_grid(grid_config, grid, jet_i, p2, N);
        if (min.j != n - 1)
          add_to_grid(grid_config, grid, jets[min.j], points[min.j], N);
      }
    }
    n--;
    __syncthreads();
  }
}
#pragma endregion

void cluster(PseudoJet *particles, int size, double r) {
  const Grid grid_config(-5., +5., 0, 2 * M_PI, r);

#pragma region vectors
  EtaPhi *d_points_ptr;
  cudaCheck(cudaMalloc(&d_points_ptr, sizeof(EtaPhi) * size));

  int *d_grid_ptr;
  cudaCheck(cudaMalloc(&d_grid_ptr, sizeof(int) * size * grid_config.max_i * grid_config.max_j));

  Dist *d_min_dists_ptr;
  cudaCheck(cudaMalloc(&d_min_dists_ptr, sizeof(Dist) * size));
#pragma endregion

#pragma region kernel_launches
  // set jets into points
  set_points<<<1, 512>>>(grid_config, particles, d_points_ptr, size);

  // create grid
  set_grid<<<grid_config.max_i + 1, grid_config.max_j>>>(grid_config, d_grid_ptr, d_points_ptr, particles, size);

  // compute dist_min
  // for (int i = n; i > 0; i--) {
  // compute_nn<<<1, n>>>(d_grid_ptr, d_points_ptr, particles,
  //                      d_min_dists_ptr, i, N);

  std::cerr << "shared memory: " << sizeof(Dist) * size << std::endl;
  reduce_recombine<<<1, size, sizeof(Dist) * size>>>(
      grid_config, d_grid_ptr, d_points_ptr, particles, d_min_dists_ptr, size, r, size);
  cudaCheck(cudaGetLastError());
#pragma endregion

  cudaCheck(cudaFree(d_points_ptr));
  cudaCheck(cudaFree(d_grid_ptr));
  cudaCheck(cudaFree(d_min_dists_ptr));
}
