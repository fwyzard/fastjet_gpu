#include <cmath>

#include <cuda_runtime.h>
#include <thrust/device_free.h>
#include <thrust/device_vector.h>
#include <thrust/memory.h>

#include "PseudoJet.h"
#include "cluster.h"
#include "cudaCheck.h"

using namespace std;

const double pi = 3.141592653589793238462643383279502884197;
const double twopi = 6.283185307179586476925286766559005768394;
const double MaxRap = 1e5;

struct EtaPhi {
  double eta;
  double phi;
  double diB;
};

struct Dist {
  double distance;
  int i;
  int j;
};

__host__ __device__ void print_distance(Dist d) {
  printf("%4d%4d%20.8e\n", d.i, d.j, d.distance);
}

__device__ double plain_distance(EtaPhi &p1, EtaPhi &p2) {
  double dphi = abs(p1.phi - p2.phi);
  if (dphi > pi) {
    dphi = twopi - dphi;
  }
  double drap = p1.eta - p2.eta;
  return (dphi * dphi + drap * drap);
}

__device__ Dist yij_distance(EtaPhi *points, int i, int j, double one_over_r2) {
  if (i > j) {
    int t = i;
    i = j;
    j = t;
  }

  Dist d;
  d.i = i;
  d.j = j;
  if (i == j)
    d.distance = points[i].diB;
  else
    d.distance = min(points[i].diB, points[j].diB) *
                 plain_distance(points[i], points[j]) * one_over_r2;

  return d;
}

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
  } // can happen if phi=-|eps<1e-15|?
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
    double effective_m2 = max(0.0, (jet.E + jet.pz) * (jet.E - jet.pz) -
                                       point.diB); // force non tachyonic mass
    double E_plus_pz = jet.E + abs(jet.pz);        // the safer of p+, p-
    // p+/p- = (p+ p-) / (p-)^2 = (kt^2+m^2)/(p-)^2
    point.eta = 0.5 * log((point.diB + effective_m2) / (E_plus_pz * E_plus_pz));
    if (jet.pz > 0) {
      point.eta = -point.eta;
    }
  }

  return point;
}

__global__ void set_points(PseudoJet *jets, EtaPhi *points) {
  int tid = threadIdx.x;
  points[tid] = _set_jet(jets[tid]);
}

__global__ void set_distances(EtaPhi *points, Dist *min_dists, double one_over_r2) {
  extern __shared__ Dist sdata[];

  int tid = threadIdx.x;
  int k = blockIdx.x;

  if (tid > k)
    return;

  Dist d1, d2;

  sdata[tid] = yij_distance(points, tid, k, one_over_r2);
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s && (tid + s) <= k) {
      if (sdata[tid + s].distance < sdata[tid].distance) {
        sdata[tid] = sdata[tid + s];
      }
    }
    __syncthreads();
  }

  // Minimum of the row
  if (tid == 0) {
    min_dists[k] = sdata[0];

    if (k == 0) {
      // Compute min for jet 0
      min_dists[0] = yij_distance(points, 0, 0, one_over_r2);

      // Compute min for jet 1
      d1 = yij_distance(points, 1, 0, one_over_r2);
      d2 = yij_distance(points, 1, 1, one_over_r2);
      if (d1.distance < d2.distance)
        min_dists[1] = d1;
      else
        min_dists[1] = d2;
    }
  }
}

__global__ void fastjet(PseudoJet *jets, EtaPhi *points, Dist *dists,
                        Dist *min_dists, int n, double one_over_r2) {
  extern __shared__ Dist sdata[];

  int tid = threadIdx.x;

  if (tid >= n)
    return;

  int k;

  Dist d1;
  Dist d2;

  // loop
  bool first_run = true;
  while (n > 0) {
    Dist min;
    Dist temp;
    sdata[tid] = min_dists[tid];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (tid < s && (tid + s) < n) {
        if (!first_run) {
          // recalculate

          // get d tid min.
          if (min.i < n && tid > min.i) {
            temp = yij_distance(points, tid, min.i, one_over_r2);
            // check if dj < sdata[tid].distance
            if (temp.distance < sdata[tid].distance) {
              sdata[tid] = temp;
              min_dists[tid] = temp;
              // print_distance(temp);
            }
          }

          if (min.i < n && tid + s > min.i) {
            temp = yij_distance(points, tid + s, min.i, one_over_r2);
            // check if di < sdata[tid+s].distance
            if (temp.distance < sdata[tid + s].distance) {
              sdata[tid + s] = temp;
              min_dists[tid + s] = temp;
              // print_distance(temp);
            }
          }

          // get d tid min.j
          if (min.j < n && tid > min.j) {
            temp = yij_distance(points, tid, min.j, one_over_r2);
            // check if dj < sdata[tid].distance
            if (temp.distance < sdata[tid].distance) {
              sdata[tid] = temp;
              min_dists[tid] = temp;
              // print_distance(temp);
            }
          }

          if (min.j < n && tid + s > min.j) {
            temp = yij_distance(points, tid + s, min.j, one_over_r2);
            // check if dj < sdata[tid+s].distance
            if (temp.distance < sdata[tid + s].distance && min.j) {
              sdata[tid + s] = temp;
              min_dists[tid + s] = temp;
              // print_distance(temp);
            }
          }
        }
        if (sdata[tid + s].distance < sdata[tid].distance) {
          sdata[tid] = sdata[tid + s];
        }
      }
      __syncthreads();
    }

    first_run = false;
    // __syncthreads();
    // // recombine
    min = sdata[0];
    if (tid == 0) {

      PseudoJet jet_i, jet_j;

      EtaPhi p1, p2;
      if (min.i == min.j) {
        jet_j = jets[min.j];
        p1 = points[min.j];

        jets[min.j] = jets[n - 1];
        points[min.j] = points[n - 1];

        jets[min.j].index = min.j;

        jet_j.isJet = true;
        jet_j.index = n - 1;
        jets[n - 1] = jet_j;
        points[n - 1] = p1;
      } else {
        jet_i = jets[min.i];
        // p1 = points[min.i];

        jet_j = jets[min.j];
        // p2 = points[min.j];

        jet_i.px += jet_j.px;
        jet_i.py += jet_j.py;
        jet_i.pz += jet_j.pz;
        jet_i.E += jet_j.E;
        p2 = _set_jet(jet_i);
        jet_i.index = min.i;

        jets[min.i] = jet_i;
        points[min.i] = p2;

        jets[min.j] = jets[n - 1];
        points[min.j] = points[n - 1];
        jets[min.j].index = min.j;
      }
    }
    n--;
    if (tid >= n)
      return;

    __syncthreads();
    Dist d = min_dists[tid];
    if (d.i == min.i || d.j == min.i) {
      min_dists[tid].distance = -1;
    } else if (d.i == min.j || d.j == min.j) {
      min_dists[tid].distance = -1;
    }

    __syncthreads();

    for (k = n - 1; k >= 0; k--) {
      if (min_dists[k].distance < 0) {
        sdata[tid] = yij_distance(points, k, tid, one_over_r2);
        // dists[(k * (k - 1) / 2) + tid];
        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
          if (tid < s && (tid + s) <= k) {
            if (sdata[tid + s].distance < sdata[tid].distance) {
              sdata[tid] = sdata[tid + s];
            }
          }
          __syncthreads();
        }

        // __syncthreads();
        // Minimum of the row
        if (tid == 0) {
          min_dists[k] = sdata[0];
        }
      }
    }

// min.i
#pragma region
    sdata[tid] = yij_distance(points, min.i, tid, one_over_r2);
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (tid < s && (tid + s) <= min.i) {
        if (sdata[tid + s].distance < sdata[tid].distance) {
          sdata[tid] = sdata[tid + s];
        }
      }
      __syncthreads();
    }

    // __syncthreads();
    // Minimum of the row
    if (tid == 0) {
      min_dists[min.i] = sdata[0];

      // Compute min for jet 0
      min_dists[0] = yij_distance(points, 0, 0, one_over_r2);

      // Compute min for jet 1
      d1 = yij_distance(points, 1, 0, one_over_r2);
      d2 = yij_distance(points, 1, 1, one_over_r2);
      if (d1.distance < d2.distance)
        min_dists[1] = d1;
      else
        min_dists[1] = d2;
    }
#pragma endregion

    __syncthreads();

    // min.j
    if (min.i != min.j) {
#pragma region
      sdata[tid] = yij_distance(points, min.j, tid, one_over_r2);
      __syncthreads();

      for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && (tid + s) <= min.j) {
          if (sdata[tid + s].distance < sdata[tid].distance) {
            sdata[tid] = sdata[tid + s];
          }
        }
        __syncthreads();
      }

      // __syncthreads();
      // Minimum of the row
      if (tid == 0) {
        min_dists[min.j] = sdata[0];
      }
#pragma endregion
      __syncthreads();
    }
    // __syncthreads();
    // }
    // __syncthreads();
    // }

    // __syncthreads();
    // if (tid == 0) {
    //   // printf("n = %d\n", n);
    //   for (int i = 0; i < n; i++) {
    //     printf("%4d%4d%20.8e\n", min_dists[i].i, min_dists[i].j,
    //            min_dists[i].distance);
    //   }
    // }

    // // recalculate (update rows minimum)
  }
}

void cluster(PseudoJet *particles, int size, double r) {
    EtaPhi *d_points_ptr;
    cudaCheck(cudaMalloc(&d_points_ptr, sizeof(EtaPhi) * size));

    Dist *d_dists_ptr;
    cudaCheck(cudaMalloc(&d_dists_ptr, sizeof(Dist) * size * (size + 1) / 2));

    Dist *d_min_dists_ptr;
    cudaCheck(cudaMalloc(&d_min_dists_ptr, sizeof(Dist) * size));

    set_points<<<1, size>>>(particles, d_points_ptr);

    const double one_over_r2 = 1. / (r * r);
    set_distances<<<size, 512, sizeof(Dist) * size>>>(d_points_ptr, d_min_dists_ptr, one_over_r2);

    fastjet<<<1, 1024, (sizeof(Dist) * size)>>>(particles, d_points_ptr, d_dists_ptr, d_min_dists_ptr, size, one_over_r2);

    cudaCheck(cudaFree(d_points_ptr));
    cudaCheck(cudaFree(d_dists_ptr));
    cudaCheck(cudaFree(d_min_dists_ptr));
}
