#include <iostream>
#include <thrust/device_vector.h>

using namespace std;

#pragma region consts
const double pi = 3.141592653589793238462643383279502884197;
const double twopi = 6.283185307179586476925286766559005768394;
const double MaxRap = 1e5;
const double R = 0.6;
const double R2 = R * R;
const double invR2 = 1.0 / R2;
// const double MAX_DOUBLE = 1.79769e+308;
const double ptmin = 5.0;
const double dcut = ptmin * ptmin;
const int grid_max_x = 50;
const int grid_max_y = twopi / R + 1;
const int eta_offsit = 25;
#pragma endregion

#pragma region struct
struct PseudoJet {
  int index;
  double px;
  double py;
  double pz;
  double E;
  bool isJet;
};

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
#pragma endregion

#pragma region util_h
// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char *msg);
__host__ __device__ void print_distance(Dist &d);
__host__ __device__ void print_point(EtaPhi &p);
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

__device__ double plain_distance(EtaPhi &p1, EtaPhi &p2) {
  double dphi = abs(p1.phi - p2.phi);
  if (dphi > pi) {
    dphi = twopi - dphi;
  }
  double drap = p1.eta - p2.eta;
  return (dphi * dphi + drap * drap);
}

__device__ Dist yij_distance(EtaPhi *points, int i, int j) {
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
    d.distance = min(points[i].diB, points[j].diB) *
                 plain_distance(points[i], points[j]) * invR2;

  return d;
}

__device__ Dist minimum_in_cell(int *grid, EtaPhi *points, PseudoJet *jets,
                                Dist min, int tid, int i, int j, int n) {
  int k = 0;
  int offset = (j * n) + (i * grid_max_y * n);
  int num = grid[offset + k];

  // PseudoJet jet1 = jets[tid];
  // PseudoJet jet2;
  Dist temp;
  while (num > 0) {
    if (tid != num) {
      temp = yij_distance(points, tid, num);

      if (temp.distance < min.distance)
        min = temp;
    }

    k++;
    num = grid[offset + k];
  }

  return min;
}

__device__ void remove_from_grid(int *grid, PseudoJet &jet, EtaPhi &p, int n) {
  // Remove from grid
  int k = 0;
  int offset = (p.box_j * n) + (p.box_i * grid_max_y * n);
  int num = grid[offset + k];
  bool shift = false;

  // if (jet.index == 212)
  //   print_point(p);
  while (num != -1) {
    if (jet.index == num)
      shift = true;
    // if (jet.index == 212)
    //   printf("%d\n", num);
    if (shift) {
      grid[offset + k] = grid[offset + k + 1];
      // k--;
    }
    k++;

    num = grid[offset + k];
  }
}

__device__ void add_to_grid(int *grid, PseudoJet &jet, EtaPhi &p, int n) {
  // Remove from grid
  int k = 0;
  int offset = (p.box_j * n) + (p.box_i * grid_max_y * n);
  int num = grid[offset + k];

  // if (jet.index == 212)
  //   print_point(p);
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
__global__ void set_points(PseudoJet *jets, EtaPhi *points, int n, float r) {
  int tid = threadIdx.x;

  if (tid >= n)
    return;

  EtaPhi p = _set_jet(jets[tid]);
  p.box_i = p.eta / r + eta_offsit;
  p.box_j = p.phi / r;

  // if (p.box_j == 0)
  //   print_point(p);
  points[tid] = p;
  // printf("%4d", tid);
  // print_point(p);
  // printf("%4d%4d%4d%20.8e%20.8e%20.8e\n", tid, p.box_i, p.box_j, p.eta,
  // p.phi,
  //  p.diB);
}

__global__ void set_grid(int *grid, EtaPhi *points, PseudoJet *jets, int n) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  // printf("%4d%4d: ", bid, tid);

  int k = 0;
  EtaPhi p;

  int offset = (tid * n) + (bid * grid_max_y * n);

  // if (bid == 0)
  //   printf("%4d%4d%10d\n", bid, tid, offset);
  for (int i = 0; i < n; i++) {
    p = points[i];

    if (p.box_i == bid && p.box_j == tid) {
      grid[offset + k] = jets[i].index;
      // printf("%4d%4d%4d\n", bid, tid, grid[bid * 64 + tid * 64 +
      // k]);
      k++;
    }
  }

  // if (bid)

  grid[offset + k] = -1;
  // printf("-1\n");
}

// __global__ void compute_nn(int *grid, EtaPhi *points, PseudoJet *jets,
//                            Dist *min_dists, int n, int N) {
//   int tid = threadIdx.x;

//   if (tid >= n)
//     return;

//

//   // if (tid == 186 && n == 212)
//   //   print_distance(min);

//   // print_distance(min);
// }

__global__ void reduce_recombine(int *grid, EtaPhi *points, PseudoJet *jets,
                                 Dist *min_dists, int n, float r, int N) {
  extern __shared__ Dist sdata[];

  int tid = threadIdx.x;

  if (tid >= n)
    return;

  while (n > 0) {

    EtaPhi p = points[tid];
    Dist min;
    min = yij_distance(points, tid, tid);

    min = minimum_in_cell(grid, points, jets, min, tid, p.box_i, p.box_j, N);
    // if (tid == 186 && n == 212)
    //   print_distance(min);
    if (p.box_i + 1 < grid_max_x + 1) {
      min = minimum_in_cell(grid, points, jets, min, tid, p.box_i + 1, p.box_j,
                            N);
      // if (tid == 186 && n == 212)
      //   print_distance(min);
    }
    if (p.box_i - 1 >= 0) {
      min = minimum_in_cell(grid, points, jets, min, tid, p.box_i - 1, p.box_j,
                            N);
      // if (tid == 186 && n == 212)
      //   print_distance(min);
    }
    // check if above grid_max_y
    int j = p.box_j + 1 <= grid_max_y ? p.box_j + 1 : 0;

    min = minimum_in_cell(grid, points, jets, min, tid, p.box_i, j, N);
    // if (tid == 186 && n == 212)
    //   print_distance(min);
    if (p.box_i + 1 < grid_max_x + 1) {
      min = minimum_in_cell(grid, points, jets, min, tid, p.box_i + 1, j, N);
      // if (tid == 186 && n == 212)
      //   print_distance(min);
    }
    if (p.box_i - 1 >= 0) {
      min = minimum_in_cell(grid, points, jets, min, tid, p.box_i - 1, j, N);
      // if (tid == 186 && n == 212)
      //   print_distance(min);
    }

    // check if bellow 0
    j = p.box_j - 1 >= 0 ? p.box_j - 1 : grid_max_y - 1;
    min = minimum_in_cell(grid, points, jets, min, tid, p.box_i, j, N);
    // if (tid == 186 && n == 212)
    //   print_distance(min);
    if (p.box_i + 1 < grid_max_x + 1) {
      min = minimum_in_cell(grid, points, jets, min, tid, p.box_i + 1, j, N);
      // if (tid == 186 && n == 212)
      //   print_distance(min);
    }
    if (p.box_i - 1 >= 0) {
      min = minimum_in_cell(grid, points, jets, min, tid, p.box_i - 1, j, N);
      // if (tid == 186 && n == 212)
      //   print_distance(min);
    }

    if (p.box_j - 1 < 0) {
      min = minimum_in_cell(grid, points, jets, min, tid, p.box_i, j - 1, N);
      // if (tid == 186 && n == 212)
      //   print_distance(min);
      if (p.box_i + 1 < grid_max_x + 1) {
        min = minimum_in_cell(grid, points, jets, min, tid, p.box_i + 1, j - 1,
                              N);
        // if (tid == 186 && n == 212)
        //   print_distance(min);
      }
      if (p.box_i - 1 >= 0) {
        min = minimum_in_cell(grid, points, jets, min, tid, p.box_i - 1, j - 1,
                              N);
        // if (tid == 186 && n == 212)
        //   print_distance(min);
      }
    }

    int t;
    if (min.i > min.j) {
      t = min.i;
      min.i = min.j;
      min.j = t;
    }
    min_dists[tid] = min;

    sdata[tid] = min;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
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
      // Dist d = yij_distance(points, 57, 61);
      // print_distance(d);
      // print_distance(min);
      PseudoJet jet_i, jet_j;

      EtaPhi p1, p2;
      if (min.i == min.j) {
        jet_j = jets[min.j];
        p1 = points[min.j];
        remove_from_grid(grid, jet_j, p1, N);
        if (min.j != n - 1)
          remove_from_grid(grid, jets[n - 1], points[n - 1], N);

        jets[min.j] = jets[n - 1];
        points[min.j] = points[n - 1];

        jets[min.j].index = min.j;

        jet_j.isJet = true;
        jet_j.index = n - 1;
        jets[n - 1] = jet_j;
        points[n - 1] = p1;

        if (min.j != n - 1)
          add_to_grid(grid, jets[min.j], points[min.j], N);

      } else {
        jet_i = jets[min.i];
        // p1 = points[min.i];

        jet_j = jets[min.j];
        // p2 = points[min.j];

        // print_point(points[287]);
        remove_from_grid(grid, jet_i, points[min.i], N);
        remove_from_grid(grid, jet_j, points[min.j], N);
        if (min.j != n - 1) {
          // printf("removing: %4d\n", n - 1);
          remove_from_grid(grid, jets[n - 1], points[n - 1], N);
        }

        jet_i.px += jet_j.px;
        jet_i.py += jet_j.py;
        jet_i.pz += jet_j.pz;
        jet_i.E += jet_j.E;
        p2 = _set_jet(jet_i);

        p2.box_i = p2.eta / r + eta_offsit;
        p2.box_j = p2.phi / r;

        jet_i.index = min.i;

        jets[min.i] = jet_i;
        points[min.i] = p2;

        jets[min.j] = jets[n - 1];
        points[min.j] = points[n - 1];
        jets[min.j].index = min.j;

        add_to_grid(grid, jet_i, p2, N);
        if (min.j != n - 1)
          add_to_grid(grid, jets[min.j], points[min.j], N);
      }
    }
    n--;
    __syncthreads();
  }
}
#pragma endregion

int main() {

#pragma region device_prop
  cudaSetDevice(0);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("Device Name: %s\n", prop.name);
#pragma endregion

  int num_events = 1;

  for (int event = 0; event < num_events; event++) {

#pragma region read_jets
    thrust::host_vector<PseudoJet> h_jets;
    PseudoJet temp;

    int i = 0;
    while (true) {
      cin >> temp.px >> temp.py >> temp.pz >> temp.E;
      temp.index = i;
      if (cin.fail())
        break;

      i++;
      h_jets.push_back(temp);
    }

    cin.clear();
    cin.ignore(numeric_limits<streamsize>::max(), '\n');

    int n = h_jets.size();
    int N = n;
#pragma endregion

#pragma region vectors
    thrust::device_vector<PseudoJet> d_jets(h_jets);
    PseudoJet *d_jets_ptr = thrust::raw_pointer_cast(d_jets.data());

    thrust::device_vector<EtaPhi> d_points(n);
    EtaPhi *d_points_ptr = thrust::raw_pointer_cast(d_points.data());

    thrust::device_vector<int> d_grid(n * grid_max_x * grid_max_y);
    int *d_grid_ptr = thrust::raw_pointer_cast(d_grid.data());

    thrust::device_vector<Dist> d_min_dists(n);
    Dist *d_min_dists_ptr = thrust::raw_pointer_cast(d_min_dists.data());
#pragma endregion

#pragma region timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
#pragma endregion;

#pragma region kernel_launches
    // set jets into points
    set_points<<<1, 512>>>(d_jets_ptr, d_points_ptr, n, R);

    // create grid
    set_grid<<<grid_max_x + 1, grid_max_y>>>(d_grid_ptr, d_points_ptr,
                                             d_jets_ptr, n);

    // compute dist_min
    // for (int i = n; i > 0; i--) {
    // compute_nn<<<1, n>>>(d_grid_ptr, d_points_ptr, d_jets_ptr,
    //                      d_min_dists_ptr, i, N);

    reduce_recombine<<<1, 512, sizeof(Dist) * n>>>(
        d_grid_ptr, d_points_ptr, d_jets_ptr, d_min_dists_ptr, n, R, N);
// }
#pragma endregion

#pragma region timing

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Check for any CUDA errors
    checkCUDAError("kernal launch");
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("%d\t%.3fms\n", h_jets.size(), milliseconds);
#pragma endregion

    thrust::host_vector<int> h_grid(d_grid);
    // // h_grid = d_grid;
    // thrust::host_vector<Dist> h_min_dists(d_min_dists);
    thrust::host_vector<EtaPhi> h_points(d_points);

#pragma region testing
    h_jets = d_jets;
    // thrust::host_vector<EtaPhi> h_points(d_points);
    // thrust::host_vector<Dist> h_min_dists(d_min_dists);

    for (int i = 0; i < n; i++) {
      // printf("%4d%4d%20.8e\n", h_min_dists[i].i, h_min_dists[i].j,
      //        h_min_dists[i].distance);
      if (h_points[i].diB >= dcut && h_jets[i].isJet)
        printf("%15.8f %15.8f %15.8f\n", h_points[i].eta, h_points[i].phi,
               sqrt(h_points[i].diB));
    }

// for (int i = 0; i < n; i++)
// print_distance(h_min_dists[i]);
// for (int i = 0; i < n; i++)
//   print_point(h_points[i]);

// for (int i = 0; i < grid_max_x; i++)
//   for (int j = 0; j < grid_max_y; j++) {
//     cout << i << " " << j << ": ";
//     int offset = (j * n) + (i * grid_max_y * n);
//     for (int k = 0; k < n; k++) {
//       int num = h_grid[offset + k];
//       if (num == -1)
//         break;
//       cout << h_grid[offset + k] << " ";
//     }
//     cout << -1 << endl;
//   }
#pragma endregion
  }

  return 0;
}

#pragma region util_c
void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(-1);
  }
}

__host__ __device__ void print_distance(Dist &d) {
  if (d.i == d.j)
    printf("%4d%4d%20.8e\n", d.i, -2, d.distance);
  else
    printf("%4d%4d%20.8e\n", d.i, d.j, d.distance);
}

__host__ __device__ void print_point(EtaPhi &p) {
  printf("%4d%4d%20.8e%20.8e%20.8e\n", p.box_i, p.box_j, p.eta, p.phi, p.diB);
}
#pragma endregion