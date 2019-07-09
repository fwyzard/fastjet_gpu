#include <iostream>
#include <limits>
#include <cmath>
#include <assert.h>

// Here you can set the device ID that was assigned to you
#define MYDEVICE 0

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char *msg);

using namespace std;

struct PseudoJet
{
    double px;
    double py;
    double pz;
    double E;
    double diB;
    double phi;
    double rap;
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

__device__ void _set_jet(PseudoJet &jet)
{
    jet.diB = jet.px * jet.px + jet.py * jet.py;

    if (jet.diB == 0.0)
    {
        jet.phi = 0.0;
    }
    else
    {
        jet.phi = atan2(jet.py, jet.px);
    }
    if (jet.phi < 0.0)
    {
        jet.phi += twopi;
    }
    if (jet.phi >= twopi)
    {
        jet.phi -= twopi;
    } // can happen if phi=-|eps<1e-15|?
    if (jet.E == abs(jet.pz) && jet.diB == 0)
    {
        // Point has infinite rapidity -- convert that into a very large
        // number, but in such a way that different 0-pt momenta will have
        // different rapidities (so as to lift the degeneracy between
        // them) [this can be relevant at parton-level]
        double MaxRapHere = MaxRap + abs(jet.pz);
        if (jet.pz >= 0.0)
        {
            jet.rap = MaxRapHere;
        }
        else
        {
            jet.rap = -MaxRapHere;
        }
    }
    else
    {
        // get the rapidity in a way that's modestly insensitive to roundoff
        // error when things pz,E are large (actually the best we can do without
        // explicit knowledge of mass)
        double effective_m2 = max(0.0, (jet.E + jet.pz) * (jet.E - jet.pz) - jet.diB); // force non tachyonic mass
        double E_plus_pz = jet.E + abs(jet.pz);                                        // the safer of p+, p-
        // p+/p- = (p+ p-) / (p-)^2 = (kt^2+m^2)/(p-)^2
        jet.rap = 0.5 * log((jet.diB + effective_m2) / (E_plus_pz * E_plus_pz));
        if (jet.pz > 0)
        {
            jet.rap = -jet.rap;
        }
    }
}

__device__ double plain_distance(PseudoJet &jet1, PseudoJet &jet2)
{
    double dphi = abs(jet1.phi - jet2.phi);
    if (dphi > pi)
    {
        dphi = twopi - dphi;
    }
    double drap = jet1.rap - jet2.rap;
    return (dphi * dphi + drap * drap);
}

__global__ void dumb_n3(PseudoJet *jets, int num_particles, double *mini)
{
    __shared__ PseudoJet s_jets[NUM_PARTICLES];
    __shared__ double s_distances[NUM_PARTICLES];
    __shared__ int ii_indices[NUM_PARTICLES];
    __shared__ int jj_indices[NUM_PARTICLES];

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < num_particles)
    {
        s_jets[tid] = jets[tid];
        _set_jet(s_jets[tid]);
        __syncthreads();
    }

    // Loop on all particles
    while (num_particles > 0 && tid < num_particles)
    {
        s_distances[tid] = s_jets[tid].diB;
        ii_indices[tid] = tid;
        jj_indices[tid] = -1;
        __syncthreads();

        int ii = -1;
        int jj = -1;

        // Find minimum yiB
        for (int s = 1; s < blockDim.x; s *= 2)
        {
            if (threadIdx.x % (2 * s) == 0)
                if (threadIdx.x + s < num_particles)
                    if (s_distances[threadIdx.x] >
                        s_distances[threadIdx.x + s])
                    {
                        s_distances[threadIdx.x] =
                            s_distances[threadIdx.x + s];

                        ii_indices[threadIdx.x] =
                            ii_indices[threadIdx.x + s];
                    }
            __syncthreads();
        }

        // Thread 0 gets the results form different blocks
        double ymin;
        if (tid == 0)
        {
            ymin = s_distances[0];
            ii = ii_indices[0];
            for (int i = 1; i < gridDim.x; i++)
                if (s_distances[i * blockDim.x + 1] < ymin)
                {
                    ymin = s_distances[i * blockDim.x + 1];
                    ii = ii_indices[i * blockDim.x + 1];
                }

            *mini = ymin;
            s_distances[0] = ymin;
            ii_indices[0] = ii;
            // printf("mini = %.17e\n", ymin);
            // printf("ii = %d\n", ii);
            // printf("jj = %d\n", jj);
        }
        __syncthreads();

        // Get the miniumum yiB and its index to all threads
        ymin = s_distances[0];
        ii = ii_indices[0];
        __syncthreads();

        // Reset the indices
        ii_indices[tid] = tid;
        __syncthreads();

        // Find the minimun yij between the current jet that is processed 
        // by the current thread (one jet with all jets that follows)
        if (tid < num_particles)
        {
            double distance = 0;
            for (int j = tid + 1; j < num_particles; j++)
            {
                distance = min(s_jets[tid].diB, s_jets[j].diB)
                * plain_distance(s_jets[tid], s_jets[j])
                * invR2;
                // if(tid == 0)
                //     printf("%.17e\n", distance);
                if (distance < ymin)
                {
                    ymin = distance;
                    ii = tid;
                    jj = j;
                }
            }
        }

        // Communicate the minimum distance between the jet and other jets
        // if the distance is less than the yiB, and communicate the indices
        // ii jj
        s_distances[tid] = ymin;
        ii_indices[tid] = ii;
        jj_indices[tid] = jj;

        __syncthreads();

        // if(jj > -1)
        // Find the smallest distance between all the threads
        for (int s = 1; s < blockDim.x; s *= 2)
        {
            if (threadIdx.x % (2 * s) == 0)
                if (threadIdx.x + s < num_particles)
                    if (s_distances[threadIdx.x] >
                        s_distances[threadIdx.x + s])
                    {
                        s_distances[threadIdx.x] =
                            s_distances[threadIdx.x + s];

                        ii_indices[threadIdx.x] = ii_indices[threadIdx.x + s];
                        jj_indices[threadIdx.x] = jj_indices[threadIdx.x + s];
                    }
            __syncthreads();
        }
        __syncthreads();

        // Get the minimum from all blocks
        if (tid == 0)
        {
            ymin = s_distances[0];
            ii = ii_indices[0];
            jj = jj_indices[0];
            for (int i = 1; i < gridDim.x; i++)
                if (s_distances[i * blockDim.x + 1] < ymin)
                {
                    ymin = s_distances[i * blockDim.x + 1];
                    ii = ii_indices[i * blockDim.x + 1];
                    jj = jj_indices[i * blockDim.x + 1];
                }

            *mini = ymin;
            s_distances[0] = ymin;
            ii_indices[0] = ii;
            jj_indices[0] = jj;
            // printf("mini = %.17e\n", ymin);
            // printf("ii = %d\n", ii);
            // printf("jj = %d\n", jj);
        // }
        // __syncthreads();

        // if (tid == 0)
        // {
            // Perform recombination using E_Scheme (Simple Sum)
            if (jj > 0)
            {
                // Do yij recombination
                s_jets[ii].px += s_jets[jj].px;
                s_jets[ii].py += s_jets[jj].py;
                s_jets[ii].pz += s_jets[jj].pz;
                s_jets[ii].E += s_jets[jj].E;
                _set_jet(s_jets[ii]);

                s_jets[jj] = s_jets[num_particles - 1];
            }
            else
            {
                // Do yiB recombination
                if (s_jets[ii].diB >= dcut)
                    printf("%15.8f %15.8f %15.8f\n",
                           s_jets[ii].rap, s_jets[ii].phi, sqrt(s_jets[ii].diB));

                s_jets[ii] = s_jets[num_particles - 1];
            }
        }

        num_particles--;
        __syncthreads();
    }
}

int main()
{
    cudaSetDevice(MYDEVICE);

    PseudoJet *h_jets = 0;
    h_jets = (PseudoJet *)malloc(NUM_PARTICLES * sizeof(PseudoJet));

    double *h_mini = 0;
    h_mini = (double *)malloc(sizeof(double));
    *h_mini = numeric_limits<double>::max();

    int i;
    for (i = 0; i < NUM_PARTICLES; i++)
    {
        cin >> h_jets[i].px >> h_jets[i].py >> h_jets[i].pz >> h_jets[i].E;
    }

    PseudoJet *d_jets = 0;
    cudaMalloc((void **)&d_jets, NUM_PARTICLES * sizeof(PseudoJet));
    cudaMemcpy(d_jets, h_jets, NUM_PARTICLES * sizeof(PseudoJet), cudaMemcpyHostToDevice);

    double *d_mini = 0;
    cudaMalloc((void **)&d_mini, sizeof(double));
    cudaMemcpy(d_mini, h_mini, sizeof(double), cudaMemcpyHostToDevice);

    // Check for any CUDA errors
    checkCUDAError("cudaMemcpy calls1");

    int num_threads = 1024;//354;
    int num_blocks = (NUM_PARTICLES - 1) / num_threads + 1;
    //std::cout << "blocks = " << num_blocks;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    dumb_n3<<<num_blocks,
              num_threads //,
              //NUM_PARTICLES * sizeof(PseudoJet) // Jets
              //    + NUM_PARTICLES * sizeof(double) // Distances
              //    + NUM_PARTICLES * 2 * sizeof(int)
              >>>(d_jets, NUM_PARTICLES, d_mini);
    cudaEventRecord(stop);

    // Check for any CUDA errors
    checkCUDAError("kernal launch");

    cudaMemcpy(h_jets, d_jets,
               NUM_PARTICLES * sizeof(PseudoJet),
               cudaMemcpyDeviceToHost);
    // Check for any CUDA errors
    checkCUDAError("cudaMemcpy2 calls");

    cudaMemcpy(h_mini, d_mini, sizeof(double), cudaMemcpyDeviceToHost);

    // Check for any CUDA errors
    checkCUDAError("cudaMemcpy3 calls");
    cudaEventSynchronize(stop);

    //    cout << "d_mini = " << *h_mini << std::endl;
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("time = %10.8f\n", milliseconds);

    // free device memory
    cudaFree(d_jets);
    cudaFree(d_mini);

    // free host memory
    free(h_jets);
    free(h_mini);

    return 0;
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(-1);
    }
}
