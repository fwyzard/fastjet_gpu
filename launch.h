#ifndef launch_h
#define launch_h

#include <algorithm>
#include <algorithm>

#include <cuda_runtime.h>

#include "cudaCheck.h"

struct LaunchParameters {
  int gridSize;
  int blockSize;

  LaunchParameters() = default;

  LaunchParameters(int gridSize, int blockSize) :
    gridSize(gridSize),
    blockSize(blockSize)
  {}
};

template <typename T>
LaunchParameters estimateSingleBlock(T* kernel) {
  cudaFuncAttributes attr;
  cudaCheck(cudaFuncGetAttributes(&attr, kernel));

  return {1, attr.maxThreadsPerBlock};
}

template <typename T>
LaunchParameters estimateSingleBlock(T* kernel, int size) {
  cudaFuncAttributes attr;
  cudaCheck(cudaFuncGetAttributes(&attr, kernel));

  return {1, std::min(attr.maxThreadsPerBlock, size)};
}

template <typename T>
LaunchParameters estimateMinimalGrid(T* kernel) {
  LaunchParameters params;
  cudaCheck(cudaOccupancyMaxPotentialBlockSize(&params.gridSize, &params.blockSize, kernel, 0, 0));
  return params;
}

template <typename T>
LaunchParameters estimateMinimalGrid(T* kernel, int size) {
  LaunchParameters params;
  cudaCheck(cudaOccupancyMaxPotentialBlockSize(&params.gridSize, &params.blockSize, kernel, 0, 0));
  params.gridSize = std::min((size + params.blockSize - 1) / params.blockSize, params.gridSize);
  return params;
}

template <typename T>
LaunchParameters estimateGrid(T* kernel, int size) {
  LaunchParameters params;
  cudaCheck(cudaOccupancyMaxPotentialBlockSize(&params.gridSize, &params.blockSize, kernel, 0, 0));
  params.gridSize = (size + params.blockSize - 1) / params.blockSize;
  return params;
}

#endif // launch_h
