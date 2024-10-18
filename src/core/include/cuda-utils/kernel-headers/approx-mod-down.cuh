#ifndef APPROX_MOD_DOWN_CUH
#define APPROX_MOD_DOWN_CUH

#include <cstdint> // for uint32_t type
#include <iostream> // for printf
#include <cuda_runtime.h>
//#include "cuda-utils/kernel-headers/shared_device_functions.cuh"

using uint128_t = unsigned __int128;

void approxModDownKernelWrapper(dim3 blocks, dim3 threads, void** args, cudaStream_t stream);

void printMemoryInfo();

#endif