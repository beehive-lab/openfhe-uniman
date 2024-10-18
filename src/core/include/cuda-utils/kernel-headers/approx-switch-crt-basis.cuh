#ifndef APPROX_SWITCH_CRT_BASIS_CUH
#define APPROX_SWITCH_CRT_BASIS_CUH

#include <cstdint> // for uint32_t type
#include <iostream> // for printf
#include <cuda_runtime.h>
//#include "cuda-utils/kernel-headers/shared_device_functions.cuh"

using uint128_t = unsigned __int128;

void callMyKernel(uint32_t ringDim, uint32_t sizeQ, uint32_t sizeP);

void approxSwitchCRTBasisKernelWrapper(dim3 blocks, dim3 threads, void** args, cudaStream_t stream);

void printMemoryInfo();


#endif

