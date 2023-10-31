#ifndef APPROX_SWITCH_CRT_BASIS_CUH
#define APPROX_SWITCH_CRT_BASIS_CUH

#include <cuda_runtime.h>
#include <cstdint> // for uint32_t type
#include <iostream> // for printf

void callMyKernel(uint32_t ringDim, uint32_t sizeQ, uint32_t sizeP);

#endif

