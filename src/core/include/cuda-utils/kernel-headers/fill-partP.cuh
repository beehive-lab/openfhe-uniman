#ifndef FILL_PARTP_CUH
#define FILL_PARTP_CUH

#include <cstdint> // for uint32_t type
#include <iostream> // for printf
#include <cuda_runtime.h>

void fillPartPKernelWrapper(dim3 blocks, dim3 threads, void** args, cudaStream_t stream);

#endif