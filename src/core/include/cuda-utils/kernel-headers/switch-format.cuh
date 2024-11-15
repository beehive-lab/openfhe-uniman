#ifndef SWITCH_FORMAT_CUH
#define SWITCH_FORMAT_CUH

#include <cstdint> // for uint32_t type
#include <iostream> // for printf
#include <cuda_runtime.h>

void fNTTKernelWrapper(dim3 blocks, dim3 threads, void** args, cudaStream_t stream);

void iNTTPart1KernelWrapper(dim3 blocks, dim3 threads, void** args, cudaStream_t stream);
void iNTTPart2KernelWrapper(dim3 blocksPt2, dim3 threads, void** args, cudaStream_t stream);

#endif