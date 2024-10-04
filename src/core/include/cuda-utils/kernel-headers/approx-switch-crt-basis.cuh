#ifndef APPROX_SWITCH_CRT_BASIS_CUH
#define APPROX_SWITCH_CRT_BASIS_CUH

#include <cstdint> // for uint32_t type
#include <iostream> // for printf
#include <cuda_runtime.h>

/*
 * Data type for m_vectors object
 */
struct m_vectors_struct {
    unsigned long* data;
    unsigned long modulus;
};

using uint128_t = unsigned __int128;

void callMyKernel(uint32_t ringDim, uint32_t sizeQ, uint32_t sizeP);

void callApproxSwitchCRTBasisKernel(int ringDim, int sizeP, int sizeQ,
                                     ulong*             m_vectors_data,
                                     ulong*             m_vectors_modulus,
                                     ulong*             QHatInvModq,
                                     ulong*             QHatInvModqPrecon,
                                     uint128_t*         QHatModp,
                                     uint128_t*         sum,
                                     uint128_t*         modpBarrettMu,
                                     ulong*             ans_m_vectors_data,
                                     ulong*             ans_m_vectors_modulus);

void approxSwitchCRTBasisKernelWrapper(dim3 blocks, dim3 threads, void** args, cudaStream_t stream);

void printMemoryInfo();


#endif

