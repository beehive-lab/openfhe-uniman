#ifndef APPROX_SWITCH_CRT_BASIS_CUH
#define APPROX_SWITCH_CRT_BASIS_CUH

#include <cuda_runtime.h>
#include <cstdint> // for uint32_t type
#include <iostream> // for printf

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
                                    m_vectors_struct*   host_m_vectors,
                                    ulong*              host_qhatinvmodq,
                                    ulong*              host_QHatInvModqPrecon,
                                    uint128_t*          host_qhatmodp,
                                    uint128_t*          host_modpBarrettMu,
                                    m_vectors_struct*   host_ans_m_vectors);

#endif

