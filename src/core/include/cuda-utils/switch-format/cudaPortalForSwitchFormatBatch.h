#ifndef CUDAPORTALFORSWITCHFORMATBATCH_H
#define CUDAPORTALFORSWITCHFORMATBATCH_H

#include <cstdint>

#include "lattice/poly.h"

#include <cuda_runtime.h>
#include <cuda-utils/cuda_util_macros.h>
#include <cuda-utils/kernel-headers/switch-format.cuh>
#include <lattice/hal/default/dcrtpoly.h>

namespace lbcrypto {

class cudaPortalForSwitchFormatBatch {

    ulong* device_m_vectors;
    uint32_t sizeX;
    uint32_t sizeY;
    int isInverse;

    cudaStream_t mainStream;

    // NTT twiddle factors
    ulong* host_rootOfUnityReverseTable;
    ulong* host_rootOfUnityPreconReverseTable;

    ulong* device_rootOfUnityReverseTable;
    ulong* device_rootOfUnityPreconReverseTable;

    // inverse NTT twiddle factors
    ulong* host_rootOfUnityInverseReverseTable;
    ulong* host_rootOfUnityInversePreconReverseTable;

    ulong* device_rootOfUnityInverseReverseTable;
    ulong* device_rootOfUnityInversePreconReverseTable;

public:
    // Constructor for functionality in batches
    cudaPortalForSwitchFormatBatch(ulong* device_m_vectors, uint32_t m_vectors_sizeX, uint32_t m_vectors_sizeY, int isInverse, cudaStream_t mainStream);

    // Destructor
    ~cudaPortalForSwitchFormatBatch();

    ulong* get_host_rootOfUnityReverseTable() { return host_rootOfUnityReverseTable; }
    ulong* get_host_rootOfUnityPreconReverseTable() { return host_rootOfUnityPreconReverseTable; }

    ulong* get_host_rootOfUnityInverseReverseTable() { return host_rootOfUnityInverseReverseTable; }
    ulong* get_host_rootOfUnityInversePreconReverseTable() { return host_rootOfUnityInversePreconReverseTable; }

    void copyInTwiddleFactorsBatch(uint32_t ptrOffset, cudaStream_t stream);
    void copyInInvTwiddleFactorsBatch(uint32_t ptrOffset, cudaStream_t stream);

    // forward NTT
    void switchFormatToEvaluationBatch(
        dim3 blocksDim, dim3 threadsPerBlockDim,
        uint32_t n, NativeInteger& modulus, uint32_t ptr_offset, cudaStream_t stream);

    // Inverse NTT
    void switchFormatToCoefficientBatch(
        dim3 blocksDim_Pt1, dim3 blocksDim_Pt2, dim3 threadsPerBlockDim,
        uint32_t n, NativeInteger& modulus, ulong cycloOrderInverse, ulong cycloOrderInversePrecon,
        uint32_t ptr_offset, cudaStream_t stream);
};
}

#endif //CUDAPORTALFORSWITCHFORMATBATCH_H
