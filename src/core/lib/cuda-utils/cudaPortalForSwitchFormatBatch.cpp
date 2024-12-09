#include "cuda-utils/switch-format/cudaPortalForSwitchFormatBatch.h"

namespace lbcrypto {

cudaPortalForSwitchFormatBatch::cudaPortalForSwitchFormatBatch(ulong* device_m_vectors, const uint32_t m_vectors_sizeX,
                                                               const uint32_t m_vectors_sizeY, const int isInverse,
                                                               cudaStream_t mainStream)
    :sizeX(m_vectors_sizeX), sizeY(m_vectors_sizeY), isInverse(isInverse), mainStream(mainStream) {

    this->device_m_vectors   = device_m_vectors;
    const size_t buffer_size = sizeX * sizeY * sizeof(ulong);

    if (!isInverse) {
        //printf("(cudaPortalForSwitchFormatBatch) is Forward\n");
        cudaMallocHost(reinterpret_cast<void**>(&host_rootOfUnityReverseTable), buffer_size,cudaHostAllocDefault);
        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&device_rootOfUnityReverseTable), buffer_size, mainStream));
        cudaMallocHost(reinterpret_cast<void**>(&host_rootOfUnityPreconReverseTable), buffer_size, cudaHostAllocDefault);
        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&device_rootOfUnityPreconReverseTable), buffer_size, mainStream));
        device_rootOfUnityInverseReverseTable = nullptr;
        device_rootOfUnityInversePreconReverseTable = nullptr;
    } else {
        //printf("(cudaPortalForSwitchFormatBatch) is Inverse\n");
        cudaMallocHost(reinterpret_cast<void**>(&host_rootOfUnityInverseReverseTable), buffer_size, cudaHostAllocDefault);
        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&device_rootOfUnityInverseReverseTable), buffer_size, mainStream));
        //printf("Allocating host_rootOfUnityInverseReverseTable with size: %lu\n", buffer_size);
        //printf("Pointer address: %p\n", host_rootOfUnityInverseReverseTable);
        cudaMallocHost(reinterpret_cast<void**>(&host_rootOfUnityInversePreconReverseTable), buffer_size, cudaHostAllocDefault);
        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&device_rootOfUnityInversePreconReverseTable), buffer_size, mainStream));
        //printf("Allocating host_rootOfUnityInversePreconReverseTable with size: %lu\n", buffer_size);
        //printf("Pointer address: %p\n", host_rootOfUnityInversePreconReverseTable);
        device_rootOfUnityReverseTable = nullptr;
        device_rootOfUnityPreconReverseTable = nullptr;
    }
}

cudaPortalForSwitchFormatBatch::~cudaPortalForSwitchFormatBatch() {
    if (!isInverse) {
        // forward NTT
        SAFE_CUDA_FREE_HOST(host_rootOfUnityReverseTable);
        SAFE_CUDA_FREE_HOST(host_rootOfUnityPreconReverseTable);

        CUDA_SAFE_FREE(device_rootOfUnityReverseTable, mainStream);
        CUDA_SAFE_FREE(device_rootOfUnityPreconReverseTable, mainStream);
    } else {
        // inverse NTT
        SAFE_CUDA_FREE_HOST(host_rootOfUnityInverseReverseTable);
        SAFE_CUDA_FREE_HOST(host_rootOfUnityInversePreconReverseTable);

        CUDA_SAFE_FREE(device_rootOfUnityInverseReverseTable, mainStream);
        CUDA_SAFE_FREE(device_rootOfUnityInversePreconReverseTable, mainStream);
    }
}

void cudaPortalForSwitchFormatBatch::copyInTwiddleFactorsBatch(const uint32_t ptrOffset, cudaStream_t stream) const {
    const size_t twiddleFactorsBatchSize = sizeY * sizeof(ulong);

    const auto device_reverse_table_ptr = device_rootOfUnityReverseTable        + ptrOffset;
    const auto host_reverse_table_ptr   = host_rootOfUnityReverseTable          + ptrOffset;
    const auto device_precon_table_ptr  = device_rootOfUnityPreconReverseTable  + ptrOffset;
    const auto host_precon_table_ptr    = host_rootOfUnityPreconReverseTable    + ptrOffset;

    CUDA_CHECK(cudaMemcpyAsync(device_reverse_table_ptr, host_reverse_table_ptr, twiddleFactorsBatchSize, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(device_precon_table_ptr, host_precon_table_ptr, twiddleFactorsBatchSize, cudaMemcpyHostToDevice, stream));
}

void cudaPortalForSwitchFormatBatch::copyInInvTwiddleFactorsBatch(const uint32_t ptrOffset, cudaStream_t stream) const {
    const size_t twiddleFactorsBatchSize = sizeY * sizeof(ulong);

    const auto device_reverse_table_ptr = device_rootOfUnityInverseReverseTable         + ptrOffset;
    const auto host_reverse_table_ptr   = host_rootOfUnityInverseReverseTable           + ptrOffset;
    const auto device_precon_table_ptr  = device_rootOfUnityInversePreconReverseTable   + ptrOffset;
    const auto host_precon_table_ptr    = host_rootOfUnityInversePreconReverseTable     + ptrOffset;

    CUDA_CHECK(cudaMemcpyAsync(device_reverse_table_ptr, host_reverse_table_ptr, twiddleFactorsBatchSize, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(device_precon_table_ptr, host_precon_table_ptr, twiddleFactorsBatchSize, cudaMemcpyHostToDevice, stream));
}

// forward NTT
void cudaPortalForSwitchFormatBatch::switchFormatToEvaluationBatch(const dim3 blocks, const dim3 threadsPerBlock,
                                                                   uint32_t n, const NativeInteger& modulus,
                                                                   const uint32_t ptr_offset,
                                                                   cudaStream_t stream) const {
    auto convertedModulus = modulus.ConvertToInt<>();

    // Calculate the pointers with the applied offset
    auto device_reverse_table_ptr   = device_rootOfUnityReverseTable + ptr_offset;
    auto device_precon_table_ptr    = device_rootOfUnityPreconReverseTable + ptr_offset;
    auto device_m_vectors_ptr       = device_m_vectors + ptr_offset;

    for (uint32_t m = 1; m < n; m <<= 1) {
        uint32_t step = (n/m) >> 1;

        void *args[] = {
            &m, &n, &step,
            &device_m_vectors_ptr, &convertedModulus,
            &device_reverse_table_ptr,
            &device_precon_table_ptr
        };
        fNTTBatchKernelWrapper(blocks, threadsPerBlock, args, stream);
    }

}


// inverse NTT
void cudaPortalForSwitchFormatBatch::switchFormatToCoefficientBatch(const dim3 blocks_Pt1, const dim3 blocks_Pt2,
                                                                    const dim3 threadsPerBlock, uint32_t n,
                                                                    NativeInteger& modulus, ulong cycloOrderInverse,
                                                                    ulong cycloOrderInversePrecon, const uint32_t ptr_offset,
                                                                    cudaStream_t stream) const {
    auto convertedModulus = modulus.ConvertToInt<>();

    // Calculate the pointers with the applied offset
    auto device_inv_reverse_table_ptr   = device_rootOfUnityInverseReverseTable + ptr_offset;
    auto device_inv_precon_table_ptr    = device_rootOfUnityInversePreconReverseTable + ptr_offset;
    auto device_m_vectors_ptr           = device_m_vectors + ptr_offset;

    for (uint32_t m = (n >> 1); m >= 1; m >>= 1) {
        uint32_t step = (n/m) >> 1;
        void *argsPt1[] = {
            &m, &n, &step,
            &device_m_vectors_ptr, &convertedModulus,
            &device_inv_reverse_table_ptr,
            &device_inv_precon_table_ptr
        };
        iNTTBatchPart1KernelWrapper(blocks_Pt1, threadsPerBlock, argsPt1, stream);
    }
    void *argsPt2[] = {
        &device_m_vectors_ptr, &convertedModulus,
        &cycloOrderInverse,
        &cycloOrderInversePrecon
    };
    iNTTBatchPart2KernelWrapper(blocks_Pt2, threadsPerBlock, argsPt2, stream);
}

}