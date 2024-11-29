#include "cuda-utils/switch-format/cudaPortalForSwitchFormatBatch.h"

namespace lbcrypto {

cudaPortalForSwitchFormatBatch::cudaPortalForSwitchFormatBatch(ulong* device_m_vectors, uint32_t m_vectors_sizeX, uint32_t m_vectors_sizeY, int isInverse, cudaStream_t mainStream)
    :sizeX(m_vectors_sizeX), sizeY(m_vectors_sizeY), isInverse(isInverse), mainStream(mainStream) {

    this->device_m_vectors = device_m_vectors;
    size_t buffer_size = sizeX * sizeY * sizeof(ulong);

    if (!isInverse) {
        //printf("(cudaPortalForSwitchFormatBatch) is Forward\n");
        cudaMallocHost((void**)&host_rootOfUnityReverseTable, buffer_size,cudaHostAllocDefault);
        CUDA_CHECK(cudaMallocAsync((void**)&device_rootOfUnityReverseTable, buffer_size, mainStream));
        cudaMallocHost((void**)&host_rootOfUnityPreconReverseTable, buffer_size, cudaHostAllocDefault);
        CUDA_CHECK(cudaMallocAsync((void**)&device_rootOfUnityPreconReverseTable, buffer_size, mainStream));
        device_rootOfUnityInverseReverseTable = nullptr;
        device_rootOfUnityInversePreconReverseTable = nullptr;
    } else {
        //printf("(cudaPortalForSwitchFormatBatch) is Inverse\n");
        cudaMallocHost((void**)&host_rootOfUnityInverseReverseTable, buffer_size, cudaHostAllocDefault);
        CUDA_CHECK(cudaMallocAsync((void**)&device_rootOfUnityInverseReverseTable, buffer_size, mainStream));
        //printf("Allocating host_rootOfUnityInverseReverseTable with size: %lu\n", buffer_size);
        //printf("Pointer address: %p\n", host_rootOfUnityInverseReverseTable);
        cudaMallocHost((void**)&host_rootOfUnityInversePreconReverseTable, buffer_size, cudaHostAllocDefault);
        CUDA_CHECK(cudaMallocAsync((void**)&device_rootOfUnityInversePreconReverseTable, buffer_size, mainStream));
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

void cudaPortalForSwitchFormatBatch::copyInTwiddleFactorsBatch(uint32_t ptrOffset, cudaStream_t stream) {
    size_t twiddleFactorsBatchSize = sizeY * sizeof(ulong);

    auto device_reverse_table_ptr   = device_rootOfUnityReverseTable        + ptrOffset;
    auto host_reverse_table_ptr     = host_rootOfUnityReverseTable          + ptrOffset;
    auto device_precon_table_ptr    = device_rootOfUnityPreconReverseTable  + ptrOffset;
    auto host_precon_table_ptr      = host_rootOfUnityPreconReverseTable    + ptrOffset;

    CUDA_CHECK(cudaMemcpyAsync(device_reverse_table_ptr, host_reverse_table_ptr, twiddleFactorsBatchSize, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(device_precon_table_ptr, host_precon_table_ptr, twiddleFactorsBatchSize, cudaMemcpyHostToDevice, stream));
}

void cudaPortalForSwitchFormatBatch::copyInInvTwiddleFactorsBatch(uint32_t ptrOffset, cudaStream_t stream) {
    size_t twiddleFactorsBatchSize = sizeY * sizeof(ulong);

    auto device_reverse_table_ptr   = device_rootOfUnityInverseReverseTable         + ptrOffset;
    auto host_reverse_table_ptr     = host_rootOfUnityInverseReverseTable           + ptrOffset;
    auto device_precon_table_ptr    = device_rootOfUnityInversePreconReverseTable   + ptrOffset;
    auto host_precon_table_ptr      = host_rootOfUnityInversePreconReverseTable     + ptrOffset;

    CUDA_CHECK(cudaMemcpyAsync(device_reverse_table_ptr, host_reverse_table_ptr, twiddleFactorsBatchSize, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(device_precon_table_ptr, host_precon_table_ptr, twiddleFactorsBatchSize, cudaMemcpyHostToDevice, stream));
}

// forward NTT
void cudaPortalForSwitchFormatBatch::switchFormatToEvaluationBatch(dim3 blocks, dim3 threadsPerBlock,
                                                                   uint32_t n, NativeInteger& modulus, uint32_t ptr_offset, cudaStream_t stream) {
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
void cudaPortalForSwitchFormatBatch::switchFormatToCoefficientBatch(
    dim3 blocks_Pt1, dim3 blocks_Pt2, dim3 threadsPerBlock,
    uint32_t n, NativeInteger& modulus, ulong cycloOrderInverse, ulong cycloOrderInversePrecon,
    uint32_t ptr_offset, cudaStream_t stream) {
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