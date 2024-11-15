#include "cuda-utils/switch-format/cudaPortalForSwitchFormat.h"

namespace lbcrypto {

cudaPortalForSwitchFormat::cudaPortalForSwitchFormat(ulong* device_m_vectors, uint32_t m_vectors_sizeX, uint32_t m_vectors_sizeY, cudaStream_t stream)
    :stream(stream), sizeX(m_vectors_sizeX), sizeY(m_vectors_sizeY) {

    this->device_m_vectors = device_m_vectors;
    this->cyclotomicOrder = m_vectors_sizeY << 1;

    host_rootOfUnityReverseTable = nullptr;
    host_rootOfUnityPreconReverseTable = nullptr;
    device_rootOfUnityReverseTable = nullptr;
    device_rootOfUnityPreconReverseTable = nullptr;

    host_rootOfUnityInverseReverseTable = nullptr;
    host_rootOfUnityInversePreconReverseTable = nullptr;
    host_cycloOrderInverseTable = nullptr;
    host_cycloOrderInversePreconTable = nullptr;
    device_rootOfUnityInverseReverseTable = nullptr;
    device_rootOfUnityInversePreconReverseTable = nullptr;
    device_cycloOrderInverseTable = nullptr;
    device_cycloOrderInversePreconTable = nullptr;
}

cudaPortalForSwitchFormat::~cudaPortalForSwitchFormat() {
    if (host_rootOfUnityReverseTable) {
        // forward NTT
        SAFE_FREE(host_rootOfUnityReverseTable);
        SAFE_FREE(host_rootOfUnityPreconReverseTable);

        CUDA_SAFE_FREE(device_rootOfUnityReverseTable, stream);
        CUDA_SAFE_FREE(device_rootOfUnityPreconReverseTable, stream);
    } else {
        // inverse NTT
        SAFE_FREE(host_rootOfUnityInverseReverseTable);
        SAFE_FREE(host_rootOfUnityInversePreconReverseTable);
        SAFE_FREE(host_cycloOrderInverseTable);
        SAFE_FREE(host_cycloOrderInversePreconTable);

        CUDA_SAFE_FREE(device_rootOfUnityInverseReverseTable, stream);
        CUDA_SAFE_FREE(device_rootOfUnityInversePreconReverseTable, stream);
        CUDA_SAFE_FREE(device_cycloOrderInverseTable, stream);
        CUDA_SAFE_FREE(device_cycloOrderInversePreconTable, stream);
    }
}

// Getters
uint32_t cudaPortalForSwitchFormat::get_cyclotomicOrder() { return cyclotomicOrder; }

ulong* cudaPortalForSwitchFormat::get_device_rootOfUnityInverseReverseTable()       { return device_rootOfUnityInverseReverseTable; }
ulong* cudaPortalForSwitchFormat::get_device_rootOfUnityInversePreconReverseTable() { return device_rootOfUnityInversePreconReverseTable; }
ulong* cudaPortalForSwitchFormat::get_device_cycloOrderInverseTable()               { return device_cycloOrderInverseTable; }
ulong* cudaPortalForSwitchFormat::get_device_cycloOrderInversePreconMap()           { return device_cycloOrderInversePreconTable; }

// Flatten Map Functions
void cudaPortalForSwitchFormat::flattenRootOfUnityReverseTableByModulus(const std::vector<PolyImpl<NativeVector>>& m_vectors,
                                                                        const std::map<ulong, std::vector<ulong>>& inputMap_rootOfUnityReverse) {
    host_rootOfUnityReverseTable = (ulong*)malloc(sizeX * sizeY * sizeof(ulong));
    for (uint32_t i = 0; i < sizeX; i++) {
        long key = m_vectors[i].GetModulus().ConvertToInt<>();
        const auto& currentValues = inputMap_rootOfUnityReverse.at(key);
        memcpy(&host_rootOfUnityReverseTable[i * sizeY], currentValues.data(), sizeY * sizeof(ulong));
    }
}

void cudaPortalForSwitchFormat::flattenRootOfUnityPreconReverseTableByModulus(const std::vector<PolyImpl<NativeVector>>& m_vectors,
                                                                              const std::map<ulong, std::vector<ulong>>& inputMap_rootOfUnityPrecon) {
    host_rootOfUnityPreconReverseTable = (ulong*)malloc(sizeX * sizeY * sizeof(ulong));
    for (uint32_t i = 0; i < sizeX; i++) {
        long key = m_vectors[i].GetModulus().ConvertToInt<>();
        const auto& currentValues = inputMap_rootOfUnityPrecon.at(key);
        memcpy(&host_rootOfUnityPreconReverseTable[i * sizeY], currentValues.data(), sizeY * sizeof(ulong));
    }
}

void cudaPortalForSwitchFormat::flattenRootOfUnityInverseReverseTableByModulus(const std::vector<PolyImpl<NativeVector>>& m_vectors,
                                                                               const std::map<ulong, std::vector<ulong>>& inputMap_rootOfUnityInverseReverse) {
    host_rootOfUnityInverseReverseTable = (ulong*)malloc(sizeX * sizeY * sizeof(ulong));
    for (uint32_t i = 0; i < sizeX; i++) {
        long key = m_vectors[i].GetModulus().ConvertToInt<>();
        const auto& currentValues = inputMap_rootOfUnityInverseReverse.at(key);
        memcpy(&host_rootOfUnityInverseReverseTable[i * sizeY], currentValues.data(), sizeY * sizeof(ulong));
    }
}

void cudaPortalForSwitchFormat::flattenRootOfUnityInversePreconReverseTableByModulus(const std::vector<PolyImpl<NativeVector>>& m_vectors,
                                                                                     const std::map<ulong, std::vector<ulong>>& inputMap_rootOfUnityInversePrecon) {
    host_rootOfUnityInversePreconReverseTable = (ulong*)malloc(sizeX * sizeY * sizeof(ulong));
    for (uint32_t i = 0; i < sizeX; i++) {
        long key = m_vectors[i].GetModulus().ConvertToInt<>();
        const auto& currentValues = inputMap_rootOfUnityInversePrecon.at(key);
        memcpy(&host_rootOfUnityInversePreconReverseTable[i * sizeY], currentValues.data(), sizeY * sizeof(ulong));
    }
}

void cudaPortalForSwitchFormat::flattenCycloOrderInverseTableByModulus(const std::vector<PolyImpl<NativeVector>>& m_vectors,
                                                                       const uint32_t cyclotomicOrder,
                                                                       const std::map<ulong, std::vector<ulong>>& inputMap_cycloOrderInverseMap) {
    uint32_t CycloOrderHf = (cyclotomicOrder >> 1);
    uint32_t msb = lbcrypto::GetMSB64(CycloOrderHf - 1);
    host_cycloOrderInverseTable = (ulong*)malloc(sizeX * sizeof(ulong));
    for (uint32_t i = 0; i < sizeX; i++) {
        long key = m_vectors[i].GetModulus().ConvertToInt<>();
        const auto& currentValues = inputMap_cycloOrderInverseMap.at(key);
        host_cycloOrderInverseTable[i] = currentValues.at(msb);
    }
}

void cudaPortalForSwitchFormat::flattenCycloOrderInversePreconTableByModulus(const std::vector<PolyImpl<NativeVector>>& m_vectors,
                                                                             const uint32_t cyclotomicOrder,
                                                                             const std::map<ulong, std::vector<ulong>>& inputMap_cycloOrderInversePreconMap) {
    uint32_t CycloOrderHf = (cyclotomicOrder >> 1);
    uint32_t msb = lbcrypto::GetMSB64(CycloOrderHf - 1);
    host_cycloOrderInversePreconTable = (ulong*)malloc(sizeX * sizeof(ulong));
    for (uint32_t i = 0; i < sizeX; i++) {
        long key = m_vectors[i].GetModulus().ConvertToInt<>();
        const auto& currentValues = inputMap_cycloOrderInversePreconMap.at(key);
        host_cycloOrderInversePreconTable[i] = currentValues.at(msb);
    }
}

void cudaPortalForSwitchFormat::marshalTwiddleFactors(const std::vector<PolyImpl<NativeVector>>& m_vectors,
                                                      const std::map<ulong, std::vector<ulong>>& inputMap_rootOfUnityReverse,
                                                      const std::map<ulong, std::vector<ulong>>& inputMap_rootOfUnityPrecon) {
    flattenRootOfUnityReverseTableByModulus(m_vectors, inputMap_rootOfUnityReverse);
    flattenRootOfUnityPreconReverseTableByModulus(m_vectors, inputMap_rootOfUnityPrecon);
}

void cudaPortalForSwitchFormat::marshalInvTwiddleFactors(const std::vector<PolyImpl<NativeVector>>& m_vectors,
                                            const std::map<ulong, std::vector<ulong>>& inputMap_rootOfUnityInverseReverse,
                                            const std::map<ulong, std::vector<ulong>>& inputMap_rootOfUnityInversePrecon,
                                            const uint32_t cyclotomicOrder,
                                            const std::map<ulong, std::vector<ulong>>& inputMap_cycloOrderInverseMap,
                                            const std::map<ulong, std::vector<ulong>>& inputMap_cycloOrderInversePreconMap) {

    flattenRootOfUnityInverseReverseTableByModulus(m_vectors, inputMap_rootOfUnityInverseReverse);
    flattenRootOfUnityInversePreconReverseTableByModulus(m_vectors, inputMap_rootOfUnityInversePrecon);
    flattenCycloOrderInverseTableByModulus(m_vectors, cyclotomicOrder, inputMap_cycloOrderInverseMap);
    flattenCycloOrderInversePreconTableByModulus(m_vectors, cyclotomicOrder, inputMap_cycloOrderInversePreconMap);
}

void cudaPortalForSwitchFormat::copyInTwiddleFactors() {
    size_t sizeRootOfUnityParams = sizeX * sizeY * sizeof(ulong);

    cudaMallocAsync((void**)&device_rootOfUnityReverseTable, sizeRootOfUnityParams, stream);
    cudaMallocAsync((void**)&device_rootOfUnityPreconReverseTable, sizeRootOfUnityParams, stream);

    cudaMemcpyAsync(device_rootOfUnityReverseTable, host_rootOfUnityReverseTable, sizeRootOfUnityParams, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_rootOfUnityPreconReverseTable, host_rootOfUnityPreconReverseTable, sizeRootOfUnityParams, cudaMemcpyHostToDevice, stream);
}


void cudaPortalForSwitchFormat::copyInInvTwiddleFactors() {
    size_t sizeRootOfUnityParams = sizeX * sizeY * sizeof(ulong);
    size_t sizeCycloOrderParams = sizeX * sizeof(ulong);

    cudaMallocAsync((void**)&device_rootOfUnityInverseReverseTable, sizeRootOfUnityParams, stream);
    cudaMallocAsync((void**)&device_rootOfUnityInversePreconReverseTable, sizeRootOfUnityParams, stream);
    cudaMallocAsync((void**)&device_cycloOrderInverseTable, sizeCycloOrderParams, stream);
    cudaMallocAsync((void**)&device_cycloOrderInversePreconTable, sizeCycloOrderParams, stream);

    cudaMemcpyAsync(device_rootOfUnityInverseReverseTable, host_rootOfUnityInverseReverseTable, sizeRootOfUnityParams, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_rootOfUnityInversePreconReverseTable, host_rootOfUnityInversePreconReverseTable, sizeRootOfUnityParams, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_cycloOrderInverseTable, host_cycloOrderInverseTable, sizeCycloOrderParams, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_cycloOrderInversePreconTable, host_cycloOrderInversePreconTable, sizeCycloOrderParams, cudaMemcpyHostToDevice, stream);
}

// Forward NTT
void cudaPortalForSwitchFormat::switchFormatToEvaluation() {
    uint32_t n = sizeY;

    int totalButterflyOps = n >> 1;
    int threadsPerBlock = 512;

    dim3 blocks = dim3(totalButterflyOps / threadsPerBlock , 1U, 1U); // Set the grid dimensions
    dim3 threads = dim3(512, 1U, 1U); // Set the block dimensions

    for (uint32_t s = 0; s < sizeX; s++) {

        for (uint32_t m = 1; m < n; m <<= 1) {
            uint32_t step = (n/m) >> 1;

            void *args[] = {
                &s, &m, &n, &step,
                &device_m_vectors, &sizeX, &sizeY,
                &device_rootOfUnityReverseTable,
                &device_rootOfUnityPreconReverseTable
            };
            fNTTKernelWrapper(blocks, threads, args, stream);
        }
    }
}

/**
 * Inverse NTT
 */
void cudaPortalForSwitchFormat::switchFormatToCoefficient() {
    int n = sizeY;

    int totalButterflyOps = n >> 1;
    int threadsPerBlock = 512;

    dim3 blocksPt1 = dim3(totalButterflyOps / threadsPerBlock , 1U, 1U); // Set the grid dimensions
    dim3 blocksPt2 = dim3(n / threadsPerBlock, 1U, 1U);
    dim3 threads = dim3(512, 1U, 1U); // Set the block dimensions

    //std::cout << "inverse NTT for n = " << ringDim << " with " << blocksPt1.x << " blocks, " << threads.x << " threads/block." << std::endl;

    for (uint32_t x = 0; x < sizeX; x++) {

        //uint32_t p_offset = p * sizeY;
        //ulong* modulus = &device_partP_empty_m_vectors[sizeX * sizeY + p];
        //ulong modulus = approxModDownData->

        for (uint32_t m = (n >> 1); m >= 1; m >>= 1) {

            uint32_t step = (n/m) >> 1;

            void *argsPt1[] = {
                &x, &m, &n, &step,
                &device_m_vectors, &sizeX, &sizeY,
                &device_rootOfUnityInverseReverseTable,
                &device_rootOfUnityInversePreconReverseTable
            };
            iNTTPart1KernelWrapper(blocksPt1, threads, argsPt1, stream);
        }

        void *argsPt2[] = {
            &x,
            &device_m_vectors, &sizeX, &sizeY,
            &device_cycloOrderInverseTable,
            &device_cycloOrderInversePreconTable
        };
        iNTTPart2KernelWrapper(blocksPt2, threads, argsPt2, stream);
    }
}

void cudaPortalForSwitchFormat::invokeSwitchFormatKernel(Format format) {
    if (format == EVALUATION) {
        // forward NTT
        switchFormatToEvaluation();
    } else {
        // inverse NTT
        switchFormatToCoefficient();
    }
}


}