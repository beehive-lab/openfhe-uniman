#include "cuda-utils/switch-format/cudaPortalForSwitchFormat.h"

namespace lbcrypto {

//cudaPortalForSwitchFormat::cudaPortalForSwitchFormat(uint32_t cyclotomicOrder, uint32_t partP_numOfElements) {
cudaPortalForSwitchFormat::cudaPortalForSwitchFormat(std::shared_ptr<cudaPortalForApproxModDownData> data)
    :approxModDownData(data){
    uint32_t rd = data->getParamsData()->get_RingDim();
    this->ringDim = rd;
    this->sizeP = data->getParamsData()->get_sizeP();
    this->cyclotomicOrder = rd << 1;
    //this->partP_numOfElements = partP_numOfElements;
    this->valuesLength = rd; // = ringDim
    //this->approxModDownData = data;
    this->stream = data->getStream();
}

cudaPortalForSwitchFormat::~cudaPortalForSwitchFormat() {
    free(host_rootOfUnityInverseReverseTable);
    free(host_rootOfUnityInversePreconReverseTable);
    free(host_cycloOrderInverseTable);
    free(host_cycloOrderInversePreconTable);

    cudaFree(device_rootOfUnityInverseReverseTable);
    cudaFree(device_rootOfUnityInversePreconReverseTable);
    cudaFree(device_cycloOrderInverseTable);
    cudaFree(device_cycloOrderInversePreconTable);
}

// Getters
uint32_t cudaPortalForSwitchFormat::get_cyclotomicOrder() { return cyclotomicOrder; }

ulong* cudaPortalForSwitchFormat::get_device_rootOfUnityInverseReverseTable()       { return device_rootOfUnityInverseReverseTable; }
ulong* cudaPortalForSwitchFormat::get_device_rootOfUnityInversePreconReverseTable() { return device_rootOfUnityInversePreconReverseTable; }
ulong* cudaPortalForSwitchFormat::get_device_cycloOrderInverseTable()               { return device_cycloOrderInverseTable; }
ulong* cudaPortalForSwitchFormat::get_device_cycloOrderInversePreconMap()           { return device_cycloOrderInversePreconTable; }

// Flatten Map Functions
void cudaPortalForSwitchFormat::flattenRootOfUnityInverseReverseTableByModulus(const std::vector<PolyImpl<NativeVector>>& partP_m_vectors,
                                                                               const std::map<ulong, std::vector<ulong>>& inputMap_rootOfUnityInverseReverse) {
    host_rootOfUnityInverseReverseTable = (ulong*)malloc(sizeP * valuesLength * sizeof(ulong));
    for (uint32_t i = 0; i < sizeP; i++) {
        long key = partP_m_vectors[i].GetModulus().ConvertToInt<>();
        const auto& currentValues = inputMap_rootOfUnityInverseReverse.at(key);
        memcpy(&host_rootOfUnityInverseReverseTable[i * valuesLength], currentValues.data(), valuesLength * sizeof(ulong));
    }
}

void cudaPortalForSwitchFormat::flattenRootOfUnityInversePreconReverseTableByModulus(const std::vector<PolyImpl<NativeVector>>& partP_m_vectors,
                                                                                     const std::map<ulong, std::vector<ulong>>& inputMap_rootOfUnityInversePrecon) {
    host_rootOfUnityInversePreconReverseTable = (ulong*)malloc(sizeP * valuesLength * sizeof(ulong));
    for (uint32_t i = 0; i < sizeP; i++) {
        long key = partP_m_vectors[i].GetModulus().ConvertToInt<>();
        const auto& currentValues = inputMap_rootOfUnityInversePrecon.at(key);
        memcpy(&host_rootOfUnityInversePreconReverseTable[i * valuesLength], currentValues.data(), valuesLength * sizeof(ulong));
    }
}

void cudaPortalForSwitchFormat::flattenCycloOrderInverseTableByModulus(const std::vector<PolyImpl<NativeVector>>& partP_m_vectors,
                                                                       const uint32_t cyclotomicOrder,
                                                                       const std::map<ulong, std::vector<ulong>>& inputMap_cycloOrderInverseMap) {
    uint32_t CycloOrderHf = (cyclotomicOrder >> 1);
    uint32_t msb = lbcrypto::GetMSB64(CycloOrderHf - 1);
    host_cycloOrderInverseTable = (ulong*)malloc(sizeP * sizeof(ulong));
    for (uint32_t i = 0; i < sizeP; i++) {
        long key = partP_m_vectors[i].GetModulus().ConvertToInt<>();
        const auto& currentValues = inputMap_cycloOrderInverseMap.at(key);
        host_cycloOrderInverseTable[i] = currentValues.at(msb);
    }
}

void cudaPortalForSwitchFormat::flattenCycloOrderInversePreconTableByModulus(const std::vector<PolyImpl<NativeVector>>& partP_m_vectors,
                                                                             const uint32_t cyclotomicOrder,
                                                                             const std::map<ulong, std::vector<ulong>>& inputMap_cycloOrderInversePreconMap) {
    uint32_t CycloOrderHf = (cyclotomicOrder >> 1);
    uint32_t msb = lbcrypto::GetMSB64(CycloOrderHf - 1);
    host_cycloOrderInversePreconTable = (ulong*)malloc(sizeP * sizeof(ulong));
    for (uint32_t i = 0; i < sizeP; i++) {
        long key = partP_m_vectors[i].GetModulus().ConvertToInt<>();
        const auto& currentValues = inputMap_cycloOrderInversePreconMap.at(key);
        host_cycloOrderInversePreconTable[i] = currentValues.at(msb);
        std::cout << "(portal-switch-format) cycloOrderInversePreconTable[" << i << "] = " << host_cycloOrderInversePreconTable[i] << std::endl;
    }
}

void cudaPortalForSwitchFormat::marshalTwiddleFactors(const std::vector<PolyImpl<NativeVector>>& partP_m_vectors,
                                            const std::map<ulong, std::vector<ulong>>& inputMap_rootOfUnityInverseReverse,
                                            const std::map<ulong, std::vector<ulong>>& inputMap_rootOfUnityInversePrecon,
                                            const uint32_t cyclotomicOrder,
                                            const std::map<ulong, std::vector<ulong>>& inputMap_cycloOrderInverseMap,
                                            const std::map<ulong, std::vector<ulong>>& inputMap_cycloOrderInversePreconMap) {

    flattenRootOfUnityInverseReverseTableByModulus(partP_m_vectors, inputMap_rootOfUnityInverseReverse);
    flattenRootOfUnityInversePreconReverseTableByModulus(partP_m_vectors, inputMap_rootOfUnityInversePrecon);
    flattenCycloOrderInverseTableByModulus(partP_m_vectors, cyclotomicOrder, inputMap_cycloOrderInverseMap);
    flattenCycloOrderInversePreconTableByModulus(partP_m_vectors, cyclotomicOrder, inputMap_cycloOrderInversePreconMap);

}

void cudaPortalForSwitchFormat::copyInTwiddleFactors() {
    size_t sizeRootOfUnityParams = sizeP * valuesLength * sizeof(ulong);
    size_t sizeCycloOrderParams = sizeP * sizeof(ulong);

    cudaMalloc((void**)&device_rootOfUnityInverseReverseTable, sizeRootOfUnityParams);
    cudaMalloc((void**)&device_rootOfUnityInversePreconReverseTable, sizeRootOfUnityParams);
    cudaMalloc((void**)&device_cycloOrderInverseTable, sizeCycloOrderParams);
    cudaMalloc((void**)&device_cycloOrderInversePreconTable, sizeCycloOrderParams);

    cudaMemcpy(device_rootOfUnityInverseReverseTable, host_rootOfUnityInverseReverseTable, sizeRootOfUnityParams, cudaMemcpyHostToDevice);
    cudaMemcpy(device_rootOfUnityInversePreconReverseTable, host_rootOfUnityInversePreconReverseTable, sizeRootOfUnityParams, cudaMemcpyHostToDevice);
    cudaMemcpy(device_cycloOrderInverseTable, host_cycloOrderInverseTable, sizeCycloOrderParams, cudaMemcpyHostToDevice);
    cudaMemcpy(device_cycloOrderInversePreconTable, host_cycloOrderInversePreconTable, sizeCycloOrderParams, cudaMemcpyHostToDevice);
}

void cudaPortalForSwitchFormat::switchFormatToCoefficient() {

    ulong* device_partP_empty_m_vectors = approxModDownData->getDevice_partP_empty_m_vectors();
    uint32_t sizeX = approxModDownData->get_partP_empty_size_x();
    uint32_t sizeY = approxModDownData->get_partP_empty_size_y();

    int n = ringDim;

    int totalButterflyOps = n >> 1;
    int threadsPerBlock = 512;

    dim3 blocksPt1 = dim3(totalButterflyOps / threadsPerBlock , 1U, 1U); // Set the grid dimensions
    dim3 blocksPt2 = dim3(n / threadsPerBlock, 1U, 1U);
    dim3 threads = dim3(512, 1U, 1U); // Set the block dimensions

    //std::cout << "inverse NTT for n = " << ringDim << " with " << blocksPt1.x << " blocks, " << threads.x << " threads/block." << std::endl;

    for (uint32_t p = 0; p < sizeP; p++) {

        //uint32_t p_offset = p * sizeY;
        //ulong* modulus = &device_partP_empty_m_vectors[sizeX * sizeY + p];
        //ulong modulus = approxModDownData->

        for (uint32_t m = (n >> 1); m >= 1; m >>= 1) {

            uint32_t step = (n/m) >> 1;

            void *argsPt1[] = {
                &p, &m, &n, &step,
                &device_partP_empty_m_vectors, &sizeX, &sizeY,
                &device_rootOfUnityInverseReverseTable,
                &device_rootOfUnityInversePreconReverseTable
            };
            iNTTKernelWrapper(blocksPt1, threads, argsPt1, stream);
        }

        void *argsPt2[] = {
            &p,
            &device_partP_empty_m_vectors, &sizeX, &sizeY,
            &device_cycloOrderInverseTable,
            &device_cycloOrderInversePreconTable
        };
        iNTTPart2Wrapper(blocksPt2, threads, argsPt2, stream);
    }
}

void cudaPortalForSwitchFormat::switchFormatToEvaluation() {
    throw std::runtime_error("switchFormatToEvaluation() unimplemented\n");
}

void cudaPortalForSwitchFormat::invokeSwitchFormatKernel(int toCoefficient) {
    if (toCoefficient)
        switchFormatToCoefficient();
    else
        switchFormatToEvaluation();
}


}