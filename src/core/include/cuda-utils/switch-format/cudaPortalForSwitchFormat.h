#ifndef CUDAPORTALFORSWITCHFORMAT_H
#define CUDAPORTALFORSWITCHFORMAT_H

#include <cstdint>

#include "lattice/poly.h"

#include <cuda_runtime.h>
#include "cuda-utils/approxModDown/cudaPortalForApproxModDownData.h"
#include <cuda-utils/kernel-headers/switch-format.cuh>

namespace lbcrypto {

    using PolyType = PolyImpl<NativeVector>;

class cudaPortalForSwitchFormat {

    std::shared_ptr<cudaPortalForApproxModDownData> approxModDownData;

    cudaStream_t stream;

    uint32_t ringDim;
    uint32_t sizeP;
    uint32_t cyclotomicOrder;
    //uint32_t partP_numOfElements;
    uint32_t valuesLength;

    //std::map<ulong, std::vector<ulong>> rootOfUnityInverseReverseTable;
    //std::map<ulong, std::vector<ulong>> rootOfUnityInversePreconReverseTable;
    //std::map<ulong, std::vector<ulong>> cycloOrderInverseTableByModulus;
    //std::map<ulong, std::vector<ulong>> cycloOrderInversePreconTableByModulus;
    // rootOfUnityInverseReverseTable
    ulong* host_rootOfUnityInverseReverseTable;
    ulong* host_rootOfUnityInversePreconReverseTable;
    ulong* host_cycloOrderInverseTable;
    ulong* host_cycloOrderInversePreconTable;

    ulong* device_rootOfUnityInverseReverseTable;
    ulong* device_rootOfUnityInversePreconReverseTable;
    ulong* device_cycloOrderInverseTable;
    ulong* device_cycloOrderInversePreconTable;

public:

    // Constructor
    //cudaPortalForSwitchFormat(uint32_t cyclotomicOrder, uint32_t partP_numOfElements);
    cudaPortalForSwitchFormat(std::shared_ptr<cudaPortalForApproxModDownData> data);

    // Destructor
    ~cudaPortalForSwitchFormat();

    // Getters
    uint32_t get_cyclotomicOrder();
    ulong* get_device_rootOfUnityInverseReverseTable();
    ulong* get_device_rootOfUnityInversePreconReverseTable();
    ulong* get_device_cycloOrderInverseTable();
    ulong* get_device_cycloOrderInversePreconMap();

    // Setters
    void set_approxModDownDataPtr(std::shared_ptr<cudaPortalForApproxModDownData> ptr) { this->approxModDownData = ptr;}

    // Marshalling Functions
    void marshalTwiddleFactors(const std::vector<PolyImpl<NativeVector>>& partP_m_vectors,
                     const std::map<ulong, std::vector<ulong>>& inputMap_rootOfUnityInverseReverse,
                     const std::map<ulong, std::vector<ulong>>& inputMap_rootOfUnityInversePrecon,
                     const uint32_t cyclotomicOrder,
                     const std::map<ulong, std::vector<ulong>>& inputMap_cycloOrderInverseMap,
                     const std::map<ulong, std::vector<ulong>>& inputMap_cycloOrderInversePreconMap);

    // Data Transfer Functions
    void copyInTwiddleFactors();

    // Kernel Invocation Functions
    void invokeSwitchFormatKernel(int toCoefficient);

private:

    // Flatten Map Functions (are called by marshalling function)
    void flattenRootOfUnityInverseReverseTableByModulus(const std::vector<PolyImpl<NativeVector>>& partP_m_vectors,
                                                        const std::map<ulong, std::vector<ulong>>& inputMap_rootOfUnityInverseReverse);

    void flattenRootOfUnityInversePreconReverseTableByModulus(const std::vector<PolyImpl<NativeVector>>& partP_m_vectors,
                                                              const std::map<ulong, std::vector<ulong>>& inputMap_rootOfUnityInversePrecon);

    void flattenCycloOrderInverseTableByModulus(const std::vector<PolyImpl<NativeVector>>& partP_m_vectors,
                                                const uint32_t cyclotomicOrder,
                                                const std::map<ulong, std::vector<ulong>>& inputMap_cycloOrderInverseMap);

    void flattenCycloOrderInversePreconTableByModulus(const std::vector<PolyImpl<NativeVector>>& partP_m_vectors,
                                                      const uint32_t cyclotomicOrder,
                                                      const std::map<ulong, std::vector<ulong>>& inputMap_cycloOrderInversePreconMap);

    void switchFormatToCoefficient();
    void switchFormatToEvaluation();

    void freeHostMemory();
    void freeDeviceMemory();

};
}

#endif //CUDAPORTALFORSWITCHFORMAT_H
