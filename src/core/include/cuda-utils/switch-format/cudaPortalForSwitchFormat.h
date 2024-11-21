#ifndef CUDAPORTALFORSWITCHFORMAT_H
#define CUDAPORTALFORSWITCHFORMAT_H

#include <cstdint>

#include "lattice/poly.h"

#include <cuda_runtime.h>
#include <cuda-utils/cuda_util_macros.h>
#include <cuda-utils/kernel-headers/switch-format.cuh>

namespace lbcrypto {

    using PolyType = PolyImpl<NativeVector>;

class cudaPortalForSwitchFormat {

    std::shared_ptr<cudaPortalForApproxModDownData> approxModDownData;

    cudaStream_t stream;

    uint32_t cyclotomicOrder;
    ulong* device_m_vectors;
    uint32_t sizeX;
    uint32_t sizeY;

    // NTT twiddle factors
    ulong* host_rootOfUnityReverseTable;
    ulong* host_rootOfUnityPreconReverseTable;

    ulong* device_rootOfUnityReverseTable;
    ulong* device_rootOfUnityPreconReverseTable;

    // inverse NTT twiddle factors
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
    cudaPortalForSwitchFormat(ulong* device_m_vectors, uint32_t m_vectors_sizeX, uint32_t m_vectors_sizeY, cudaStream_t stream);

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
    // forward NTT
    void marshalTwiddleFactors(const std::vector<PolyImpl<NativeVector>>& partP_m_vectors,
                               const std::map<ulong, std::vector<ulong>>& inputMap_rootOfUnityReverse,
                               const std::map<ulong, std::vector<ulong>>& inputMap_rootOfUnityPrecon);
    // inverse NTT
    void marshalInvTwiddleFactors(const std::vector<PolyImpl<NativeVector>>& partP_m_vectors,
                                  const std::map<ulong, std::vector<ulong>>& inputMap_rootOfUnityInverseReverse,
                                  const std::map<ulong, std::vector<ulong>>& inputMap_rootOfUnityInversePrecon,
                                  const uint32_t cyclotomicOrder,
                                  const std::map<ulong, std::vector<ulong>>& inputMap_cycloOrderInverseMap,
                                  const std::map<ulong, std::vector<ulong>>& inputMap_cycloOrderInversePreconMap);

    // Data Transfer Functions
    // forward NTT
    void copyInTwiddleFactors();
    // inverse NTT
    void copyInInvTwiddleFactors();

    /**
     * Kernel Invocation Function
     * Invokes either NTT or inverse NTT CUDA kernel to switch format.
     * @param format EVALUATION -> forward NTT
     * @param format COEFFICIENT -> inverse NTT
     */
    void invokeSwitchFormatKernel(Format format);

private:

    // Flatten Map Functions (are called by marshalling functions)
    void flattenRootOfUnityReverseTableByModulus(const std::vector<PolyImpl<NativeVector>>& m_vectors,
                                                 const std::map<ulong, std::vector<ulong>>& inputMap_rootOfUnityReverse);

    void flattenRootOfUnityPreconReverseTableByModulus(const std::vector<PolyImpl<NativeVector>>& m_vectors,
                                                       const std::map<ulong, std::vector<ulong>>& inputMap_rootOfUnityPrecon);

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

    // forward NTT
    void switchFormatToEvaluation();
    // inverse NTT
    void switchFormatToCoefficient();

    void freeHostMemory();
    void freeDeviceMemory();

};
}

#endif //CUDAPORTALFORSWITCHFORMAT_H
