#ifndef CUDAPORTALFORAPPROXMODDOWN_H
#define CUDAPORTALFORAPPROXMODDOWN_H

#include <cstdint> // for uint32_t type
#include <cuda_runtime.h>
#include <cuda-utils/cuda-data-utils.h>

#include "cudaPortalForParamsData.h"
#include "cuda-utils/kernel-headers/approx-switch-crt-basis.cuh"

namespace lbcrypto {
    class Params;

    using PolyType = PolyImpl<NativeVector>;

class cudaPortalForApproxSwitchCRTBasis {

private:

    std::shared_ptr<cudaPortalForParamsData> paramsData;

    cudaStream_t stream;

    uint32_t ringDim;
    uint32_t sizeP;
    uint32_t sizeQ;

    m_vectors_struct*   host_m_vectors;
    m_vectors_struct*   host_ans_m_vectors;

    uint128_t*          device_sum;

    m_vectors_struct*   device_m_vectors;
    unsigned long**     device_m_vectors_data_ptr;

    m_vectors_struct*   device_ans_m_vectors;
    unsigned long**     device_ans_m_vectors_data_ptr;

public:

    // Constructor
    cudaPortalForApproxSwitchCRTBasis(std::shared_ptr<cudaPortalForParamsData> params_data);

    // Destructor
    ~cudaPortalForApproxSwitchCRTBasis();

    // Getter Functions
    cudaStream_t                                getStream() const;
    std::shared_ptr<cudaPortalForParamsData>    getParamsData() const;
    m_vectors_struct*                           getHost_ans_m_vectors() const;
    uint128_t*                                  getDevice_sum() const;
    m_vectors_struct*                           getDevice_m_vectors() const;
    m_vectors_struct*                           getDevice_ans_m_vectors() const;

    // Data Marshalling Functions
    void marshalWorkData(const std::vector<PolyImpl<NativeVector>>& m_vectors,
                         const std::vector<PolyImpl<NativeVector>>& ans_m_vectors);

    void unmarshalWorkData(std::vector<PolyImpl<NativeVector>>& ans_m_vectors);

    // Data Transfer Functions
    void copyInWorkData();

    void copyOutResult();

    // Kernel Invocation Function
    void invokeKernelOfApproxSwitchCRTBasis(int gpuBlocks, int gpuThreads);

    // Resources Allocation/Deallocation - Error Handling - Misc Functions

    void print_host_m_vectors();

private:
    void allocateHostData();
    static void handleFreeError(const std::string& operation, void* ptr);
    static void handleCUDAError(const std::string& operation, cudaError_t err);
    void createCUDAStream();
    void destroyCUDAStream();
    void freeHostMemory();
    void freeDeviceMemory();

};

}

#endif //CUDAPORTALFORAPPROXMODDOWN_H
