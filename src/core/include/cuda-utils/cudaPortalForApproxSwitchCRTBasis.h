#ifndef CUDAPORTALFORAPPROXMODDOWN_H
#define CUDAPORTALFORAPPROXMODDOWN_H

#include <cstdint> // for uint32_t type
#include <iostream> // for printf

#include "lattice/poly.h"
#include <cuda_runtime.h>

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

    unsigned long*      host_m_vectors_data;
    unsigned long*      host_m_vectors_modulus;
    unsigned long*      host_ans_m_vectors_data;
    unsigned long*      host_ans_m_vectors_modulus;

    uint128_t*          device_sum;

    unsigned long*      device_m_vectors_data;
    unsigned long*      device_m_vectors_modulus;
    unsigned long*      device_ans_m_vectors_data;
    unsigned long*      device_ans_m_vectors_modulus;

public:

    // Constructor
    cudaPortalForApproxSwitchCRTBasis(std::shared_ptr<cudaPortalForParamsData> params_data, cudaStream_t workDataStream);

    // Destructor
    ~cudaPortalForApproxSwitchCRTBasis();

    // Getter Functions
    cudaStream_t                                getStream() const;
    std::shared_ptr<cudaPortalForParamsData>    getParamsData() const;
    unsigned long*                              getHost_ans_m_vectors_data() const;
    unsigned long*                              getHost_ans_m_vectors_modulus() const;
    uint128_t*                                  getDevice_sum() const;
    unsigned long*                              getDevice_m_vectors_data() const;
    unsigned long*                              getDevice_m_vectors_modulus() const;
    unsigned long*                              getDevice_ans_m_vectors_data() const;
    unsigned long*                              getDevice_ans_m_vectors_modulus() const;

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
    void freeHostMemory();
    void freeDeviceMemory();

};

}

#endif //CUDAPORTALFORAPPROXMODDOWN_H
