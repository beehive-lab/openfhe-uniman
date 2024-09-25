#ifndef CUDAPORTALFORAPPROXMODDOWN_H
#define CUDAPORTALFORAPPROXMODDOWN_H

//#include "math/hal.h"
//#include "lattice/poly.h"
#include <cstdint> // for uint32_t type
//#include <lattice/hal/default/lat-backend-default.h>
#include <cuda_runtime.h>
#include <cuda-utils/cuda-data-utils.h>

#include "cudaPortalForParamsData.h"
#include "cuda-utils/kernel-headers/approx-switch-crt-basis.cuh"

namespace lbcrypto {
    class Params;

    using PolyType = PolyImpl<NativeVector>;

class cudaPortalForApproxModDown {

private:

    std::shared_ptr<cudaPortalForParamsData> paramsData;

    cudaStream_t stream;

    //uint32_t ringDim;
    //uint32_t sizeP;
    //uint32_t sizeQ;

    m_vectors_struct*   host_m_vectors;
    m_vectors_struct*   host_ans_m_vectors;

    uint128_t*          device_sum;

    m_vectors_struct*   device_m_vectors;
    m_vectors_struct*   device_ans_m_vectors;

    //unsigned long** device_m_vectors_data_ptr_0;
    unsigned long** device_m_vectors_data_ptr;
    unsigned long** device_ans_m_vectors_data_ptr;

    void destroyCUDAStream();
    void freeHostMemory(uint32_t sizeP, uint32_t sizeQ);
    void freeDeviceMemory(uint32_t sizeP, uint32_t sizeQ);

public:

    //constructor
    cudaPortalForApproxModDown(uint32_t ringDim, uint32_t sizeP, uint32_t sizeQ, std::shared_ptr<cudaPortalForParamsData> params_data);

    ~cudaPortalForApproxModDown();

    void print_host_m_vectors();

    // getters setters
    cudaStream_t getStream();

    std::shared_ptr<cudaPortalForParamsData> getParamsData();


    uint128_t*          getDevice_sum();

    m_vectors_struct*   getDevice_m_vectors();
    m_vectors_struct*   getDevice_ans_m_vectors();

    m_vectors_struct*   getHost_ans_m_vectors();

    // allocate host buffers

    void allocateHostData(uint32_t ringDim, uint32_t sizeP, uint32_t sizeQ);

    // marshal


    void marshalWorkData(uint32_t ringDim, uint32_t sizeP, uint32_t sizeQ,
                         const std::vector<PolyImpl<NativeVector>> m_vectors,
                         const std::vector<PolyImpl<NativeVector>> ans_m_vectors);
    /*void marshalWorkData(const std::vector<PolyImpl<NativeVector>> m_vectors,
                         const std::vector<PolyImpl<NativeVector>> ans_m_vectors,
                         const cudaStream_t stream);*/

    // transfer


    void copyInWorkData(uint32_t ringDim, uint32_t sizeP, uint32_t sizeQ);

    void copyOutResult(uint32_t ringDim, uint32_t sizeP);

    // kernel invocation
    void callApproxSwitchCRTBasisKernel_Simple(int gpuBlocks, int gpuThreads,
                                                  uint32_t ringDim, uint32_t sizeP, uint32_t sizeQ);

    //template <typename VecType>
    //DCRTPolyImpl<VecType> unmarshal(uint32_t ringDim, uint32_t sizeP, std::vector<PolyImpl<NativeVector>>& ans_m_vectors, m_vectors_struct*  host_ans_m_vectors);
    void unmarshal(uint32_t ringDim, uint32_t sizeP, std::vector<PolyImpl<NativeVector>>& ans_m_vectors);

    // etc
    //void assignHostPointers(m_vectors_struct*&  host_m_vectors, m_vectors_struct*&  host_ans_m_vectors, const cudaStream_t stream);
    //void assignDevicePointers(m_vectors_struct*&  device_m_vectors, m_vectors_struct*&  device_ans_m_vectors, const cudaStream_t stream);
    //void assignSumPointers(uint128_t*&  device_sum, const cudaStream_t stream);

};

}

#endif //CUDAPORTALFORAPPROXMODDOWN_H
