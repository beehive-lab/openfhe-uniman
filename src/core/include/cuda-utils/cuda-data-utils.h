#ifndef OPENFHE_CUDA_DATA_UTILS_H
#define OPENFHE_CUDA_DATA_UTILS_H

#include "math/hal.h"
#include "cuda-utils/kernel-headers/approx-switch-crt-basis.cuh"
#include <cstdint> // for uint32_t type

// use this namespace in order to access openFHE internal data types
namespace lbcrypto {

using PolyType = PolyImpl<NativeVector>;

class cudaDataUtils {

private:
    static int gpuBlocks;
    static int gpuThreads;

public:

    // constructor
    cudaDataUtils();

    static void setGpuBlocks(int blocks);
    static void setGpuThreads(int threads);

    static int getGpuBlocks();
    static int getGpuThreads();

    // data marshaling methods
    static void marshalDataForApproxSwitchCRTBasisKernel(uint32_t ringDim, uint32_t sizeQ, uint32_t sizeP,
                                                         const std::vector<PolyImpl<NativeVector>> m_vectors,
                                                         const std::vector<NativeInteger>& QHatInvModq,
                                                         const std::vector<NativeInteger>& QHatInvModqPrecon,
                                                         const std::vector<std::vector<NativeInteger>>& QHatModp,
                                                         const std::vector<DoubleNativeInt>& modpBarrettMu,
                                                         const std::vector<PolyImpl<NativeVector>> ans_m_vectors,
                                                         m_vectors_struct*  host_m_vectors,
                                                         unsigned long*     host_QHatInvModq,
                                                         unsigned long*     host_QHatInvModqPrecon,
                                                         uint128_t*         host_QHatModp,
                                                         uint128_t*         host_modpBarrettMu,
                                                         m_vectors_struct*  host_ans_m_vectors);

    static void unmarshalDataForApproxSwitchCRTBasisKernel(uint32_t ringDim, uint32_t sizeP,
                                                           std::vector<PolyImpl<NativeVector>>& ans_m_vectors,
                                                           m_vectors_struct*  host_ans_m_vectors);

    // deallocations
    static void DeallocateMemoryForApproxSwitchCRTBasisKernel(int sizeQ,
                                                              m_vectors_struct* host_m_vectors,
                                                              unsigned long*    host_QHatInvModq,
                                                              unsigned long*    host_QHatInvModqPrecon,
                                                              uint128_t *       host_QHatModp,
                                                              uint128_t*        host_modpBarrettMu,
                                                              m_vectors_struct* host_ans_m_vectors);

    // validations
    static int isValid(uint32_t ringDim, uint32_t sizeP,
                               const std::vector<PolyImpl<NativeVector>> ans_m_vectors,
                               m_vectors_struct*  host_ans_m_vectors);

};

}

#endif  //OPENFHE_CUDA_DATA_UTILS_H
