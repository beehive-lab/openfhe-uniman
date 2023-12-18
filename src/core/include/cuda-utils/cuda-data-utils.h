#ifndef OPENFHE_CUDA_DATA_UTILS_H
#define OPENFHE_CUDA_DATA_UTILS_H

#include "math/hal.h"
#include "cuda-utils/kernel-headers/approx-switch-crt-basis.cuh"
#include <cstdint> // for uint32_t type

// use this namespace in order to access openFHE internal data types
namespace lbcrypto {

using PolyType = PolyImpl<NativeVector>;

class cudaDataUtils {

public:

    // constructor
    cudaDataUtils();

    // data marshaling methods
    static void marshalDataForApproxSwitchCRTBasisKernel(uint32_t ringDim, uint32_t sizeQ, uint32_t sizeP,
                                                         const std::vector<PolyImpl<NativeVector>> m_vectors,
                                                         const std::vector<NativeInteger>& QHatInvModq,
                                                         const std::vector<NativeInteger>& QHatInvModqPrecon,
                                                         const std::vector<std::vector<NativeInteger>>& QHatModp,
                                                         const std::vector<PolyImpl<NativeVector>> ans_m_vectors,
                                                         m_vectors_struct*  host_m_vectors,
                                                         unsigned long*     host_QHatInvModq,
                                                         unsigned long*     host_QHatInvModqPrecon,
                                                         uint128_t*         host_QHatModp,
                                                         m_vectors_struct*  host_ans_m_vectors);

    // deallocations
    static void DeallocateMemoryForApproxSwitchCRTBasisKernel(int sizeQ,
                                                              m_vectors_struct* host_m_vectors,
                                                              unsigned long*    host_QHatInvModq,
                                                              unsigned long*    host_QHatInvModqPrecon,
                                                              uint128_t *       host_QHatModp,
                                                              uint128_t*        host_sum);

};

}

#endif  //OPENFHE_CUDA_DATA_UTILS_H
