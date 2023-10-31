#ifndef OPENFHE_CUDA_DATA_UTILS_H
#define OPENFHE_CUDA_DATA_UTILS_H

#include "math/hal.h"
#include <cstdint> // for uint32_t type

/*
 * Data type for m_vectors object
 */
struct m_vectors_struct {
    unsigned long* data;
    unsigned long modulus;
};

// use this namespace in order to access openFHE internal data types
namespace lbcrypto {

using PolyType = PolyImpl<NativeVector>;

class cudaDataUtils {

public:

    // constructor
    cudaDataUtils();

    // data marshaling methods
    void marshalDataForApproxSwitchCRTBasisKernel(
        // input
        uint32_t ringDim, uint32_t sizeQ, uint32_t sizeP,
        const std::vector<PolyImpl<NativeVector>> m_vectors,
        const std::vector<NativeInteger>& QHatInvModq,
        const std::vector<std::vector<NativeInteger>>& QHatModp,
        // output
        m_vectors_struct* my_m_vectors) const;

};

}

#endif  //OPENFHE_CUDA_DATA_UTILS_H
