#include "math/hal.h"
#include "lattice/poly.h"
#include <cstdint> // for uint32_t type
#include <cassert>
#include "cuda-utils/cuda-data-utils.h"

// use this namespace in order to access openFHE internal data types
namespace lbcrypto {

using PolyType = PolyImpl<NativeVector>;

// constructor impl
cudaDataUtils::cudaDataUtils() {

}

// data marshaling methods
void cudaDataUtils::marshalDataForApproxSwitchCRTBasisKernel(
                                            // input
                                            uint32_t ringDim, uint32_t sizeQ, uint32_t sizeP,
                                            const std::vector<PolyImpl<NativeVector>> m_vectors,
                                            const std::vector<NativeInteger>& QHatInvModq,
                                            const std::vector<std::vector<NativeInteger>>& QHatModp,
                                            // output
                                            m_vectors_struct* my_m_vectors) const {
    //m_vectors_struct
    //m_vectors_struct* my_m_vectors = (m_vectors_struct*)malloc(sizeof(m_vectors_struct) * sizeQ);
    //assert( sizeof(my_m_vectors) == (sizeof(m_vectors_struct) * sizeQ) );
    // QHatInvModq
    unsigned long* qhatinvmodq = (unsigned long*)malloc(sizeQ * sizeof(unsigned long));
    // qhatmodp
    unsigned long* qhatmodp = (unsigned long*)malloc(sizeQ * sizeP * sizeof(unsigned long));

    // Iterate through m_vectors and extract m_values
    for (uint32_t q = 0; q < sizeQ; ++q) {
        // Dynamically allocate memory for vectorData
        my_m_vectors[q].data = (unsigned long*)malloc(sizeof(unsigned long) * ringDim);
        if (my_m_vectors[q].data == NULL) {
            printf("my_m_vectors[%d].data allocation failed\n", q);
        }
        // Populate the dynamically allocated array with extracted values
        for (size_t rd = 0; rd < ringDim; ++rd) {
            my_m_vectors[q].data[rd] = m_vectors[q][rd].template ConvertToInt<>();
        }
        // Set the modulus value associated with this vector
        my_m_vectors[q].modulus = m_vectors[q].GetModulus().ConvertToInt();
        // set the value of qHatInvModQ[q] to qhatinvmodq[q]
        qhatinvmodq[q] = QHatInvModq[q].ConvertToInt();
        // Set the values of QHatModp to qhatmodp
        for (usint sp = 0; sp < sizeP; sp++) {
            //qhatmodp[q][sp] = QHatModp[q][sp].ConvertToInt();
            qhatmodp[q * sizeP + sp] = QHatModp[q][sp].ConvertToInt();
        }
    }
    return;
}

}