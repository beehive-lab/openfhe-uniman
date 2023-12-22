#include "math/hal.h"
#include "lattice/poly.h"
#include <cstdint> // for uint32_t type
#include <cassert>
#include "cuda-utils/cuda-data-utils.h"
#include "cuda-utils/kernel-headers/approx-switch-crt-basis.cuh"

// use this namespace in order to access openFHE internal data types
namespace lbcrypto {

using PolyType = PolyImpl<NativeVector>;

// constructor impl
cudaDataUtils::cudaDataUtils() {

}

// data marshaling methods
/**
 * Marshal data for approx-switch-crt-basis kernel.
 * Input:
 * @param ringDim
 * @param sizeQ
 * @param sizeP
 * @param m_vectors
 * @param QHatInvModq
 * @param QHatModp
 * Output:
 * @param host_m_vectors
 * @param host_qhatinvmodq
 * @param host_qhatmodp
 */
void cudaDataUtils::marshalDataForApproxSwitchCRTBasisKernel(uint32_t ringDim, uint32_t sizeQ, uint32_t sizeP,
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
                                                             m_vectors_struct*  host_ans_m_vectors) {
    // debugging:
    //std::cout << "==> marshal data" << std::endl;
    for (uint32_t q = 0; q < sizeQ; ++q) {
        for (uint32_t rd = 0; rd < ringDim; ++rd) {
            host_m_vectors[q].data[rd] = m_vectors[q][rd].template ConvertToInt<>();
        }
        host_m_vectors[q].modulus = m_vectors[q].GetModulus().ConvertToInt();
        host_QHatInvModq[q] = QHatInvModq[q].ConvertToInt();
        host_QHatInvModqPrecon[q] = QHatInvModqPrecon[q].ConvertToInt();
        for (uint32_t sp = 0; sp < sizeP; sp++) {
            host_QHatModp[q * sizeP + sp] = QHatModp[q][sp].ConvertToInt();
        }
    }
    for (uint32_t sp = 0; sp < sizeP; sp++) {
        host_modpBarrettMu[sp] = modpBarrettMu[sp];
        host_ans_m_vectors[sp].modulus = ans_m_vectors[sp].GetModulus().ConvertToInt();
    }
}

void cudaDataUtils::unmarshalDataForApproxSwitchCRTBasisKernel(uint32_t ringDim, uint32_t sizeP, std::vector<PolyImpl<NativeVector>>& ans_m_vectors, m_vectors_struct*  host_ans_m_vectors) {
    for (usint j = 0; j < sizeP; j++) {
        for(usint ri = 0; ri < ringDim; ri++) {
            ans_m_vectors[j][ri] = NativeInteger(host_ans_m_vectors[j].data[ri]);
        }
    }

}

void cudaDataUtils::DeallocateMemoryForApproxSwitchCRTBasisKernel(int sizeQ,
                                                                  m_vectors_struct* host_m_vectors,
                                                                  unsigned long*    host_QHatInvModq,
                                                                  unsigned long*    host_QHatInvModqPrecon,
                                                                  uint128_t *       host_QHatModp,
                                                                  uint128_t*        host_sum,
                                                                  uint128_t*        host_modpBarrettMu,
                                                                  m_vectors_struct* host_ans_m_vectors) {

    // debugging:
    //std::cout << "==> DeallocateMemoryForApproxSwitchCRTBasisKernel" << std::endl;
    for (int q = 0; q < sizeQ; ++q) {
        free(host_m_vectors[q].data);
    }
    free(host_m_vectors);
    free(host_QHatInvModq);
    free(host_QHatInvModqPrecon);
    free(host_QHatModp);
    free(host_sum);
    free(host_modpBarrettMu);
    free(host_ans_m_vectors);
}

int cudaDataUtils::isValid(uint32_t ringDim, uint32_t sizeP,
                           const std::vector<PolyImpl<NativeVector>> ans_m_vectors,
                           m_vectors_struct*  host_ans_m_vectors) {

    for (usint p = 0; p < sizeP; p++) {
        for (usint ri = 0; ri < ringDim; ri++) {
            if (ans_m_vectors[p][ri] != host_ans_m_vectors[p].data[ri]) {
                std::cout << "ans_m_vectors[" << p << "][" << ri << "] = " << ans_m_vectors[p][ri] << ", host_ans_m_vectors[" << p << "].data[" << ri << "] = " << host_ans_m_vectors[p].data[ri] << std::endl;
                return -1;
            }
        }
    }
    return 0;
}

}