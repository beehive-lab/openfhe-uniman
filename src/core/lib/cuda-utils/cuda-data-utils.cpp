#include <cassert>
#include "cuda-utils/cuda-data-utils.h"

#include <iomanip>

// use this namespace in order to access openFHE internal data types
namespace lbcrypto {

using PolyType = PolyImpl<NativeVector>;

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
    free(host_modpBarrettMu);
    free(host_ans_m_vectors);
}

// Misc Functions

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

void cudaDataUtils::printUint128(uint128_t value) {
    // Cast the higher and lower 64 bits of the uint128_t value
    uint64_t high = static_cast<uint64_t>(value >> 64); // Upper 64 bits
    uint64_t low = static_cast<uint64_t>(value);        // Lower 64 bits

    // Print the parts in hex or as two parts (high, low)
    std::cout << "0x" << std::hex << high << std::setw(16) << std::setfill('0') << low << std::dec;
}


void cudaDataUtils::printParams(uint32_t sizeQ, uint32_t sizeP,
                                        const unsigned long* host_QHatInvModq,
                                        const unsigned long* host_QHatInvModqPrecon,
                                        const uint128_t* host_QHatModp,
                                        const uint128_t* host_modpBarrettMu) {

    std::cout << "cudaDataUtils::printMarshalledData" << std::endl;

    // Print host_QHatInvModq
    std::cout << "host_QHatInvModq: ";
    for (uint32_t q = 0; q < sizeQ; ++q) {
        std::cout << host_QHatInvModq[q] << " ";
    }
    std::cout << std::endl;

    // Print host_QHatInvModqPrecon
    std::cout << "host_QHatInvModqPrecon: ";
    for (uint32_t q = 0; q < sizeQ; ++q) {
        std::cout << host_QHatInvModqPrecon[q] << " ";
    }
    std::cout << std::endl;

    // Print host_QHatModp (flattened 2D array)
    std::cout << "host_QHatModp: " << std::endl;
    for (uint32_t q = 0; q < sizeQ; ++q) {
        std::cout << "QHatModp[" << q << "]: ";
        for (uint32_t sp = 0; sp < sizeP; ++sp) {
            printUint128(host_QHatModp[q * sizeP + sp]);
            std::cout << " ";
        }
        std::cout << std::endl;
    }

    // Print host_modpBarrettMu
    std::cout << "host_modpBarrettMu: ";
    for (uint32_t sp = 0; sp < sizeP; ++sp) {
        printUint128(host_modpBarrettMu[sp]);
        std::cout << " ";
    }
    std::cout << std::endl;

}

void cudaDataUtils::print_host_m_vectors(uint32_t sizeQ, m_vectors_struct*  host_m_vectors) {

    std::cout << "cudaDataUtils::print_host_m_vectors" << std::endl;

    for (uint32_t q = 0; q < sizeQ; ++q) {
        std::cout << "host_m_vectors[" << q << "].data[0-3/ringDim]: ";
        for (uint32_t rd = 0; rd < 3; ++rd) {
            std::cout << host_m_vectors[q].data[rd] << " ";
        }
        std::cout << std::endl;
    }
}

}