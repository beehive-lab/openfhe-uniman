#ifndef OPENFHE_CUDA_DATA_UTILS_H
#define OPENFHE_CUDA_DATA_UTILS_H

#include <cstdint> // for uint32_t type
#include "lattice/poly.h"
#include "cuda-utils/kernel-headers/approx-switch-crt-basis.cuh"


// use this namespace in order to access openFHE internal data types
namespace lbcrypto {

using PolyType = PolyImpl<NativeVector>;

/**
 * @class cudaDataUtils
 * @brief Singleton class for centralized management of CUDA configurations.
 */
class cudaDataUtils {

public:
    /**
     * @brief Returns the singleton instance of cudaDataUtils.
     *
     * This function provides access to the single instance of the class.
     * The instance is created when first called and reused afterward.
     * Currently is instatiated it in the application.
     *
     * @return cudaDataUtils& The singleton instance.
     */

    static cudaDataUtils& getInstance() {
        static cudaDataUtils instance; // Guaranteed to be destroyed and instantiated on first use
        return instance;
    }

    void setGpuBlocks(const int blocks) { gpuBlocks = blocks; }
    void setGpuThreads(const int threads) { gpuThreads = threads; }

    int getGpuBlocks() const { return gpuBlocks; }
    int getGpuThreads() const { return gpuThreads; }


private:

    int gpuBlocks;
    int gpuThreads;

    // private constructor to prevent instantatiation from outside
    cudaDataUtils() {
        gpuBlocks = 0;
        gpuThreads = 0;
    }


    // Delete copy constructor and assignment operator to prevent copying of the singleton instance
    cudaDataUtils(const cudaDataUtils&) = delete;
    cudaDataUtils& operator=(const cudaDataUtils&) = delete;

public:

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

    // Misc Functions
    static int isValid(uint32_t ringDim, uint32_t sizeP,
                               const std::vector<PolyImpl<NativeVector>> ans_m_vectors,
                               m_vectors_struct*  host_ans_m_vectors);

    static void printUint128(uint128_t value);

    static void printParams(uint32_t sizeQ, uint32_t sizeP,
                                    const unsigned long* host_QHatInvModq,
                                    const unsigned long* host_QHatInvModqPrecon,
                                    const uint128_t* host_QHatModp,
                                    const uint128_t* host_modpBarrettMu);

    static void print_host_m_vectors(uint32_t sizeQ, m_vectors_struct*  host_m_vectors);

};

}

#endif  //OPENFHE_CUDA_DATA_UTILS_H
