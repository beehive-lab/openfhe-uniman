#ifndef OPENFHE_CUDA_DATA_UTILS_H
#define OPENFHE_CUDA_DATA_UTILS_H

#include <cstdint> // for uint32_t type
#include "lattice/poly.h"
#include "cuda-utils/kernel-headers/approx-switch-crt-basis.cuh"
#include "cuda-utils/m_vectors.h"


// use this namespace in order to access openFHE internal data types
namespace lbcrypto {

/**
 *  A macro that checks the return value of CUDA calls and reports errors.
 * @param call
 */
#define CUDA_CHECK(call) do {                                    \
    cudaError_t err = call;                                      \
    if (err != cudaSuccess) {                                    \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(err));    \
        exit(EXIT_FAILURE);                                      \
    }                                                            \
} while (0)

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

    cudaStream_t getParamsStream() const { return paramsStream; }
    cudaStream_t getWorkDataStream0() const { return workDataStream0; }
    cudaStream_t getWorkDataStream1() const { return workDataStream1; }


private:

    int gpuBlocks;
    int gpuThreads;

    cudaStream_t        paramsStream;
    cudaStream_t        workDataStream0;
    cudaStream_t        workDataStream1;

    // private constructor to prevent instantatiation from outside
    cudaDataUtils() {
        gpuBlocks = 0;
        gpuThreads = 0;

        createCUDAStreams();
    }

    /**
     * @brief Destructor to ensure proper cleanup of CUDA resources.
     *
     * This destructor is responsible for destroying the CUDA streams when the singleton
     * instance is destroyed, preventing resource leaks.
     */
    ~cudaDataUtils() {
        destroyCUDAStreams();
    }

    void createCUDAStreams() {
        cudaError_t err;
        err = cudaStreamCreate(&paramsStream);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA error during paramsStream creation: " + std::string(cudaGetErrorString(err)));
        }
        err = cudaStreamCreate(&workDataStream0);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA error during workDataStream0 creation: " + std::string(cudaGetErrorString(err)));
        }
        err = cudaStreamCreate(&workDataStream1);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA error during workDataStream1 creation: " + std::string(cudaGetErrorString(err)));
        }
    }

    void destroyCUDAStreams() {
        cudaError_t err;
        if (paramsStream) {
            err = cudaStreamDestroy(paramsStream);
            if (err != cudaSuccess) {
                std::cerr << "CUDA error during paramsStream destruction: "
                          << cudaGetErrorString(err) << std::endl;
            }
        }

        if (workDataStream0) {
            err = cudaStreamDestroy(workDataStream0);
            if (err != cudaSuccess) {
                std::cerr << "CUDA error during workDataStream0 destruction: "
                          << cudaGetErrorString(err) << std::endl;
            }
        }

        if (workDataStream1) {
            err = cudaStreamDestroy(workDataStream1);
            if (err != cudaSuccess) {
                std::cerr << "CUDA error during workDataStream1 destruction: "
                          << cudaGetErrorString(err) << std::endl;
            }
        }
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
