#ifndef OPENFHE_CUDA_DATA_UTILS_H
#define OPENFHE_CUDA_DATA_UTILS_H

#include <cstdint> // for uint32_t type

#include "cuda_util_macros.h"
#include "cuda-utils/approxModDown/AMDBuffers.h"
#include "lattice/poly.h"
#include "cuda-utils/kernel-headers/approx-switch-crt-basis.cuh"
#include "cuda-utils/m_vectors.h"


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
    void setRingDim(const uint32_t rd) { ringDim = rd;}
    void setSizeP(const uint32_t size) { sizeP = size;}
    void setSizeQ(const uint32_t size) { sizeQ = size;}
    void setParamSizeY(const uint32_t size) { param_sizeY = size;}
    void setNumOfPipelineStreams(const int streams) { numOfPipelineStreams = streams; }

    int getGpuBlocks() const { return gpuBlocks; }
    int getGpuThreads() const { return gpuThreads; }

    cudaStream_t getParamsStream() const { return paramsStream; }
    cudaStream_t getWorkDataStream0() const { return workDataStream0; }
    cudaStream_t getWorkDataStream1() const { return workDataStream1; }
    cudaStream_t* getPipelineStreams0() const { return pipelineStreams0; }
    cudaStream_t* getPipelineStreams1() const { return pipelineStreams1; }

    cudaEvent_t  getWorkDataEvent0() const { return workDataEvent0; }
    cudaEvent_t  getWorkDataEvent1() const { return workDataEvent1; }
    cudaEvent_t* getEvents0() const { return events0; }
    cudaEvent_t* getEvents1() const { return events1; }
    AMDBuffers* getAMDBuffers0() const { return buffers0.get(); }
    AMDBuffers* getAMDBuffers1() const { return buffers1.get(); }


    void initialize(const int blocks, const int threads, const int streams, const uint32_t ringDim, const uint32_t sizeP, const uint32_t sizeQ, const uint32_t paramSizeY) {
        setGpuBlocks(blocks);
        setGpuThreads(threads);
        setNumOfPipelineStreams(streams);

        setRingDim(ringDim);
        setSizeP(sizeP);
        setSizeQ(sizeQ);
        setParamSizeY(paramSizeY);

        createCUDAStreams();
        initializeAMDBuffers();
    }

    void destroy() {
        destroyBuffers();
        destroyCUDAStreams();
    }


private:

    int gpuBlocks;
    int gpuThreads;

    // predefined by application. use with caution.
    uint32_t ringDim;
    // predefined by application. use with caution.
    uint32_t sizeP;
    // predefined by application. use with caution.
    uint32_t sizeQ;
    // predefined by application. use with caution.
    uint32_t param_sizeY;

    cudaStream_t        paramsStream;
    cudaStream_t        workDataStream0;
    cudaStream_t        workDataStream1;
    cudaEvent_t         workDataEvent0;
    cudaEvent_t         workDataEvent1;

    int                 numOfPipelineStreams;
    cudaStream_t*       pipelineStreams0;
    cudaStream_t*       pipelineStreams1;
    cudaEvent_t*        events0;
    cudaEvent_t*        events1;

    std::unique_ptr<AMDBuffers> buffers0;
    std::unique_ptr<AMDBuffers> buffers1;

    // private constructor to prevent instantatiation from outside
    cudaDataUtils() = default;

    /**
     * @brief Destructor to ensure proper cleanup of CUDA resources.
     *
     * This destructor is responsible for destroying the CUDA streams when the singleton
     * instance is destroyed, preventing resource leaks.
     */
    ~cudaDataUtils() = default;

    void createCUDAStreams() {
        // numOfPipelineStreams should have already been set
        if (numOfPipelineStreams == 0)
            throw std::runtime_error("numOfPipelineStreams not set. Should: cudaUtils.setNumOfPipelineStreams(n) in application.\n");
        cudaError_t err;
        err = cudaStreamCreateWithFlags(&paramsStream, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA error during paramsStream creation: " + std::string(cudaGetErrorString(err)));
        }
        err = cudaStreamCreateWithFlags(&workDataStream0, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA error during workDataStream0 creation: " + std::string(cudaGetErrorString(err)));
        }
        err = cudaStreamCreateWithFlags(&workDataStream1, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA error during workDataStream1 creation: " + std::string(cudaGetErrorString(err)));
        }
        CUDA_CHECK(cudaEventCreate(&workDataEvent0));
        CUDA_CHECK(cudaEventCreate(&workDataEvent1));

        pipelineStreams0 = (cudaStream_t*) malloc(numOfPipelineStreams * sizeof(cudaStream_t));
        pipelineStreams1 = (cudaStream_t*) malloc(numOfPipelineStreams * sizeof(cudaStream_t));
        events0 = (cudaEvent_t*) malloc(numOfPipelineStreams * sizeof(cudaEvent_t));
        events1 = (cudaEvent_t*) malloc(numOfPipelineStreams * sizeof(cudaEvent_t));
        for (int i = 0; i < numOfPipelineStreams; i++) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&pipelineStreams0[i], cudaStreamNonBlocking));
            CUDA_CHECK(cudaStreamCreateWithFlags(&pipelineStreams1[i], cudaStreamNonBlocking));
            CUDA_CHECK(cudaEventCreate(&events0[i]));
            CUDA_CHECK(cudaEventCreate(&events1[i]));
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

        CUDA_CHECK(cudaEventDestroy(workDataEvent0));
        CUDA_CHECK(cudaEventDestroy(workDataEvent1));
        for (int i = 0; i < numOfPipelineStreams; i++) {
            if (pipelineStreams0[i])
                CUDA_CHECK(cudaStreamDestroy(pipelineStreams0[i]));
            if (pipelineStreams1[i])
                CUDA_CHECK(cudaStreamDestroy(pipelineStreams1[i]));
            if (events0)
                CUDA_CHECK(cudaEventDestroy(events0[i]));
            if (events1)
                CUDA_CHECK(cudaEventDestroy(events1[i]));
        }
    }

    void initializeAMDBuffer(std::unique_ptr<AMDBuffers>& buffer, cudaStream_t stream, const std::string& bufferName) {
        if (!buffer) {
            buffer = std::make_unique<AMDBuffers>(ringDim, sizeP, sizeQ, param_sizeY, stream);
        } else {
            std::cerr << bufferName << " already initialized!" << std::endl;
        }
    }

    void initializeAMDBuffers() {
        initializeAMDBuffer(buffers0, workDataStream0, "AMDBuffers 0");
        initializeAMDBuffer(buffers1, workDataStream1, "AMDBuffers 1");
    }

    void destroyBuffers() {
        buffers0.reset();
        buffers1.reset();
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
