#ifndef CUDAPORTALFORAPPROXMODDOWNDATA_H
#define CUDAPORTALFORAPPROXMODDOWNDATA_H

#include <cstdint> // for uint32_t type

#include "lattice/poly.h"
#include <cuda_runtime.h>

#include "cuda-utils/m_vectors.h"
#include "cuda-utils/cuda-data-utils.h"
#include "cuda-utils/kernel-headers/approx-mod-down.cuh"
#include "cuda-utils/unmarshal_data_batch.h"

namespace lbcrypto {
    class Params;

    using PolyType = PolyImpl<NativeVector>;

class cudaPortalForApproxModDownData {

private:

    int id;

    cudaStream_t stream;
    cudaStream_t* pipelineStreams;
    cudaEvent_t  event;
    cudaEvent_t* pipelineEvents;

    uint32_t ringDim;
    uint32_t sizeQP;
    uint32_t sizeP;
    uint32_t sizeQ;

    // a crypto-parameter
    uint32_t    PHatModq_size_x;
    uint32_t    PHatModq_size_y;
    uint128_t*  host_PHatModq;
    uint128_t*  device_PHatModq;

    // cTilda
    uint32_t            cTilda_m_vectors_size_x;
    uint32_t            cTilda_m_vectors_size_y; // values size
    unsigned long*      host_cTilda_m_vectors; // flat
    unsigned long*      device_cTilda_m_vectors; // flat

    uint32_t            cTildaQ_m_vectors_size_x;
    uint32_t            cTildaQ_m_vectors_size_y;
    unsigned long*      host_cTildaQ_m_vectors;
    unsigned long*      device_cTildaQ_m_vectors;

    // partP_empty
    uint32_t            partP_empty_m_vectors_size_x;
    uint32_t            partP_empty_m_vectors_size_y; // values size
    unsigned long*      device_partP_empty_m_vectors; // flat

    // sum
    uint128_t*          device_sum;

    // partPSwitchedToQ
    uint32_t            partPSwitchedToQ_m_vectors_size_x;
    uint32_t            partPSwitchedToQ_m_vectors_size_y;
    unsigned long*      host_partPSwitchedToQ_m_vectors; //flat
    unsigned long*      device_partPSwitchedToQ_m_vectors; //flat
    //m_vectors_struct*   host_partPSwitchedToQ_m_vectors;
    //m_vectors_struct*   device_partPSwitchedToQ_m_vectors;
    //unsigned long**     device_partPSwitchedToQ_m_vectors_data_ptr;

    // ans
    uint32_t            ans_m_vectors_size_x;
    uint32_t            ans_m_vectors_size_y;
    unsigned long*      host_ans_m_vectors; //flat
    unsigned long*      device_ans_m_vectors; //flat

public:

    // Constructor
    cudaPortalForApproxModDownData(uint32_t ringDim, uint32_t sizeP, uint32_t sizeQ, const std::vector<std::vector<NativeInteger>>& PHatModq, cudaStream_t workDataStream, cudaStream_t* pipelineStreams, cudaEvent_t workDataEvent, cudaEvent_t* pipelineEvents, int id);

    // Destructor
    ~cudaPortalForApproxModDownData();

    // Setter Functions
    void                                                set_SizeQP(const uint32_t size) { sizeQP = size; }

    // Getter Functions
    cudaStream_t                                        getStream() const { return stream; }
    cudaStream_t                                        getPipelineStream(const uint32_t i) const { return pipelineStreams[i]; }
    cudaEvent_t                                         getEvent() const { return event;}
    cudaEvent_t                                         getPipelineEvent(const uint32_t i) const { return pipelineEvents[i]; }

    uint32_t                                            get_partP_empty_size_x() const { return partP_empty_m_vectors_size_x; }
    uint32_t                                            get_partP_empty_size_y() const { return partP_empty_m_vectors_size_y; }
    ulong*                                              getDevice_partP_empty_m_vectors() const { return device_partP_empty_m_vectors; }

    uint32_t                                            get_partPSwitchedToQ_size_x() const { return partPSwitchedToQ_m_vectors_size_x; }
    uint32_t                                            get_partPSwitchedToQ_size_y() const { return partPSwitchedToQ_m_vectors_size_y; }
    ulong*                                              getHost_partPSwitchedToQ_m_vectors() const { return host_partPSwitchedToQ_m_vectors; }
    uint128_t*                                          getDevice_sum() const { return device_sum; }
    ulong*                                              getDevice_partPSwitchedToQ_m_vectors() const { return device_partPSwitchedToQ_m_vectors; }

    // Data Marshalling Functions
    void marshalCTildaBatch(const std::vector<PolyImpl<NativeVector>>& cTilda_m_vectors, uint32_t partP_index, uint32_t cTilda_index) const;
    void marshalCTildaQBatch(const std::vector<PolyImpl<NativeVector>>& cTilda_m_vectors, uint32_t index) const;
    void marshalPHatModqBatch(const std::vector<std::vector<NativeInteger>>& PHatModq, uint32_t index) const;
    void unmarshalWorkDataBatchWrapper(std::vector<PolyImpl<NativeVector>>& ans_m_vectors, uint32_t i, uint32_t ptr_offset, cudaStream_t pipelineStream) const;

    // Data Transfer Functions
    void copyInCTildaQ_Batch(uint32_t ptrOffset, cudaStream_t stream) const;
    void copyInPartP_Batch(uint32_t ptrOffset, cudaStream_t stream) const;
    void copyInPHatModqBatch(uint32_t index, cudaStream_t stream) const;

    void copyOutResultBatch(uint32_t ptrOffset, cudaStream_t stream) const;

    // Kernel Invocation Functions
    void invokeKernelOfApproxModDownBatchPt1(int gpuBlocks, int gpuThreads, ulong modulus, ulong tInvModp, ulong tInvModpPrecon, uint32_t ptr_offset, cudaStream_t stream);
    void invokeKernelOfApproxSwitchCRTBasisPt1Batch(int gpuBlocks, int gpuThreads,
                                                    uint32_t i, ulong modulus, uint32_t ptr_offset,
                                                    ulong QHatInvModq, ulong QHatInvModqPrecon,
                                                    cudaStream_t stream);
    void invokeKernelOfApproxSwitchCRTBasisPt2Batch(int gpuBlocks, int gpuThreads,
                                                    uint32_t i, ulong ans_modulus, uint32_t ptr_offset,
                                                    uint128_t modpBarrettMu, uint32_t t, ulong tModqPrecon,
                                                    cudaStream_t stream);
    void invokeAnsFillBatchKernel(int gpuBlocks, int gpuThreads,
                                  uint32_t i, ulong modulus, uint32_t ptr_offset,
                                  ulong pInvModq, ulong pInvModqPrecon,
                                  cudaStream_t stream) const;

    // Resources Allocation/Deallocation - Error Handling - Misc Functions

    void print_host_m_vectors();

private:
    void allocateHostData();
    static void handleFreeError(const std::string& operation, void* ptr);
    static void handleCUDAError(const std::string& operation, cudaError_t err);
    void freeHostMemory();
    void freeDeviceMemory();

    template<typename T>
    void safeFree(T*& ptr) {
        if (ptr != nullptr) {
            free(ptr);
            ptr = nullptr; // Set to nullptr to avoid dangling pointer
        }
    }

    template<typename T>
    void safeCudaFreeHost(T*& ptr) {
        if (ptr != nullptr) {
            cudaFreeHost(ptr);
            ptr = nullptr; // Set to nullptr to avoid dangling pointer
        }
    }

    template<typename T>
    void safeCudaFreeAsync(T*& ptr, cudaStream_t stream) {
        if (ptr != nullptr) {
            CUDA_CHECK(cudaFreeAsync(ptr, stream));  // Custom macro to check CUDA calls
            ptr = nullptr; // Set to nullptr to prevent dangling pointers
        }
    }

    int whoAmI() {
        return id;
    }

};

}

#endif //CUDAPORTALFORAPPROXMODDOWNDATA_H
