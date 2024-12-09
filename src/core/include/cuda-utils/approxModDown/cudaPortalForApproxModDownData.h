#ifndef CUDAPORTALFORAPPROXMODDOWNDATA_H
#define CUDAPORTALFORAPPROXMODDOWNDATA_H

#include <cstdint> // for uint32_t type

#include "lattice/poly.h"
#include <cuda_runtime.h>

#include "cuda-utils/m_vectors.h"
#include "cudaPortalForApproxModDownParams.h"
//#include "cuda-utils/kernel-headers/shared_device_functions.cuh"
#include "cuda-utils/kernel-headers/approx-mod-down.cuh"
#include "cuda-utils/unmarshal_data_batch.h"

namespace lbcrypto {
    class Params;

    using PolyType = PolyImpl<NativeVector>;

class cudaPortalForApproxModDownData {

private:

    int id;

    std::shared_ptr<cudaPortalForApproxModDownParams> paramsData;

    cudaStream_t stream;
    cudaStream_t* pipelineStreams;
    cudaEvent_t  event;
    cudaEvent_t* pipelineEvents;

    uint32_t ringDim;
    uint32_t sizeQP;
    uint32_t sizeP;
    uint32_t sizeQ;

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
    cudaPortalForApproxModDownData(std::shared_ptr<cudaPortalForApproxModDownParams> params_data, cudaStream_t workDataStream, cudaStream_t* pipelineStreams, cudaEvent_t workDataEvent, cudaEvent_t* pipelineEvents, int id);

    // Destructor
    ~cudaPortalForApproxModDownData();

    // Setter Functions
    void                                                set_SizeQP(uint32_t size) { sizeQP = size; }

    // Getter Functions
    cudaStream_t                                        getStream() const { return stream; }
    cudaStream_t                                        getPipelineStream(int i) const { return pipelineStreams[i]; }
    cudaEvent_t                                         getEvent() const { return event;}
    cudaEvent_t                                         getPipelineEvent(int i) const { return pipelineEvents[i]; }
    std::shared_ptr<cudaPortalForApproxModDownParams>   getParamsData() const { return paramsData; }

    uint32_t                                            get_partP_empty_size_x() const { return partP_empty_m_vectors_size_x; }
    uint32_t                                            get_partP_empty_size_y() const { return partP_empty_m_vectors_size_y; }
    ulong*                                              getDevice_partP_empty_m_vectors() const { return device_partP_empty_m_vectors; }

    uint32_t                                            get_partPSwitchedToQ_size_x() const { return partPSwitchedToQ_m_vectors_size_x; }
    uint32_t                                            get_partPSwitchedToQ_size_y() const { return partPSwitchedToQ_m_vectors_size_y; }
    ulong*                                              getHost_partPSwitchedToQ_m_vectors() const { return host_partPSwitchedToQ_m_vectors; }
    uint128_t*                                          getDevice_sum() const { return device_sum; }
    ulong*                                              getDevice_partPSwitchedToQ_m_vectors() const { return device_partPSwitchedToQ_m_vectors; }

    // Host allocations
    void allocateHostCTilda(uint32_t cTilda_size_x, uint32_t cTilda_size_y);

    // Data Marshalling Functions
    void marshalCTilda(const std::vector<PolyImpl<NativeVector>>& cTilda_m_vectors);

    void marshalCTildaBatch(const std::vector<PolyImpl<NativeVector>>& cTilda_m_vectors, uint32_t partP_index, uint32_t cTilda_index);
    void marshalCTildaQBatch(const std::vector<PolyImpl<NativeVector>>& cTilda_m_vectors, uint32_t index);

    void marshalWorkData(const std::vector<PolyImpl<NativeVector>>& partPSwitchedToQ_m_vectors);

    void unmarshalWorkData(std::vector<PolyImpl<NativeVector>>& ans_m_vectors);
    void unmarshalWorkDataBatchWrapper(std::vector<PolyImpl<NativeVector>>& ans_m_vectors, uint32_t i, uint32_t ptr_offset, cudaStream_t pipelineStream);

    // Data Transfer Functions
    void copyInCTilda();
    void copyInCTildaQ_Batch(uint32_t ptrOffset, cudaStream_t stream);
    void copyInPartP_Empty();
    void copyInPartP_Batch(uint32_t ptrOffset, cudaStream_t stream);
    void copyInWorkData();

    void copyOutResult();
    void copyOutResultBatch(uint32_t ptrOffset, cudaStream_t stream);

    // Kernel Invocation Function
    void invokePartPFillKernel(int gpuBlocks, int gpuThreads);
    void invokeKernelOfApproxModDown(int gpuBlocks, int gpuThreads);
    void invokeKernelOfApproxModDownBatchPt1(int gpuBlocks, int gpuThreads, ulong modulus, ulong tInvModp, ulong tInvModpPrecon, uint32_t ptr_offset, cudaStream_t stream);
    void invokeKernelOfApproxSwitchCRTBasisPt1Batch(int gpuBlocks, int gpuThreads,
                                                    uint32_t i, ulong modulus, uint32_t ptr_offset,
                                                    ulong QHatInvModq, ulong QHatInvModqPrecon,
                                                    //ulong ans_modulus, uint128_t modpBarrettMu,
                                                    cudaStream_t stream);
    void invokeKernelOfApproxSwitchCRTBasisPt2Batch(int gpuBlocks, int gpuThreads,
                                                    uint32_t i, ulong ans_modulus, uint32_t ptr_offset,
                                                    uint128_t modpBarrettMu, uint32_t t, ulong tModqPrecon,
                                                    cudaStream_t stream);
    void invokeAnsFillKernel(int gpuBlocks, int gpuThreads);
    void invokeAnsFillBatchKernel(int gpuBlocks, int gpuThreads,
                                  uint32_t i, ulong modulus, uint32_t ptr_offset,
                                  ulong pInvModq, ulong pInvModqPrecon,
                                  cudaStream_t stream);

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
