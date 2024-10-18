#ifndef CUDAPORTALFORAPPROXMODDOWNDATA_H
#define CUDAPORTALFORAPPROXMODDOWNDATA_H

#include <cstdint> // for uint32_t type

#include "lattice/poly.h"
#include <cuda_runtime.h>

#include "cuda-utils/m_vectors.h"
#include "cudaPortalForApproxModDownParams.h"
//#include "cuda-utils/kernel-headers/shared_device_functions.cuh"
#include "cuda-utils/kernel-headers/approx-mod-down.cuh"

namespace lbcrypto {
    class Params;

    using PolyType = PolyImpl<NativeVector>;

class cudaPortalForApproxModDownData {

private:

    int id;

    std::shared_ptr<cudaPortalForApproxModDownParams> paramsData;

    cudaStream_t stream;

    uint32_t ringDim;
    uint32_t sizeP;
    uint32_t sizeQ;

    // partP
    uint32_t            partP_m_vectors_size_x;
    uint32_t            partP_m_vectors_size_y; // values size
    unsigned long*      host_partP_m_vectors; // flat
    unsigned long*      device_partP_m_vectors; // flat
    //m_vectors_struct*   host_partP_m_vectors;
    //m_vectors_struct*   device_partP_m_vectors;
    //unsigned long**     device_partP_m_vectors_data_ptr;

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

public:

    // Constructor
    cudaPortalForApproxModDownData(std::shared_ptr<cudaPortalForApproxModDownParams> params_data, cudaStream_t workDataStream, int id);

    // Destructor
    ~cudaPortalForApproxModDownData();

    // Getter Functions
    cudaStream_t                                        getStream() const { return stream; }
    std::shared_ptr<cudaPortalForApproxModDownParams>   getParamsData() const { return paramsData; }
    uint32_t                                            get_partP_size_x() const { return partP_m_vectors_size_x; }
    uint32_t                                            get_partP_size_y() const { return partP_m_vectors_size_y; }
    ulong*                                   getHost_partP_m_vectors() const { return host_partP_m_vectors; }
    ulong*                                   getHost_partPSwitchedToQ_m_vectors() const { return host_partPSwitchedToQ_m_vectors; }
    uint128_t*                                          getDevice_sum() const { return device_sum; }
    ulong*                                   getDevice_partP_m_vectors() const { return device_partP_m_vectors; }
    ulong*                                   getDevice_partPSwitchedToQ_m_vectors() const { return device_partPSwitchedToQ_m_vectors; }

    void allocAndMarshalHostWorkData(const std::vector<PolyImpl<NativeVector>>& partP_m_vectors,
                                     const std::vector<PolyImpl<NativeVector>>& partPSwitchedToQ_m_vectors);

    // Data Marshalling Functions
    void marshalWorkData(const std::vector<PolyImpl<NativeVector>>& partP_vectors,
                         const std::vector<PolyImpl<NativeVector>>& partPSwitchedToQ_m_vectors);

    void unmarshalWorkData(std::vector<PolyImpl<NativeVector>>& partPSwitchedToQ_m_vectors);

    // Data Transfer Functions
    void copyInWorkData();

    void copyOutResult();

    // Kernel Invocation Function
    void invokeKernelOfApproxModDown(int gpuBlocks, int gpuThreads);

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
