#include "cuda-utils/cudaPortalForApproxSwitchCRTBasis.h"

namespace lbcrypto {

using PolyType = PolyImpl<NativeVector>;

//constructor impl
cudaPortalForApproxSwitchCRTBasis::cudaPortalForApproxSwitchCRTBasis(const std::shared_ptr<cudaPortalForParamsData> params_data, cudaStream_t workDataStream) {
    //std::cout << "[CONSTRUCTOR] Call constructor for " << this << "(cudaPortalForApproxModDown)" << std::endl;

    this->paramsData = params_data;

    this->ringDim = params_data->getRingDim();
    this->sizeP = params_data->getSizeP();
    this->sizeQ = params_data->getSizeQ();

    this->stream = workDataStream;

    allocateHostData();
}

cudaPortalForApproxSwitchCRTBasis::~cudaPortalForApproxSwitchCRTBasis() {
    //std::cout << "[DESTRUCTOR] Call destructor for " << this << "(cudaPortalForApproxModDown)" << std::endl;

    freeHostMemory();
    freeDeviceMemory();
}


// Getter Functions
cudaStream_t                                cudaPortalForApproxSwitchCRTBasis::getStream() const { return stream; }
std::shared_ptr<cudaPortalForParamsData>    cudaPortalForApproxSwitchCRTBasis::getParamsData() const { return paramsData; }
m_vectors_struct*                           cudaPortalForApproxSwitchCRTBasis::getHost_ans_m_vectors() const {return host_ans_m_vectors;}
//uint128_t*                                  cudaPortalForApproxSwitchCRTBasis::getDevice_sum() const { return device_sum;}
m_vectors_struct*                           cudaPortalForApproxSwitchCRTBasis::getDevice_m_vectors() const { return device_m_vectors;}
m_vectors_struct*                           cudaPortalForApproxSwitchCRTBasis::getDevice_ans_m_vectors() const { return device_ans_m_vectors;}

// marshal

void cudaPortalForApproxSwitchCRTBasis::marshalWorkData(const std::vector<PolyImpl<NativeVector>>& m_vectors,
                                                 const std::vector<PolyImpl<NativeVector>>& ans_m_vectors) {
    for (uint32_t q = 0; q < sizeQ; ++q) {
        for (uint32_t rd = 0; rd < ringDim; ++rd) {
            host_m_vectors[q].data[rd] = m_vectors[q][rd].template ConvertToInt<>();
        }
        host_m_vectors[q].modulus = m_vectors[q].GetModulus().ConvertToInt();
    }
    for (uint32_t sp = 0; sp < sizeP; sp++) {
        host_ans_m_vectors[sp].modulus = ans_m_vectors[sp].GetModulus().ConvertToInt();
    }

    // print for debugging
    /*std::cout << "[DEBUG Print]: " << std::endl;
    for (uint32_t q = 0; q < sizeQ; ++q) {
        for (uint32_t rd = 0; rd < 3; ++rd) {
            std::cout << "host_m_vectors[" << q << "].data[" << rd << "] = " << host_m_vectors[q].data[rd] <<
                         ", m_vectors["      << q << "]["      << rd << "] = " << m_vectors[q][rd].template ConvertToInt<>() << std::endl;
        }
    }*/
}

void cudaPortalForApproxSwitchCRTBasis::unmarshalWorkData(std::vector<PolyImpl<NativeVector>>& ans_m_vectors) {
    // make sure stream has finished all queued tasks before touching the results from host
    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        std::cerr << "Stream synchronization failed: " << cudaGetErrorString(err) << std::endl;
    }
    //std::cout << "==> UNMARSHAL START" << std::endl;

    for (usint j = 0; j < sizeP; j++) {
        for(usint ri = 0; ri < ringDim; ri++) {
            ans_m_vectors[j][ri] = NativeInteger(host_ans_m_vectors[j].data[ri]);
        }
    }
}

// Data Transfer Functions

void cudaPortalForApproxSwitchCRTBasis::copyInWorkData() {

    cudaError_t err = cudaMallocAsync((void**)&device_m_vectors, sizeQ * sizeof(m_vectors_struct), stream);
    if (err != cudaSuccess) {
        printf("Error allocating device_m_vectors: %s (%d)\n", cudaGetErrorString(err), err);
        exit(-1); // or handle error appropriately
    }

    err = cudaMemcpyAsync(device_m_vectors, host_m_vectors, sizeQ * sizeof(m_vectors_struct), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        printf("Error copying to device_m_vectors: %s (%d)\n", cudaGetErrorString(err), err);
        return; // or handle error appropriately
    }

    cudaHostAlloc((void**)&this->device_m_vectors_data_ptr, sizeQ * sizeof(unsigned long*), cudaHostAllocDefault);
    if (!device_m_vectors_data_ptr) {
        printf("Error allocating device_m_vectors_data_ptr\n");
        return; // or handle error appropriately
    }

    for (uint32_t q = 0; q < sizeQ; ++q) {
        err = cudaMallocAsync((void**)&(device_m_vectors_data_ptr[q]), ringDim * sizeof(unsigned long), stream);
        if (err != cudaSuccess) {
            printf("Error allocating device_m_vectors_data_ptr[%d]: %s (%d)\n", q, cudaGetErrorString(err), err);
            return; // or handle error appropriately
        }

        err = cudaMemcpyAsync(&(device_m_vectors[q].data), &(device_m_vectors_data_ptr[q]), sizeof(unsigned long*), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            printf("Error copying to device_m_vectors[%d].data: %s (%d)\n", q, cudaGetErrorString(err), err);
            return; // or handle error appropriately
        }

        err = cudaMemcpyAsync(device_m_vectors_data_ptr[q], host_m_vectors[q].data, ringDim * sizeof(unsigned long), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            printf("Error copying to device_m_vectors_data_ptr[%d]: %s (%d)\n", q, cudaGetErrorString(err), err);
            return; // or handle error appropriately
        }
    }

    // sum
    //err = cudaMallocAsync((void**)&device_sum, sizeP * ringDim * sizeof(uint128_t), stream);
    //if (err != cudaSuccess) {
        //printf("Error allocating device_sum: %s (%d)\n", cudaGetErrorString(err), err);
        //return; // or handle error appropriately
    //}

    //err = cudaMemsetAsync(device_sum, 0, sizeP * ringDim * sizeof(uint128_t), stream);
    //if (err != cudaSuccess) {
        //printf("Error setting device_sum to zero: %s (%d)\n", cudaGetErrorString(err), err);
        //return; // or handle error appropriately
    //}

    // ans_m_vectors
    err = cudaMallocAsync((void**)&device_ans_m_vectors, sizeP * sizeof(m_vectors_struct), stream);
    if (err != cudaSuccess) {
        printf("Error allocating device_ans_m_vectors: %s (%d)\n", cudaGetErrorString(err), err);
        return; // or handle error appropriately
    }

    err = cudaMemcpyAsync(device_ans_m_vectors, host_ans_m_vectors, sizeP * sizeof(m_vectors_struct), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        printf("Error copying to device_ans_m_vectors: %s (%d)\n", cudaGetErrorString(err), err);
        return; // or handle error appropriately
    }

    cudaHostAlloc((void**)&this->device_ans_m_vectors_data_ptr, sizeP * sizeof(unsigned long*), cudaHostAllocDefault);

    for (uint32_t p = 0; p < sizeP; ++p) {
        err = cudaMallocAsync((void**)&(device_ans_m_vectors_data_ptr[p]), ringDim * sizeof(unsigned long), stream);
        if (err != cudaSuccess) {
            printf("Error allocating device_ans_m_vectors_data_ptr[%d]: %s (%d)\n", p, cudaGetErrorString(err), err);
            return; // or handle error appropriately
        }

        err = cudaMemcpyAsync(&(device_ans_m_vectors[p].data), &(device_ans_m_vectors_data_ptr[p]), sizeof(unsigned long*), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            printf("Error copying to device_ans_m_vectors[%d].data: %s (%d)\n", p, cudaGetErrorString(err), err);
            return; // or handle error appropriately
        }
    }
}

void cudaPortalForApproxSwitchCRTBasis::copyOutResult() {

    for(uint32_t p = 0; p < sizeP; p++) {
        cudaMemcpyAsync(host_ans_m_vectors[p].data, device_ans_m_vectors_data_ptr[p], ringDim * sizeof(unsigned long), cudaMemcpyDeviceToHost, stream);
    }

    //std::cout << "==> COPY OUT FINISHED" << std::endl;
}

// Kernel Invocation Function

void cudaPortalForApproxSwitchCRTBasis::invokeKernelOfApproxSwitchCRTBasisV2(int gpuBlocks, int gpuThreads) {
    dim3 blocks = dim3(gpuBlocks, 1U, 1U); // Set the grid dimensions
    dim3 threads = dim3(gpuThreads, 1U, 1U); // Set the block dimensions

    ulong*              device_QHatInvModq          = paramsData->getDevice_QHatInvModq();
    ulong*              device_QHatInvModqPrecon    = paramsData->getDevice_QHatInvModqPrecon();
    uint128_t*          device_QHatModp             = paramsData->getDevice_QHatModp();
    uint128_t*          device_modpBarrettMu        = paramsData->getDevice_modpBarrettMu();

    size_t sharedMemSize = gpuThreads * sizeof(ulong);
    //printf("sharedMem1024=%lu\n", 1024 * sizeP * sizeof(uint128_t));
    //printf("sharedMem512=%lu\n", 512 * sizeP * sizeof(uint128_t));
    //printf("sharedMem256=%lu\n", 256 * sizeP * sizeof(uint128_t));
    //size_t sharedMemSize = gpuThreads * sizeof(ulong);
    //size_t sharedMemSize2 = gpuThreads * sizeof(ulong);

    // Loop over `sizeP` and invoke kernel for each `pIndex`
    for (uint32_t pIndex = 0; pIndex < sizeP; ++pIndex) {
        void* args[] = {
            &ringDim, &sizeP, &sizeQ, &pIndex,
            &device_m_vectors, &device_QHatInvModq, &device_QHatInvModqPrecon,
            &device_QHatModp, &device_modpBarrettMu, &device_ans_m_vectors
        };

        approxSwitchCRTBasisKernelWrapperV2(blocks, threads, args, sharedMemSize, stream);
        //approxSwitchCRTBasisKernelWrapperV2(blocks, threads, args, 0U, stream);

        // Optional: Check for errors after each kernel launch
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel launch failed for pIndex %d: %s\n", pIndex, cudaGetErrorString(err));
            return;
        }
    }
}

void cudaPortalForApproxSwitchCRTBasis::invokeKernelOfApproxSwitchCRTBasis(int gpuBlocks, int gpuThreads) {

    dim3 blocks = dim3(gpuBlocks, 1U, 1U); // Set the grid dimensions
    dim3 threads = dim3(gpuThreads, 1U, 1U); // Set the block dimensions

    ulong*              device_QHatInvModq          = paramsData->getDevice_QHatInvModq();
    ulong*              device_QHatInvModqPrecon    = paramsData->getDevice_QHatInvModqPrecon();
    uint128_t*          device_QHatModp             = paramsData->getDevice_QHatModp();
    uint128_t*          device_modpBarrettMu        = paramsData->getDevice_modpBarrettMu();

    void *args[] = {&ringDim, &sizeP, &sizeQ, &device_m_vectors, &device_QHatInvModq, &device_QHatInvModqPrecon, &device_QHatModp, /*&device_sum,*/ &device_modpBarrettMu, &device_ans_m_vectors};

    size_t sharedMemSize = gpuThreads * sizeP * sizeof(uint128_t);

    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // Log shared memory information
    printf("Max shared memory per block: %ld bytes\n", prop.sharedMemPerBlock);
    printf("Requested shared memory: %ld bytes\n", sharedMemSize);
    printf("Requested shared memory (without sizeP): %ld bytes\n", gpuThreads * sizeof(uint128_t));


    approxSwitchCRTBasisKernelWrapper(blocks, threads, args, sharedMemSize, stream);
}

// Resources Allocation/Deallocation - Error Handling - Misc Functions

void cudaPortalForApproxSwitchCRTBasis::allocateHostData() {
    cudaHostAlloc((void**)&host_m_vectors, sizeQ * sizeof(m_vectors_struct), cudaHostAllocDefault);
    for (uint32_t q = 0; q < sizeQ; ++q) {
        cudaHostAlloc((void**)&host_m_vectors[q].data, ringDim * sizeof(unsigned long), cudaHostAllocDefault);
    }

    cudaHostAlloc((void**)&host_ans_m_vectors, sizeP * sizeof(m_vectors_struct), cudaHostAllocDefault);
    for (uint32_t p = 0; p < sizeP; ++p) {
        cudaHostAlloc((void**)&host_ans_m_vectors[p].data, ringDim * sizeof(unsigned long), cudaHostAllocDefault);
    }
}

void cudaPortalForApproxSwitchCRTBasis::handleFreeError(const std::string& operation, void* ptr) {
    if (ptr == nullptr) {
        throw std::runtime_error("Memory error during " + operation + ": null pointer passed for freeing.");
    } else {
        free(ptr);  // Actual free operation happens here
        ptr = nullptr;  // Reset to nullptr after free
    }
}


void cudaPortalForApproxSwitchCRTBasis::handleCUDAError(const std::string& operation, cudaError_t err) {
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error during " + operation + ": " + std::string(cudaGetErrorString(err)));
    }
}

void cudaPortalForApproxSwitchCRTBasis::freeHostMemory() {
    // Free host_m_vectors[q].data memory
    for (uint32_t q = 0; q < sizeQ; ++q) {
        cudaFreeHost(host_m_vectors[q].data);
    }

    // Free host_m_vectors structure
    cudaFreeHost(host_m_vectors);

    // Free host_ans_m_vectors[p].data memory
    for (uint32_t p = 0; p < sizeP; ++p) {
        cudaFreeHost(host_ans_m_vectors[p].data);
    }

    // Free host_ans_m_vectors structure
    cudaFreeHost(host_ans_m_vectors);
}

void cudaPortalForApproxSwitchCRTBasis::freeDeviceMemory() {
    cudaError_t err;

    // Free device_m_vectors_data_ptr memory
    for (uint32_t q = 0; q < sizeQ; ++q) {
        if (device_m_vectors_data_ptr[q]) {
            err = cudaFreeAsync(device_m_vectors_data_ptr[q], stream);
            handleCUDAError("freeing device_m_vectors_data_ptr[" + std::to_string(q) + "]", err);
        }
    }
    cudaFreeHost(device_m_vectors_data_ptr);

    // Free device_m_vectors memory
    if (device_m_vectors) {
        err = cudaFreeAsync(device_m_vectors, stream);
        handleCUDAError("freeing device_m_vectors", err);
        device_m_vectors = nullptr;
    }

    // Free device_sum memory
    //if (device_sum) {
        //err = cudaFreeAsync(device_sum, stream);
        //handleCUDAError("freeing device_sum", err);
        //device_sum = nullptr;
    //}

    // Free device_ans_m_vectors_data_ptr memory
    for (uint32_t p = 0; p < sizeP; ++p) {
        if (device_ans_m_vectors_data_ptr[p]) {
            err = cudaFreeAsync(device_ans_m_vectors_data_ptr[p], stream);
            handleCUDAError("freeing device_ans_m_vectors_data_ptr[" + std::to_string(p) + "]", err);
        }
    }
    cudaFreeHost(device_ans_m_vectors_data_ptr);

    // Free device_ans_m_vectors memory
    if (device_ans_m_vectors) {
        err = cudaFreeAsync(device_ans_m_vectors, stream);
        handleCUDAError("freeing device_ans_m_vectors", err);
        device_ans_m_vectors = nullptr;
    }
}

void cudaPortalForApproxSwitchCRTBasis::print_host_m_vectors() {

    std::cout << "cudaPortalForApproxModDown::print_host_m_vectors" << std::endl;

    for (uint32_t q = 0; q < sizeQ; ++q) {
        std::cout << "host_m_vectors[" << q << "].data[0-3/ringDim]: ";
        for (uint32_t rd = 0; rd < 3; ++rd) {
            std::cout << host_m_vectors[q].data[rd] << " ";
        }
        std::cout << std::endl;
    }
}

}
