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
unsigned long*                              cudaPortalForApproxSwitchCRTBasis::getHost_ans_m_vectors_data() const {return host_ans_m_vectors_data;}
unsigned long*                              cudaPortalForApproxSwitchCRTBasis::getHost_ans_m_vectors_modulus() const {return host_ans_m_vectors_modulus;}
uint128_t*                                  cudaPortalForApproxSwitchCRTBasis::getDevice_sum() const { return device_sum;}
unsigned long*                              cudaPortalForApproxSwitchCRTBasis::getDevice_m_vectors_data() const { return device_m_vectors_data;}
unsigned long*                              cudaPortalForApproxSwitchCRTBasis::getDevice_m_vectors_modulus() const { return device_m_vectors_modulus;}
unsigned long*                              cudaPortalForApproxSwitchCRTBasis::getDevice_ans_m_vectors_data() const { return device_ans_m_vectors_data;}
unsigned long*                              cudaPortalForApproxSwitchCRTBasis::getDevice_ans_m_vectors_modulus() const { return device_ans_m_vectors_modulus;}

// marshal

void cudaPortalForApproxSwitchCRTBasis::marshalWorkData(const std::vector<PolyImpl<NativeVector>>& m_vectors,
                                                 const std::vector<PolyImpl<NativeVector>>& ans_m_vectors) {
    for (uint32_t q = 0; q < sizeQ; ++q) {
        for (uint32_t rd = 0; rd < ringDim; ++rd) {
            host_m_vectors_data[(q * ringDim) + rd] = m_vectors[q][rd].template ConvertToInt<>();
        }
        host_m_vectors_modulus[q] = m_vectors[q].GetModulus().ConvertToInt();
    }
    for (uint32_t sp = 0; sp < sizeP; sp++) {
        host_ans_m_vectors_modulus[sp] = ans_m_vectors[sp].GetModulus().ConvertToInt();
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
            ans_m_vectors[j][ri] = NativeInteger(host_ans_m_vectors_data[(j * ringDim) + ri]);
        }
    }
}

// Data Transfer Functions

void cudaPortalForApproxSwitchCRTBasis::copyInWorkData() {
    cudaError_t err;

    size_t m_vectors_data_size = sizeQ * ringDim * sizeof(unsigned long);
    size_t m_vectors_modulus_size = sizeQ * sizeof(unsigned long);

    err = cudaMalloc((void**)&device_m_vectors_data, m_vectors_data_size);
    if (err != cudaSuccess) {
        printf("Error allocating device_m_vectors_data: %s (%d)\n", cudaGetErrorString(err), err);
        throw std::runtime_error("");
    }
    err = cudaMalloc((void**)&device_m_vectors_modulus, m_vectors_modulus_size);
    if (err != cudaSuccess) {
        printf("Error allocating device_m_vectors_modulus: %s (%d)\n", cudaGetErrorString(err), err);
        throw std::runtime_error("");
    }

    err = cudaMemcpyAsync(device_m_vectors_data, host_m_vectors_data, m_vectors_data_size, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        printf("Error copying to device_m_vectors_data: %s (%d)\n", cudaGetErrorString(err), err);
        throw std::runtime_error("");
    }
    err = cudaMemcpyAsync(device_m_vectors_modulus, host_m_vectors_modulus, m_vectors_modulus_size, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        printf("Error copying to device_m_vectors_modulus: %s (%d)\n", cudaGetErrorString(err), err);
        throw std::runtime_error("");
    }

    // sum
    err = cudaMalloc((void**)&device_sum, sizeP * ringDim * sizeof(uint128_t));
    if (err != cudaSuccess) {
        printf("Error allocating device_sum: %s (%d)\n", cudaGetErrorString(err), err);
        throw std::runtime_error("");
    }

    err = cudaMemset(device_sum, 0, sizeP * ringDim * sizeof(uint128_t));
    if (err != cudaSuccess) {
        printf("Error setting device_sum to zero: %s (%d)\n", cudaGetErrorString(err), err);
        throw std::runtime_error("");
    }

    // ans
    size_t ans_m_vectors_data_size = sizeP * ringDim * sizeof(unsigned long);
    size_t ans_m_vectors_modulus_size = sizeP * sizeof(unsigned long);
    err = cudaMalloc((void**)&device_ans_m_vectors_data, ans_m_vectors_data_size);
    if (err != cudaSuccess) {
        printf("Error allocating device_ans_m_vectors_data: %s (%d)\n", cudaGetErrorString(err), err);
        throw std::runtime_error("");
    }
    err = cudaMalloc((void**)&device_ans_m_vectors_modulus, ans_m_vectors_modulus_size);
    if (err != cudaSuccess) {
        printf("Error allocating device_ans_m_vectors_modulus: %s (%d)\n", cudaGetErrorString(err), err);
        throw std::runtime_error("");
    }

    err = cudaMemcpyAsync(device_ans_m_vectors_data, host_ans_m_vectors_data, ans_m_vectors_data_size, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        printf("Error copying to device_ans_m_vectors_data: %s (%d)\n", cudaGetErrorString(err), err);
        throw std::runtime_error("");
    }
    err = cudaMemcpyAsync(device_ans_m_vectors_modulus, host_ans_m_vectors_modulus, ans_m_vectors_modulus_size, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        printf("Error copying to device_ans_m_vectors_modulus: %s (%d)\n", cudaGetErrorString(err), err);
        throw std::runtime_error("");
    }
}

void cudaPortalForApproxSwitchCRTBasis::copyOutResult() {
    cudaError_t prevErr = cudaGetLastError();
    if (prevErr != cudaSuccess) {
        printf("Previous CUDA error: %s (%d)\n", cudaGetErrorString(prevErr), prevErr);
        throw std::runtime_error("");
    }
    if (host_ans_m_vectors_data == nullptr || device_ans_m_vectors_data == nullptr) {
        printf("Memory for host/device answer vectors not allocated.\n");
        throw std::runtime_error("");
    }
    cudaError_t err = cudaMemcpyAsync(host_ans_m_vectors_data, device_ans_m_vectors_data, sizeP * ringDim * sizeof(unsigned long), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        printf("Error copying from device_ans_m_vectors_data: %s (%d)\n", cudaGetErrorString(err), err);
        throw std::runtime_error("");
    }

    //std::cout << "==> COPY OUT FINISHED" << std::endl;
}

// Kernel Invocation Function

void cudaPortalForApproxSwitchCRTBasis::invokeKernelOfApproxSwitchCRTBasis(int gpuBlocks, int gpuThreads) {

    dim3 blocks = dim3(gpuBlocks, 1U, 1U); // Set the grid dimensions
    dim3 threads = dim3(gpuThreads, 1U, 1U); // Set the block dimensions

    ulong*              device_QHatInvModq          = paramsData->getDevice_QHatInvModq();
    ulong*              device_QHatInvModqPrecon    = paramsData->getDevice_QHatInvModqPrecon();
    uint128_t*          device_QHatModp             = paramsData->getDevice_QHatModp();
    uint128_t*          device_modpBarrettMu        = paramsData->getDevice_modpBarrettMu();

    void *args[] = {&ringDim, &sizeP, &sizeQ, &device_m_vectors_data, &device_m_vectors_modulus, &device_QHatInvModq, &device_QHatInvModqPrecon, &device_QHatModp, &device_sum, &device_modpBarrettMu, &device_ans_m_vectors_data, &device_ans_m_vectors_modulus};

    approxSwitchCRTBasisKernelWrapper(blocks, threads, args, stream);
}

// Resources Allocation/Deallocation - Error Handling - Misc Functions

void cudaPortalForApproxSwitchCRTBasis::allocateHostData() {
    size_t host_m_vectors_data_size = sizeQ * ringDim * sizeof(unsigned long);
    size_t host_m_vectors_modulus_size = sizeQ * sizeof(unsigned long);
    size_t host_ans_m_vectors_data_size = sizeP * ringDim * sizeof(unsigned long);
    size_t host_ans_m_vectors_modulus_size = sizeP * sizeof(unsigned long);

    host_m_vectors_data = (unsigned long*) malloc(host_m_vectors_data_size);
    if (host_m_vectors_data == nullptr) {
        fprintf(stderr, "Error: Failed to allocate memory for host_m_vectors_data\n");
        throw std::runtime_error("Memory allocation failed");
    }

    host_m_vectors_modulus = (unsigned long*) malloc(host_m_vectors_modulus_size);
    if (host_m_vectors_modulus == nullptr) {
        fprintf(stderr, "Error: Failed to allocate memory for host_m_vectors_modulus\n");
        free(host_m_vectors_data);  // Free previously allocated memory to avoid leaks
        throw std::runtime_error("Memory allocation failed");
    }

    host_ans_m_vectors_data = (unsigned long*) malloc(host_ans_m_vectors_data_size);
    if (host_ans_m_vectors_data == nullptr) {
        fprintf(stderr, "Error: Failed to allocate memory for host_ans_m_vectors_data\n");
        free(host_m_vectors_data);   // Clean up previous allocations
        free(host_m_vectors_modulus);
        throw std::runtime_error("Memory allocation failed");
    }

    host_ans_m_vectors_modulus = (unsigned long*) malloc(host_ans_m_vectors_modulus_size);
    if (host_ans_m_vectors_modulus == nullptr) {
        fprintf(stderr, "Error: Failed to allocate memory for host_ans_m_vectors_modulus\n");
        free(host_m_vectors_data);   // Clean up previous allocations
        free(host_m_vectors_modulus);
        free(host_ans_m_vectors_data);
        throw std::runtime_error("Memory allocation failed");
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
    free(host_m_vectors_data);
    free(host_m_vectors_modulus);
    free(host_ans_m_vectors_data);
    free(host_ans_m_vectors_modulus);
}

void cudaPortalForApproxSwitchCRTBasis::freeDeviceMemory() {
    cudaError_t err;

    // Free device_m_vectors_data and modulus
    if (device_m_vectors_data) {
        err = cudaFree(device_m_vectors_data);
        handleCUDAError("freeing device_m_vectors_data", err);
        device_m_vectors_data = nullptr;
    }
    if (device_m_vectors_modulus) {
        err = cudaFree(device_m_vectors_modulus);
        handleCUDAError("freeing device_m_vectors_modulus", err);
        device_m_vectors_modulus = nullptr;
    }

    // Free device_sum memory
    if (device_sum) {
        err = cudaFree(device_sum);
        handleCUDAError("freeing device_sum", err);
        device_sum = nullptr;
    }

    // Free device_ans_m_vectors_data and modulus
    if (device_ans_m_vectors_data) {
        err = cudaFree(device_ans_m_vectors_data);
        handleCUDAError("freeing device_ans_m_vectors_data", err);
        device_ans_m_vectors_data = nullptr;
    }
    if (device_ans_m_vectors_modulus) {
        err = cudaFree(device_ans_m_vectors_modulus);
        handleCUDAError("freeing device_ans_m_vectors_modulus", err);
        device_ans_m_vectors_modulus = nullptr;
    }
}

void cudaPortalForApproxSwitchCRTBasis::print_host_m_vectors() {

    std::cout << "cudaPortalForApproxModDown::print_host_m_vectors" << std::endl;

    for (uint32_t q = 0; q < sizeQ; ++q) {
        std::cout << "host_m_vectors[" << q << "].data[0-3/ringDim]: ";
        for (uint32_t rd = 0; rd < 3; ++rd) {
            std::cout << host_m_vectors_data[(q * ringDim) + rd] << " ";
        }
        std::cout << std::endl;
    }
}

}
