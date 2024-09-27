#include "cuda-utils/cudaPortalForParamsData.h"

namespace lbcrypto {

using uint128_t = unsigned __int128;

cudaPortalForParamsData::cudaPortalForParamsData(uint32_t ringDim, uint32_t sizeP, uint32_t sizeQ)
    : ringDim(ringDim), sizeP(sizeP), sizeQ(sizeQ) // Initializer list
{
    //std::cout << "[CONSTRUCTOR] Call constructor for " << this << "(cudaPortalForParamsData)" << std::endl;
    createCUDAStream();
    allocateHostParams();
}

cudaPortalForParamsData::~cudaPortalForParamsData() {
    //std::cout << "[DESTRUCTOR] Call destructor for " << this << "(cudaPortalForParamsData)" << std::endl;

    destroyCUDAStream();
    freeHostMemory();
    freeDeviceMemory();

}

// Get Functions
uint32_t    cudaPortalForParamsData::getRingDim() { return ringDim;}
uint32_t    cudaPortalForParamsData::getSizeP() { return sizeP;}
uint32_t    cudaPortalForParamsData::getSizeQ() { return sizeQ;}

ulong*      cudaPortalForParamsData::getDevice_QHatInvModq() { return device_QHatInvModq;}
ulong*      cudaPortalForParamsData::getHost_qhatinvmodq() {return host_qhatinvmodq;}
ulong*      cudaPortalForParamsData::getDevice_QHatInvModqPrecon() { return device_QHatInvModqPrecon;}
ulong*      cudaPortalForParamsData::getHost_QHatInvModqPrecon() { return host_QHatInvModqPrecon;}
uint128_t*  cudaPortalForParamsData::getDevice_QHatModp() { return device_QHatModp;}
uint128_t*  cudaPortalForParamsData::getDevice_modpBarrettMu() { return device_modpBarrettMu;}

// Data Marshaling Functions
void cudaPortalForParamsData::marshalParams(const std::vector<NativeInteger>& QHatInvModq,
                                            const std::vector<NativeInteger>& QHatInvModqPrecon,
                                            const std::vector<std::vector<NativeInteger>>& QHatModp,
                                            const std::vector<DoubleNativeInt>& modpBarrettMu) const {
    for (uint32_t q = 0; q < sizeQ; ++q) {
        host_qhatinvmodq[q] = QHatInvModq[q].ConvertToInt();
        host_QHatInvModqPrecon[q] = QHatInvModqPrecon[q].ConvertToInt();
        for (uint32_t sp = 0; sp < sizeP; sp++) {
            host_qhatmodp[q * sizeP + sp] = QHatModp[q][sp].ConvertToInt();
        }
    }
    for (uint32_t sp = 0; sp < sizeP; sp++) {
        host_modpBarrettMu[sp] = modpBarrettMu[sp];
    }
}

// Copy-In Functions

void cudaPortalForParamsData::copyInParams() {
    cudaError_t err;

    // Allocate device_QHatInvModq
    err = cudaMalloc((void**)&device_QHatInvModq, sizeQ * sizeof(unsigned long));
    if (err != cudaSuccess) {
        printf("Error allocating device_QHatInvModq: %s (%d)\n", cudaGetErrorString(err), err);
        return; // or handle error appropriately
    }

    // Copy host_qhatinvmodq to device_QHatInvModq
    err = cudaMemcpyAsync(device_QHatInvModq, host_qhatinvmodq, sizeQ * sizeof(unsigned long), cudaMemcpyHostToDevice, paramsStream);
    if (err != cudaSuccess) {
        printf("Error copying to device_QHatInvModq: %s (%d)\n", cudaGetErrorString(err), err);
        return; // or handle error appropriately
    }

    // Allocate device_QHatInvModqPrecon
    err = cudaMalloc((void**)&device_QHatInvModqPrecon, sizeQ * sizeof(unsigned long));
    if (err != cudaSuccess) {
        printf("Error allocating device_QHatInvModqPrecon: %s (%d)\n", cudaGetErrorString(err), err);
        return; // or handle error appropriately
    }

    // Copy host_QHatInvModqPrecon to device_QHatInvModqPrecon
    err = cudaMemcpyAsync(device_QHatInvModqPrecon, host_QHatInvModqPrecon, sizeQ * sizeof(unsigned long), cudaMemcpyHostToDevice, paramsStream);
    if (err != cudaSuccess) {
        printf("Error copying to device_QHatInvModqPrecon: %s (%d)\n", cudaGetErrorString(err), err);
        return; // or handle error appropriately
    }

    // Allocate device_QHatModp
    err = cudaMalloc((void**)&device_QHatModp, sizeQ * sizeP * sizeof(uint128_t));
    if (err != cudaSuccess) {
        printf("Error allocating device_QHatModp: %s (%d)\n", cudaGetErrorString(err), err);
        return; // or handle error appropriately
    }

    // Copy host_qhatmodp to device_QHatModp
    err = cudaMemcpyAsync(device_QHatModp, host_qhatmodp, sizeQ * sizeP * sizeof(uint128_t), cudaMemcpyHostToDevice, paramsStream);
    if (err != cudaSuccess) {
        printf("Error copying to device_QHatModp: %s (%d)\n", cudaGetErrorString(err), err);
        return; // or handle error appropriately
    }

    // Allocate device_modpBarrettMu
    err = cudaMalloc((void**)&device_modpBarrettMu, sizeP * sizeof(uint128_t));
    if (err != cudaSuccess) {
        printf("Error allocating device_modpBarrettMu: %s (%d)\n", cudaGetErrorString(err), err);
        return; // or handle error appropriately
    }

    // Copy host_modpBarrettMu to device_modpBarrettMu
    err = cudaMemcpyAsync(device_modpBarrettMu, host_modpBarrettMu, sizeP * sizeof(uint128_t), cudaMemcpyHostToDevice, paramsStream);
    if (err != cudaSuccess) {
        printf("Error copying to device_modpBarrettMu: %s (%d)\n", cudaGetErrorString(err), err);
        return; // or handle error appropriately
    }
}

// Resources Allocation/Deallocation - Error Handling - Misc Functions

void cudaPortalForParamsData::handleMallocError(const std::string& allocationName, void* ptr) {
    if (ptr == nullptr) {
        throw std::runtime_error("Memory allocation failed for " + allocationName);
    }
}

void cudaPortalForParamsData::freePtrAndHandleError(const std::string& operation, void* ptr) {
    if (ptr == nullptr) {
        throw std::runtime_error("Memory error during " + operation + ": null pointer passed for freeing.");
    } else {
        free(ptr);  // Actual free operation happens here
        ptr = nullptr;  // Reset to nullptr after free
    }
}

void cudaPortalForParamsData::freeCUDAPtrAndHandleError(void* device_ptr) {
    cudaError_t err;
    if (device_ptr) {
        err = cudaFree(device_ptr);
        handleCUDAError("cudaFree", err);
    }
}

void cudaPortalForParamsData::handleCUDAError(const std::string& operation, cudaError_t err) {
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error during " + operation + ": " + std::string(cudaGetErrorString(err)));
    }
}

void cudaPortalForParamsData::createCUDAStream() {
    cudaError_t err = cudaStreamCreate(&paramsStream);
    if (err != cudaSuccess) {
        handleCUDAError("stream creation", err);
    }
}

void cudaPortalForParamsData::allocateHostParams() {
    host_qhatinvmodq        = (unsigned long*) malloc(sizeQ * sizeof(unsigned long));
    handleMallocError("host_qhatinvmodq", host_qhatinvmodq);
    host_QHatInvModqPrecon  = (unsigned long*) malloc(sizeQ * sizeof(unsigned long));
    handleMallocError("host_QHatInvModqPrecon", host_QHatInvModqPrecon);
    host_qhatmodp           = (uint128_t*) malloc(sizeQ * sizeP * sizeof(uint128_t));
    handleMallocError("host_qhatmodp", host_qhatmodp);
    host_modpBarrettMu      = (uint128_t*) malloc(sizeP * sizeof(uint128_t));
    handleMallocError("host_modpBarrettMu", host_modpBarrettMu);
}

void cudaPortalForParamsData::destroyCUDAStream() {
    if (paramsStream) {
        cudaError_t err = cudaStreamDestroy(paramsStream);
        if (err != cudaSuccess) {
            handleCUDAError("stream destruction", err);
        }
        paramsStream = nullptr; // Set pointer to nullptr after destroying
    }
}

void cudaPortalForParamsData::freeHostMemory() const {
    freePtrAndHandleError("host_qhatinvmodq", host_qhatinvmodq);
    freePtrAndHandleError("host_QHatInvModqPrecon",host_QHatInvModqPrecon);
    freePtrAndHandleError("host_qhatmodp",host_qhatmodp);
    freePtrAndHandleError("host_modpBarrettMu", host_modpBarrettMu);
}

void cudaPortalForParamsData::freeDeviceMemory() const {
    freeCUDAPtrAndHandleError(device_QHatInvModq);
    freeCUDAPtrAndHandleError(device_QHatInvModqPrecon);
    freeCUDAPtrAndHandleError(device_QHatModp);
    freeCUDAPtrAndHandleError(device_modpBarrettMu);
}

void cudaPortalForParamsData::printUint128(uint128_t value) {
    // Cast the higher and lower 64 bits of the uint128_t value
    uint64_t high = static_cast<uint64_t>(value >> 64); // Upper 64 bits
    uint64_t low = static_cast<uint64_t>(value);        // Lower 64 bits

    // Print the parts in hex or as two parts (high, low)
    std::cout << "0x" << std::hex << high << std::setw(16) << std::setfill('0') << low << std::dec;
}

void cudaPortalForParamsData::printParams() const {
    std::cout << "cudaPortalForParamsData::printParams" << std::endl;
    // Print host_QHatInvModq
    std::cout << "host_QHatInvModq: ";
    for (uint32_t q = 0; q < sizeQ; ++q) {
        std::cout << host_qhatinvmodq[q] << " ";
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
            printUint128(host_qhatmodp[q * sizeP + sp]);
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

}
