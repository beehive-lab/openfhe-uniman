#include "cuda-utils/approxModDown/cudaPortalForApproxModDownParams.h"

namespace lbcrypto {

using uint128_t = unsigned __int128;

cudaPortalForApproxModDownParams::cudaPortalForApproxModDownParams(
    uint32_t ringDim, uint32_t sizeQP, uint32_t sizeQ, uint32_t sizeP, cudaStream_t stream,
    uint32_t PInvModq_size, uint32_t PInvModqPrecon_size, uint32_t PHatInvModp_size,
    uint32_t PHatInvModpPrecon_size, uint32_t PHatModq_size_x, uint32_t PHatModq_size_y,
    uint32_t modqBarrettMu_size, uint32_t tInvModp_size, uint32_t tInvModpPrecon_size, uint32_t t, uint32_t tModqPrecon_size)
: ringDim(ringDim), sizeQP(sizeQP), sizeQ(sizeQ), sizeP(sizeP), t(t),
PInvModq_size(PInvModq_size), PInvModqPrecon_size(PInvModqPrecon_size), PHatInvModp_size(PHatInvModp_size),
PHatInvModpPrecon_size(PHatInvModpPrecon_size), PHatModq_size_x(PHatModq_size_x), PHatModq_size_y(PHatModq_size_y),
modqBarrettMu_size(modqBarrettMu_size), tInvModp_size(tInvModp_size), tInvModpPrecon_size(tInvModpPrecon_size),
tModqPrecon_size(tModqPrecon_size) {

    this->paramsStream = stream;
    allocateHostParams();
}

cudaPortalForApproxModDownParams::~cudaPortalForApproxModDownParams() {
    freeHostMemory();
    freeDeviceMemory();
}

void cudaPortalForApproxModDownParams::allocateHostParams() {
    cudaHostAlloc((void**)&host_PInvModq, PInvModq_size * sizeof(unsigned long), cudaHostAllocDefault);
    cudaHostAlloc((void**)&host_PInvModqPrecon , PInvModqPrecon_size * sizeof(unsigned long), cudaHostAllocDefault);
    cudaHostAlloc((void**)&host_PHatInvModp , PHatInvModp_size * sizeof(unsigned long), cudaHostAllocDefault);
    cudaHostAlloc((void**)&host_PHatInvModpPrecon , PHatInvModpPrecon_size * sizeof(unsigned long), cudaHostAllocDefault);
    cudaHostAlloc((void**)&host_PHatModq , PHatModq_size_x * PHatModq_size_y * sizeof(uint128_t), cudaHostAllocDefault);
    cudaHostAlloc((void**)&host_modqBarrettMu , modqBarrettMu_size * sizeof(uint128_t), cudaHostAllocDefault);
    cudaHostAlloc((void**)&host_tInvModp , tInvModp_size * sizeof(unsigned long), cudaHostAllocDefault);
    cudaHostAlloc((void**)&host_tInvModpPrecon , tInvModpPrecon_size * sizeof(unsigned long), cudaHostAllocDefault);
    cudaHostAlloc((void**)&host_tModqPrecon , tModqPrecon_size * sizeof(unsigned long), cudaHostAllocDefault);
}

// Data Marshaling Functions
// TODO: opt by merging for loops
void cudaPortalForApproxModDownParams::marshalParams(const std::vector<NativeInteger>& PInvModq, const std::vector<NativeInteger>& PInvModqPrecon,
                           const std::vector<NativeInteger>& PHatInvModp, const std::vector<NativeInteger>& PHatInvModpPrecon,
                           const std::vector<std::vector<NativeInteger>>& PHatModq, const std::vector<DoubleNativeInt>& modqBarrettMu,
                           const std::vector<NativeInteger>& tInvModp, const std::vector<NativeInteger>& tInvModpPrecon,
                           const std::vector<NativeInteger>& tModqPrecon) const {
    uint32_t i;
    for (i = 0; i < PInvModq_size; ++i)
        host_PInvModq[i] = PInvModq[i].ConvertToInt<>();
    for (i = 0; i < PInvModqPrecon_size; ++i)
        host_PInvModqPrecon[i] = PInvModqPrecon[i].ConvertToInt<>();
    for (i = 0; i < PHatInvModp_size; ++i)
        host_PHatInvModp[i] = PHatInvModp[i].ConvertToInt<>();
    for (i = 0; i < PHatInvModpPrecon_size; ++i)
        host_PHatInvModpPrecon[i] = PHatInvModpPrecon[i].ConvertToInt<>();

    //std::cout << "(marshalParams): sizeQ = "<< sizeQ << ", sizeP = " << sizeP << ", PHatModq_size_x = " << PHatModq_size_x << ", PHatModq_size_y = " << PHatModq_size_y << std::endl;
    for (i = 0; i < PHatModq_size_x; ++i) {
        for (uint32_t j = 0; j < PHatModq_size_y; ++j) {
            host_PHatModq[i * PHatModq_size_y + j] = PHatModq[i][j].ConvertToInt<>();
            //std::cout << "host_PHatModq[" << i * PHatModq_size_y + j<< "] = ";
            //printUint128(host_PHatModq[i * PHatModq_size_y + j]);
            //std::cout << std::endl;
        }

    }
    for (i = 0; i < modqBarrettMu_size; ++i)
        host_modqBarrettMu[i] = modqBarrettMu[i];
    for (i = 0; i < tInvModp_size; ++i)
        host_tInvModp[i] = tInvModp[i].ConvertToInt<>();
    for (i = 0; i < tInvModpPrecon_size; ++i)
        host_tInvModpPrecon[i] = tInvModpPrecon[i].ConvertToInt<>();
    for (i = 0; i < tModqPrecon_size; ++i)
        host_tModqPrecon[i] = tModqPrecon[i].ConvertToInt<>();
}

void cudaPortalForApproxModDownParams::copyInParams() {

    // Allocate
    CUDA_CHECK(cudaMallocAsync((void**)&device_PInvModq,          PInvModq_size * sizeof(unsigned long), paramsStream));
    CUDA_CHECK(cudaMallocAsync((void**)&device_PInvModqPrecon,    PInvModqPrecon_size * sizeof(unsigned long), paramsStream));
    CUDA_CHECK(cudaMallocAsync((void**)&device_PHatInvModp,       PHatInvModp_size * sizeof(unsigned long), paramsStream));
    CUDA_CHECK(cudaMallocAsync((void**)&device_PHatInvModpPrecon, PHatInvModpPrecon_size * sizeof(unsigned long), paramsStream));
    CUDA_CHECK(cudaMallocAsync((void**)&device_PHatModq,      PHatModq_size_x * PHatModq_size_y * sizeof(uint128_t), paramsStream));
    CUDA_CHECK(cudaMallocAsync((void**)&device_modqBarrettMu,     modqBarrettMu_size * sizeof(uint128_t), paramsStream));
    CUDA_CHECK(cudaMallocAsync((void**)&device_tInvModp,          tInvModp_size * sizeof(unsigned long), paramsStream));
    CUDA_CHECK(cudaMallocAsync((void**)&device_tInvModpPrecon,    tInvModpPrecon_size * sizeof(unsigned long), paramsStream));
    CUDA_CHECK(cudaMallocAsync((void**)&device_tModqPrecon,       tModqPrecon_size * sizeof(unsigned long), paramsStream));

    // Copy to device
    CUDA_CHECK(cudaMemcpyAsync(device_PInvModq, host_PInvModq, PInvModq_size * sizeof(unsigned long), cudaMemcpyHostToDevice, paramsStream));
    CUDA_CHECK(cudaMemcpyAsync(device_PInvModqPrecon, host_PInvModqPrecon, PInvModqPrecon_size * sizeof(unsigned long), cudaMemcpyHostToDevice, paramsStream));
    CUDA_CHECK(cudaMemcpyAsync(device_PHatInvModp, host_PHatInvModp, PHatInvModp_size * sizeof(unsigned long), cudaMemcpyHostToDevice, paramsStream));
    CUDA_CHECK(cudaMemcpyAsync(device_PHatInvModpPrecon, host_PHatInvModpPrecon, PHatInvModpPrecon_size * sizeof(unsigned long), cudaMemcpyHostToDevice, paramsStream));
    CUDA_CHECK(cudaMemcpyAsync(device_PHatModq, host_PHatModq, PHatModq_size_x * PHatModq_size_y * sizeof(uint128_t), cudaMemcpyHostToDevice, paramsStream));
    CUDA_CHECK(cudaMemcpyAsync(device_modqBarrettMu, host_modqBarrettMu, modqBarrettMu_size * sizeof(uint128_t), cudaMemcpyHostToDevice, paramsStream));
    CUDA_CHECK(cudaMemcpyAsync(device_tInvModp, host_tInvModp, tInvModp_size * sizeof(unsigned long), cudaMemcpyHostToDevice, paramsStream));
    CUDA_CHECK(cudaMemcpyAsync(device_tInvModpPrecon, host_tInvModpPrecon, tInvModpPrecon_size * sizeof(unsigned long), cudaMemcpyHostToDevice, paramsStream));
    CUDA_CHECK(cudaMemcpyAsync(device_tModqPrecon, host_tModqPrecon, tModqPrecon_size * sizeof(unsigned long), cudaMemcpyHostToDevice, paramsStream));
}

// Resources Deallocation - Error Handling - Misc Functions

void cudaPortalForApproxModDownParams::freeHostMemory() const {
    cudaFreeHost(host_PInvModq);
    cudaFreeHost(host_PInvModqPrecon);
    cudaFreeHost(host_PHatInvModp);
    cudaFreeHost(host_PHatInvModpPrecon);
    cudaFreeHost(host_PHatModq);
    cudaFreeHost(host_modqBarrettMu);
    cudaFreeHost(host_tInvModp);
    cudaFreeHost(host_tInvModpPrecon);
    cudaFreeHost(host_tModqPrecon);
}

void cudaPortalForApproxModDownParams::freeDeviceMemory() const {
    CUDA_CHECK(cudaFreeAsync(device_PInvModq, paramsStream));
    CUDA_CHECK(cudaFreeAsync(device_PInvModqPrecon, paramsStream));
    CUDA_CHECK(cudaFreeAsync(device_PHatInvModp, paramsStream));
    CUDA_CHECK(cudaFreeAsync(device_PHatInvModpPrecon, paramsStream));
    CUDA_CHECK(cudaFreeAsync(device_modqBarrettMu, paramsStream));
    CUDA_CHECK(cudaFreeAsync(device_tInvModp, paramsStream));
    CUDA_CHECK(cudaFreeAsync(device_tInvModpPrecon, paramsStream));
    CUDA_CHECK(cudaFreeAsync(device_tModqPrecon, paramsStream));
}

void cudaPortalForApproxModDownParams::handleMallocError(const std::string& allocationName, void* ptr) {
    if (ptr == nullptr) {
        throw std::runtime_error("Memory allocation failed for " + allocationName);
    }
}

void cudaPortalForApproxModDownParams::freePtrAndHandleError(const std::string& operation, void* ptr) {
    if (ptr == nullptr) {
        throw std::runtime_error("Memory error during " + operation + ": null pointer passed for freeing.");
    } else {
        free(ptr);  // Actual free operation happens here
        ptr = nullptr;  // Reset to nullptr after free
    }
}

void cudaPortalForApproxModDownParams::printUint128(uint128_t value) {
    // Cast the higher and lower 64 bits of the uint128_t value
    uint64_t high = static_cast<uint64_t>(value >> 64); // Upper 64 bits
    uint64_t low = static_cast<uint64_t>(value);        // Lower 64 bits

    // Print the parts in hex or as two parts (high, low)
    std::cout << "0x" << std::hex << high << std::setw(16) << std::setfill('0') << low << std::dec;
}

}