#ifndef CUDAPORTALFORPARAMDATA_H
#define CUDAPORTALFORPARAMDATA_H

#include <cstdint> // for uint32_t type
#include <iostream> // for printf
#include <iomanip>
#include <vector>
#include <cstdint>
#include <cstdlib>

#include "lattice/poly.h"
#include <cuda_runtime.h>

namespace lbcrypto {
    /**
     * This class: 1) holds all data related to parameters (QHatInvModq, QHatInvModqPrecon, QhatModp, modpBarrettMu)
     * and, 2) implements all functionality needed related to CUDA for these data.
     */
    class cudaPortalForParamsData {

private:
    cudaStream_t        paramsStream;

    uint32_t            ringDim;
    uint32_t            sizeP;
    uint32_t            sizeQ;

    unsigned long*      host_qhatinvmodq;
    unsigned long*      host_QHatInvModqPrecon;
    uint128_t*          host_qhatmodp;
    uint128_t*          host_modpBarrettMu;

    ulong*              device_QHatInvModq;
    ulong*              device_QHatInvModqPrecon;
    uint128_t*          device_QHatModp;
    uint128_t*          device_modpBarrettMu;

    void allocateHostParams();

    static void handleMallocError(const std::string& allocationName, void* ptr);
    static void freePtrAndHandleError(const std::string& operation, void* ptr);
    static void freeCUDAPtrAndHandleError(void* device_ptr);
    static void handleCUDAError(const std::string& operation, cudaError_t err);

    void freeHostMemory() const;
    void freeDeviceMemory() const;

public:

    // Constructor
    cudaPortalForParamsData(uint32_t ringDim, uint32_t sizeP, uint32_t sizeQ, cudaStream_t stream);

    // Destructor
    ~cudaPortalForParamsData();

    // Get Functions
    uint32_t            getRingDim();
    uint32_t            getSizeP();
    uint32_t            getSizeQ();

    ulong*              getDevice_QHatInvModq();
    ulong*              getHost_qhatinvmodq();
    ulong*              getDevice_QHatInvModqPrecon();
    ulong*              getHost_QHatInvModqPrecon();
    uint128_t*          getDevice_QHatModp();
    uint128_t*          getDevice_modpBarrettMu();

    void marshalParams(const std::vector<NativeInteger>& QHatInvModq,
                       const std::vector<NativeInteger>& QHatInvModqPrecon,
                       const std::vector<std::vector<NativeInteger>>& QHatModp,
                       const std::vector<DoubleNativeInt>& modpBarrettMu) const;

    void copyInParams();

    static void printUint128(unsigned __int128 value);

    void printParams() const;
};
}

#endif //CUDAPORTALFORPARAMDATA_H
