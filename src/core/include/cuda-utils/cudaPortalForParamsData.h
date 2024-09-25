#ifndef CUDAPORTALFORPARAMDATA_H
#define CUDAPORTALFORPARAMDATA_H

#include <cstdint> // for uint32_t type
#include <cuda_runtime.h>
#include <cuda-utils/cuda-data-utils.h>
#include "cuda-utils/kernel-headers/approx-switch-crt-basis.cuh"

namespace lbcrypto {

class cudaPortalForParamsData {

protected:

    cudaStream_t paramsStream;

private:

    uint32_t             ringDim;
    uint32_t             sizeP;
    uint32_t             sizeQ;

    unsigned long*      host_qhatinvmodq;
    unsigned long*      host_QHatInvModqPrecon;
    uint128_t*          host_qhatmodp;
    uint128_t*          host_modpBarrettMu;

    ulong*              device_QHatInvModq;
    ulong*              device_QHatInvModqPrecon;
    uint128_t*          device_QHatModp;
    uint128_t*          device_modpBarrettMu;

public:

    //constructor
    cudaPortalForParamsData(uint32_t ringDim, uint32_t sizeP, uint32_t sizeQ);

    ~cudaPortalForParamsData();

    uint32_t            getRingDim();
    uint32_t            getSizeP();
    uint32_t            getSizeQ();

    ulong*              getDevice_QHatInvModq();
    ulong*              getHost_qhatinvmodq();
    ulong*              getDevice_QHatInvModqPrecon();
    ulong*              getHost_QHatInvModqPrecon();
    uint128_t*          getDevice_QHatModp();
    uint128_t*          getDevice_modpBarrettMu();

    static void printUint128(unsigned __int128 value);

    void printParams();

    void allocateHostParams(uint32_t sizeP, uint32_t sizeQ);

    void marshalParams(uint32_t sizeP, uint32_t sizeQ,
                       const std::vector<NativeInteger>& QHatInvModq,
                       const std::vector<NativeInteger>& QHatInvModqPrecon,
                       const std::vector<std::vector<NativeInteger>>& QHatModp,
                       const std::vector<DoubleNativeInt>& modpBarrettMu);

    void copyInParams(uint32_t sizeP, uint32_t sizeQ);
};
}

#endif //CUDAPORTALFORPARAMDATA_H
