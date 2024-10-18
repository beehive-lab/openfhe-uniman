#ifndef CUDAPORTALFORAPPROXMODDOWNPARAMS_H
#define CUDAPORTALFORAPPROXMODDOWNPARAMS_H

#include <cstdint> // for uint32_t type
#include <iostream> // for printf
#include <iomanip>
#include <vector>
#include <cstdlib>

#include "lattice/poly.h"
#include <cuda_runtime.h>
#include <cuda-utils/cuda-data-utils.h>

namespace lbcrypto {

    class cudaPortalForApproxModDownParams {

    private:
        cudaStream_t        paramsStream;
        uint32_t            ringDim;
        uint32_t            sizeQP;
        uint32_t            sizeQ;
        uint32_t            sizeP;

        // sizes
        uint32_t    PInvModq_size;             // (size1) should = PInvModqPrecon_size = modqBarrettMu_size = tModqPrecon
        uint32_t    PInvModqPrecon_size;       // (size1)
        uint32_t    PHatInvModp_size;          // (size2) should = PHatInvModpPrecon_size = PHatModq_size_x = tInvModp_size = tInvModpPrecon
        uint32_t    PHatInvModpPrecon_size;    // (size2)
        uint32_t    PHatModq_size_x;           // (size2)
        uint32_t    PHatModq_size_y;           // (size1)
        uint32_t    modqBarrettMu_size;        // (size1)
        uint32_t    tInvModp_size;             // (size2)
        uint32_t    tInvModpPrecon_size;       // (size2)
        uint32_t    tModqPrecon_size;          // (size1)

        // host buffers
        ulong*        host_PInvModq;
        ulong*        host_PInvModqPrecon;
        ulong*        host_PHatInvModp;
        ulong*        host_PHatInvModpPrecon;
        uint128_t*    host_PHatModq;
        uint128_t*    host_modqBarrettMu;
        ulong*        host_tInvModp;
        ulong*        host_tInvModpPrecon;
        ulong*        host_tModqPrecon;

        // device buffers
        ulong*        device_PInvModq;
        ulong*        device_PInvModqPrecon;
        ulong*        device_PHatInvModp;
        ulong*        device_PHatInvModpPrecon;
        uint128_t*    device_PHatModq;
        uint128_t*    device_modqBarrettMu;
        ulong*        device_tInvModp;
        ulong*        device_tInvModpPrecon;
        ulong*        device_tModqPrecon;

        void allocateHostParams();

        static void handleMallocError(const std::string& allocationName, void* ptr);
        static void freePtrAndHandleError(const std::string& operation, void* ptr);
        void freeCUDAPtrAndHandleError(void* device_ptr) const;
        static void handleCUDAError(const std::string& operation, cudaError_t err);

        void freeHostMemory() const;
        void freeDeviceMemory() const;

    public:

        // Constructor
        cudaPortalForApproxModDownParams(uint32_t ringDim, uint32_t sizeQP, uint32_t sizeQ, uint32_t sizeP, cudaStream_t stream,
                            uint32_t PInvModq_size, uint32_t PInvModqPrecon_size, uint32_t PHatInvModp_size,
                            uint32_t PHatInvModpPrecon_size, uint32_t PHatModq_size_x, uint32_t PHatModq_size_y,
                            uint32_t modqBarrettMu_size, uint32_t tInvModp_size, uint32_t tInvModpPrecon_size,
                            uint32_t tModqPrecon_size);

        // Destructor
        ~cudaPortalForApproxModDownParams();

        // Get Functions
        uint32_t get_RingDim() const {
            return ringDim;
        }

        uint32_t get_sizeQP() const {
            return sizeQP;
        }

        uint32_t get_sizeQ() const {
            return sizeQ;
        }

        uint32_t get_sizeP() const {
            return sizeP;
        }

        ulong* get_device_PInvModq() const {
            return device_PInvModq;
        }

        ulong* get_device_PInvModqPrecon() const {
            return device_PInvModqPrecon;
        }

        ulong* get_device_PHatInvModp() const {
            return device_PHatInvModp;
        }

        ulong* get_device_PHatInvModpPrecon() const {
            return device_PHatInvModpPrecon;
        }

        uint128_t* get_device_PHatModq() const {
            return device_PHatModq;
        }

        uint32_t get_PHatModq_sizeY() const {
            return PHatModq_size_y;
        }

        uint128_t* get_device_modqBarrettMu() const {
            return device_modqBarrettMu;
        }

        ulong* get_device_tInvModp() const {
            return device_tInvModp;
        }

        ulong* get_device_tInvModpPrecon() const {
            return device_tInvModpPrecon;
        }

        ulong* get_device_tModqPrecon() const {
            return device_tModqPrecon;
        }

        void marshalParams(const std::vector<NativeInteger>& PInvModq, const std::vector<NativeInteger>& PInvModqPrecon,
                           const std::vector<NativeInteger>& PHatInvModp, const std::vector<NativeInteger>& PHatInvModpPrecon,
                           const std::vector<std::vector<NativeInteger>>& PHatModq, const std::vector<DoubleNativeInt>& modqBarrettMu,
                           const std::vector<NativeInteger>& tInvModp, const std::vector<NativeInteger>& tInvModpPrecon,
                           const std::vector<NativeInteger>& tModqPrecon) const;

        void copyInParams();

        static void printUint128(unsigned __int128 value);

        void printParams() const;
    };
}

#endif //CUDAPORTALFORAPPROXMODDOWNPARAMS_H
