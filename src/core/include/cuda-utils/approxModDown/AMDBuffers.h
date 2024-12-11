#ifndef AMDBUFFERS_H
#define AMDBUFFERS_H

#include <cstdint> // for uint32_t type
#include "lattice/poly.h"
#include <cuda_runtime.h>

namespace lbcrypto {

/**
 * This class holds the necessary buffers (host & device) for the ApproxModDown CUDA implementation.
 * The purpose of this class is to allocate the buffers once and facilitate their reuse.
 *
 */
class AMDBuffers {
public:
    uint128_t* get_host_p_hat_modq() const {
        return host_PHatModq;
    }

    uint128_t* get_device_p_hat_modq() const {
        return device_PHatModq;
    }

    ulong* get_host_c_tilda() const {
        return host_cTilda;
    }

    ulong* get_device_part_p_empty() const {
        return device_partP_empty;
    }

    ulong* get_device_part_p_switched_to_q() const {
        return device_partPSwitchedToQ;
    }

    uint128_t* get_device_sum() const {
        return device_sum;
    }

    ulong* get_host_c_tilda_q() const {
        return host_cTildaQ;
    }

    ulong* get_device_c_tilda_q() const {
        return device_cTildaQ;
    }

    ulong* get_host_ans() const {
        return host_ans;
    }

    ulong* get_device_ans() const {
        return device_ans;
    }

    ulong* get_host_root_of_unity_rev() const {
        return host_rootOfUnityRev;
    }

    ulong* get_device_root_of_unity_rev() const {
        return device_rootOfUnityRev;
    }

    ulong* get_host_root_of_unity_precon_rev() const {
        return host_rootOfUnityPreconRev;
    }

    ulong* get_device_root_of_unity_precon_rev() const {
        return device_rootOfUnityPreconRev;
    }

    ulong* get_host_root_of_unity_inv_rev() const {
        return host_rootOfUnityInvRev;
    }

    ulong* get_device_root_of_unity_inv_rev() const {
        return device_rootOfUnityInvRev;
    }

    ulong* get_host_root_of_unity_inv_precon_rev() const {
        return host_rootOfUnityInvPreconRev;
    }

    ulong* get_device_root_of_unity_inv_precon_rev() const {
        return device_rootOfUnityInvPreconRev;
    }

private:
    uint128_t*    host_PHatModq;            // p
    uint128_t*    device_PHatModq;          // p
    ulong*        host_cTilda;              // p
    ulong*        device_partP_empty;       // p
    ulong*        device_partPSwitchedToQ;  // q
    uint128_t*    device_sum;               // q
    ulong*        host_cTildaQ;             // q
    ulong*        device_cTildaQ;           // q
    ulong*        host_ans;                 // q
    ulong*        device_ans;               // q

    ulong*          host_rootOfUnityRev;            // q
    ulong*          device_rootOfUnityRev;          // q
    ulong*          host_rootOfUnityPreconRev;      // q
    ulong*          device_rootOfUnityPreconRev;    // q
    ulong*          host_rootOfUnityInvRev;         // p
    ulong*          device_rootOfUnityInvRev;       // p
    ulong*          host_rootOfUnityInvPreconRev;   // p
    ulong*          device_rootOfUnityInvPreconRev; // p

    cudaStream_t  stream;

public:
    AMDBuffers(const uint32_t ringDim, const uint32_t sizeP, const uint32_t sizeQ, const uint32_t param_sizeY, cudaStream_t mainStream) {
        const size_t param_elements = sizeP * param_sizeY;
        const size_t p_elements = sizeP * ringDim;
        const size_t q_elements = sizeQ * ringDim;
        this->stream = mainStream;

        CUDA_CHECK(cudaHostAlloc    (reinterpret_cast<void**>(&host_PHatModq),           param_elements * sizeof(uint128_t), cudaHostAllocDefault));
        CUDA_CHECK(cudaMallocAsync  (reinterpret_cast<void**>(&device_PHatModq),         param_elements * sizeof(uint128_t), stream));
        CUDA_CHECK(cudaHostAlloc    (reinterpret_cast<void**>(&host_cTilda),             p_elements * sizeof(ulong), cudaHostAllocDefault));
        CUDA_CHECK(cudaMallocAsync  (reinterpret_cast<void**>(&device_partP_empty),      p_elements * sizeof(ulong), stream));
        CUDA_CHECK(cudaMallocAsync  (reinterpret_cast<void**>(&device_partPSwitchedToQ), q_elements * sizeof(ulong), stream));
        CUDA_CHECK(cudaMallocAsync  (reinterpret_cast<void**>(&device_sum),              q_elements * sizeof(uint128_t), stream));
        CUDA_CHECK(cudaHostAlloc    (reinterpret_cast<void**>(&host_cTildaQ),            q_elements * sizeof(ulong), cudaHostAllocDefault));
        CUDA_CHECK(cudaMallocAsync  (reinterpret_cast<void**>(&device_cTildaQ),          q_elements * sizeof(ulong), stream));
        CUDA_CHECK(cudaHostAlloc    (reinterpret_cast<void**>(&host_ans),                q_elements * sizeof(ulong), cudaHostAllocDefault));
        CUDA_CHECK(cudaMallocAsync  (reinterpret_cast<void**>(&device_ans),              q_elements * sizeof(ulong), stream));

        CUDA_CHECK(cudaHostAlloc    (reinterpret_cast<void**>(&host_rootOfUnityRev),            q_elements * sizeof(ulong), cudaHostAllocDefault));
        CUDA_CHECK(cudaMallocAsync  (reinterpret_cast<void**>(&device_rootOfUnityRev),          q_elements * sizeof(ulong), stream));
        CUDA_CHECK(cudaHostAlloc    (reinterpret_cast<void**>(&host_rootOfUnityPreconRev),      q_elements * sizeof(ulong), cudaHostAllocDefault));
        CUDA_CHECK(cudaMallocAsync  (reinterpret_cast<void**>(&device_rootOfUnityPreconRev),    q_elements * sizeof(ulong), stream));
        CUDA_CHECK(cudaHostAlloc    (reinterpret_cast<void**>(&host_rootOfUnityInvRev),         p_elements * sizeof(ulong), cudaHostAllocDefault));
        CUDA_CHECK(cudaMallocAsync  (reinterpret_cast<void**>(&device_rootOfUnityInvRev),       p_elements * sizeof(ulong), stream));
        CUDA_CHECK(cudaHostAlloc    (reinterpret_cast<void**>(&host_rootOfUnityInvPreconRev),   p_elements * sizeof(ulong), cudaHostAllocDefault));
        CUDA_CHECK(cudaMallocAsync  (reinterpret_cast<void**>(&device_rootOfUnityInvPreconRev), p_elements * sizeof(ulong), stream));

        //CUDA_CHECK(cudaMemsetAsync(device_sum, 0, q_elements * sizeof(uint128_t), stream));
    }

    ~AMDBuffers() {
        CUDA_CHECK(cudaFreeHost(host_PHatModq));
        CUDA_CHECK(cudaFreeAsync(device_PHatModq, stream));
        CUDA_CHECK(cudaFreeHost(host_cTilda));
        CUDA_CHECK(cudaFreeAsync(device_partP_empty, stream));
        CUDA_CHECK(cudaFreeAsync(device_partPSwitchedToQ, stream));
        CUDA_CHECK(cudaFreeAsync(device_sum, stream));
        CUDA_CHECK(cudaFreeHost(host_cTildaQ));
        CUDA_CHECK(cudaFreeAsync(device_cTildaQ, stream));
        CUDA_CHECK(cudaFreeHost(host_ans));
        CUDA_CHECK(cudaFreeAsync(device_ans, stream));

        CUDA_CHECK(cudaFreeHost(host_rootOfUnityRev));
        CUDA_CHECK(cudaFreeAsync(device_rootOfUnityRev, stream));
        CUDA_CHECK(cudaFreeHost(host_rootOfUnityPreconRev));
        CUDA_CHECK(cudaFreeAsync(device_rootOfUnityPreconRev, stream));
        CUDA_CHECK(cudaFreeHost(host_rootOfUnityInvRev));
        CUDA_CHECK(cudaFreeAsync(device_rootOfUnityInvRev, stream));
        CUDA_CHECK(cudaFreeHost(host_rootOfUnityInvPreconRev));
        CUDA_CHECK(cudaFreeAsync(device_rootOfUnityInvPreconRev, stream));
    }
};
}

#endif //AMDBUFFERS_H
