//#include "math/hal.h"
//#include "lattice/poly.h"
#include <cstdint> // for uint32_t type

#include "cuda-utils/cudaPortalForApproxModDown.h"
#include <cuda_runtime.h>
#include "cuda-utils/kernel-headers/approx-switch-crt-basis.cuh"

namespace lbcrypto {

using PolyType = PolyImpl<NativeVector>;

//constructor impl
cudaPortalForApproxModDown::cudaPortalForApproxModDown(uint32_t ringDim, uint32_t sizeP, uint32_t sizeQ, std::shared_ptr<cudaPortalForParamsData> params_data) {
    std::cout << "[CONSTRUCTOR] Call constructor for " << this << "(cudaPortalForApproxModDown)" << std::endl;

    this->paramsData = params_data;

    //this->ringDim = ringDim;
    //this->sizeP = sizeP;
    //this->sizeQ = sizeQ;

    //this->stream = stream;
    cudaStreamCreate(&this->stream);

    allocateHostData(ringDim, sizeP, sizeQ);
}

cudaPortalForApproxModDown::~cudaPortalForApproxModDown() {
    std::cout << "[DESTRUCTOR] Call destructor for " << this << "(cudaPortalForApproxModDown)" << std::endl;

    uint32_t sizeP = this->paramsData->getSizeP();
    uint32_t sizeQ = this->paramsData->getSizeQ();

    destroyCUDAStream();
    freeHostMemory(sizeP, sizeQ);
    freeDeviceMemory(sizeP, sizeQ);

}

void cudaPortalForApproxModDown::print_host_m_vectors() {

    std::cout << "cudaPortalForApproxModDown::print_host_m_vectors" << std::endl;

    uint32_t sizeQ = this->getParamsData()->getSizeQ();

    for (uint32_t q = 0; q < sizeQ; ++q) {
        std::cout << "host_m_vectors[" << q << "].data[0-3/ringDim]: ";
        for (uint32_t rd = 0; rd < 3; ++rd) {
            std::cout << host_m_vectors[q].data[rd] << " ";
        }
        std::cout << std::endl;
    }
}


// getters setters
cudaStream_t cudaPortalForApproxModDown::getStream() {
    return stream;
}

std::shared_ptr<cudaPortalForParamsData> cudaPortalForApproxModDown::getParamsData() {
    return paramsData;
}



uint128_t*          cudaPortalForApproxModDown::getDevice_sum() { return device_sum;}

m_vectors_struct*   cudaPortalForApproxModDown::getDevice_m_vectors() { return device_m_vectors;}
m_vectors_struct*   cudaPortalForApproxModDown::getDevice_ans_m_vectors() { return device_ans_m_vectors;}

m_vectors_struct*   cudaPortalForApproxModDown::getHost_ans_m_vectors() {return host_ans_m_vectors;}

// allocations

void cudaPortalForApproxModDown::allocateHostData(uint32_t ringDim, uint32_t sizeP, uint32_t sizeQ) {
    host_m_vectors          = (m_vectors_struct*) malloc(sizeQ * sizeof(m_vectors_struct));
    for (uint32_t q = 0; q < sizeQ; ++q) {
        host_m_vectors[q].data                  = (unsigned long*) malloc(ringDim * sizeof(unsigned long));
    }

    host_ans_m_vectors      = (m_vectors_struct*) malloc(sizeP * sizeof(m_vectors_struct));
    for (uint32_t p = 0; p < sizeP; ++p) {
        host_ans_m_vectors[p].data              = (unsigned long*) malloc(ringDim * sizeof(unsigned long));
    }
}

// marshal

void cudaPortalForApproxModDown::marshalWorkData(uint32_t ringDim, uint32_t sizeP, uint32_t sizeQ,
                                                 const std::vector<PolyImpl<NativeVector>> m_vectors,
                                                 const std::vector<PolyImpl<NativeVector>> ans_m_vectors) {

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
    std::cout << "[DEBUG Print]: " << std::endl;
    for (uint32_t q = 0; q < sizeQ; ++q) {
        for (uint32_t rd = 0; rd < 3; ++rd) {
            std::cout << "host_m_vectors[" << q << "].data[" << rd << "] = " << host_m_vectors[q].data[rd] <<
                         ", m_vectors["      << q << "]["      << rd << "] = " << m_vectors[q][rd].template ConvertToInt<>() << std::endl;
        }
    }
}

void cudaPortalForApproxModDown::copyInWorkData(uint32_t ringDim, uint32_t sizeP, uint32_t sizeQ) {

    cudaError_t err;

    err = cudaMalloc((void**)&device_m_vectors, sizeQ * sizeof(m_vectors_struct));
    if (err != cudaSuccess) {
        printf("Error allocating device_m_vectors: %s (%d)\n", cudaGetErrorString(err), err);
        exit(-1); // or handle error appropriately
    }

    err = cudaMemcpyAsync(device_m_vectors, host_m_vectors, sizeQ * sizeof(m_vectors_struct), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        printf("Error copying to device_m_vectors: %s (%d)\n", cudaGetErrorString(err), err);
        return; // or handle error appropriately
    }

    this->device_m_vectors_data_ptr = (unsigned long**)malloc(sizeQ * sizeof(unsigned long*));
    if (!device_m_vectors_data_ptr) {
        printf("Error allocating device_m_vectors_data_ptr\n");
        return; // or handle error appropriately
    }

    for (uint32_t q = 0; q < sizeQ; ++q) {
        err = cudaMalloc((void**)&(device_m_vectors_data_ptr[q]), ringDim * sizeof(unsigned long));
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
    err = cudaMalloc((void**)&device_sum, sizeP * ringDim * sizeof(uint128_t));
    if (err != cudaSuccess) {
        printf("Error allocating device_sum: %s (%d)\n", cudaGetErrorString(err), err);
        return; // or handle error appropriately
    }

    err = cudaMemset(device_sum, 0, sizeP * ringDim * sizeof(uint128_t));
    if (err != cudaSuccess) {
        printf("Error setting device_sum to zero: %s (%d)\n", cudaGetErrorString(err), err);
        return; // or handle error appropriately
    }

    // ans_m_vectors
    err = cudaMalloc((void**)&device_ans_m_vectors, sizeP * sizeof(m_vectors_struct));
    if (err != cudaSuccess) {
        printf("Error allocating device_ans_m_vectors: %s (%d)\n", cudaGetErrorString(err), err);
        return; // or handle error appropriately
    }

    err = cudaMemcpyAsync(device_ans_m_vectors, host_ans_m_vectors, sizeP * sizeof(m_vectors_struct), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        printf("Error copying to device_ans_m_vectors: %s (%d)\n", cudaGetErrorString(err), err);
        return; // or handle error appropriately
    }

    ////
    this->device_ans_m_vectors_data_ptr = (unsigned long**)malloc(sizeP * sizeof(unsigned long*));


    for (uint32_t p = 0; p < sizeP; ++p) {
        err = cudaMalloc((void**)&(device_ans_m_vectors_data_ptr[p]), ringDim * sizeof(unsigned long));
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


// copy out

void cudaPortalForApproxModDown::copyOutResult(uint32_t ringDim, uint32_t sizeP) {

    for(uint32_t p = 0; p < sizeP; p++) {
        cudaMemcpyAsync(host_ans_m_vectors[p].data, device_ans_m_vectors_data_ptr[p], ringDim * sizeof(unsigned long), cudaMemcpyDeviceToHost, stream);
    }
}

// unmarshal
//template <typename VecType>
//DCRTPolyImpl<VecType> cudaPortalForApproxModDown::unmarshal(uint32_t ringDim, uint32_t sizeP, std::vector<PolyImpl<NativeVector>>& ans_m_vectors, m_vectors_struct*  host_ans_m_vectors) {
void cudaPortalForApproxModDown::unmarshal(uint32_t ringDim, uint32_t sizeP, std::vector<PolyImpl<NativeVector>>& ans_m_vectors) {
    for (usint j = 0; j < sizeP; j++) {
        for(usint ri = 0; ri < ringDim; ri++) {
            ans_m_vectors[j][ri] = NativeInteger(host_ans_m_vectors[j].data[ri]);
        }
    }
}

// kernel invocation
void cudaPortalForApproxModDown::callApproxSwitchCRTBasisKernel_Simple(int gpuBlocks, int gpuThreads,
                                           uint32_t ringDim, uint32_t sizeP, uint32_t sizeQ) {

    dim3 blocks = dim3(gpuBlocks, 1U, 1U); // Set the grid dimensions
    dim3 threads = dim3(gpuThreads, 1U, 1U); // Set the block dimensions

    ulong*              device_QHatInvModq          = this->getParamsData()->getDevice_QHatInvModq();
    ulong*              device_QHatInvModqPrecon    = this->getParamsData()->getDevice_QHatInvModqPrecon();
    uint128_t*          device_QHatModp             = this->getParamsData()->getDevice_QHatModp();
    uint128_t*          device_modpBarrettMu        = this->getParamsData()->getDevice_modpBarrettMu();
    m_vectors_struct*   device_m_vectors            = this->getDevice_m_vectors();
    uint128_t*          device_sum                  = this->getDevice_sum();
    m_vectors_struct*   device_ans_m_vectors        = this->getDevice_ans_m_vectors();

    void *args[] = {&ringDim, &sizeP, &sizeQ, &device_m_vectors, &device_QHatInvModq, &device_QHatInvModqPrecon, &device_QHatModp, &device_sum, &device_modpBarrettMu, &device_ans_m_vectors};
    // debugging:
    printf("Before approxSwitchCRTBasis kernel launch\n");
    printf("blocks = %d, threads = %d\n", gpuBlocks, gpuThreads);
    printMemoryInfo();
    approxSwitchCRTBasisKernelWrapper(blocks, threads, args, stream);

}

void cudaPortalForApproxModDown::destroyCUDAStream() {
    if (stream) {
        cudaError_t err = cudaStreamDestroy(stream);
        if (err != cudaSuccess) {
            printf("Error destroying stream: %s\n", cudaGetErrorString(err));
        }
        stream = nullptr; // Set pointer to nullptr after destroying
    }
}

void cudaPortalForApproxModDown::freeDeviceMemory(uint32_t sizeP, uint32_t sizeQ) {
    for (uint32_t q = 0; q < sizeQ; ++q) {
        cudaFree(device_m_vectors_data_ptr[q]);
    }
    free(device_m_vectors_data_ptr);
    cudaFree(device_m_vectors);

    cudaFree(device_sum);

    for (uint32_t p = 0; p < sizeP; ++p) {
        cudaFree(device_ans_m_vectors_data_ptr[p]);
    }
    free(device_ans_m_vectors_data_ptr);
    cudaFree(device_ans_m_vectors);
}

void cudaPortalForApproxModDown::freeHostMemory(uint32_t sizeP, uint32_t sizeQ) {
    for (uint32_t q = 0; q < sizeQ; ++q) {
        free(host_m_vectors[q].data);
    }
    free(host_m_vectors);

    for (uint32_t p = 0; p < sizeP; ++p) {
        free(host_ans_m_vectors[p].data);
    }
    free(host_ans_m_vectors);
}


}
