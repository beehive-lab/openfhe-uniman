#include "cuda-utils/approxModDown/cudaPortalForApproxModDownData.h"

#include <cassert>

namespace lbcrypto {

using PolyType = PolyImpl<NativeVector>;

cudaPortalForApproxModDownData::cudaPortalForApproxModDownData(std::shared_ptr<cudaPortalForApproxModDownParams> params_data, cudaStream_t workDataStream, int id) {
    this->id = id;
    this->paramsData = params_data;

    this->ringDim = params_data->get_RingDim();
    this->sizeP = params_data->get_sizeP();
    this->sizeQ = params_data->get_sizeQ();

    this->stream = workDataStream;

    allocateHostData();
}

cudaPortalForApproxModDownData::~cudaPortalForApproxModDownData() {
    //std::cout << "[DESTRUCTOR] Call destructor for " << this << "(cudaPortalForApproxModDown)" << std::endl;

    freeHostMemory();
    freeDeviceMemory();
}

void cudaPortalForApproxModDownData::allocateHostCTilda(uint32_t cTilda_size_x, uint32_t cTilda_size_y) {
    // cTilda
    this->cTilda_m_vectors_size_x = cTilda_size_x;
    this->cTilda_m_vectors_size_y = cTilda_size_y;

    size_t cTilda_data_size     = cTilda_m_vectors_size_x * cTilda_m_vectors_size_y * sizeof(unsigned long);
    size_t cTilda_modulus_size  = cTilda_m_vectors_size_x * sizeof(unsigned long);
    size_t cTilda_total_size    = cTilda_data_size + cTilda_modulus_size;
    host_cTilda_m_vectors       = (unsigned long*) malloc (cTilda_total_size);

    // partP_empty
    this->partP_empty_m_vectors_size_x = sizeP;
    this->partP_empty_m_vectors_size_y = ringDim;
}


void cudaPortalForApproxModDownData::allocateHostData() {
    // partP
    this->partP_m_vectors_size_x = sizeP;//partP_m_vectors.size();
    this->partP_m_vectors_size_y = ringDim;//partP_m_vectors[0].GetLength();

    size_t partP_data_size = partP_m_vectors_size_x * partP_m_vectors_size_y * sizeof(unsigned long);
    size_t partP_modulus_size = partP_m_vectors_size_x * sizeof(unsigned long);
    size_t partP_total_size = partP_data_size + partP_modulus_size;
    host_partP_m_vectors = (unsigned long*) malloc (partP_total_size);

    /*host_partP_m_vectors = (m_vectors_struct*) malloc(partP_m_vectors_size_x * sizeof(m_vectors_struct));
    for (uint32_t p = 0; p < partP_m_vectors_size_x; ++p) {
        host_partP_m_vectors[p].data = (unsigned long*) malloc(partP_m_vectors_size_y * sizeof(unsigned long));
    }*/

    // partPSwitchedToQ
    this->partPSwitchedToQ_m_vectors_size_x = sizeQ;//partPSwitchedToQ_m_vectors.size();
    this->partPSwitchedToQ_m_vectors_size_y = ringDim;//partPSwitchedToQ_m_vectors[0].GetLength();

    /*host_partPSwitchedToQ_m_vectors = (m_vectors_struct*) malloc(partPSwitchedToQ_m_vectors_size_x * sizeof(m_vectors_struct));
    for (uint32_t q = 0; q < partPSwitchedToQ_m_vectors_size_x; ++q) {
        host_partPSwitchedToQ_m_vectors[q].data = (unsigned long*) malloc(partPSwitchedToQ_m_vectors_size_y * sizeof(unsigned long));
    }*/
    size_t data_size = partPSwitchedToQ_m_vectors_size_x * partPSwitchedToQ_m_vectors_size_y * sizeof(unsigned long);
    size_t modulus_size = partPSwitchedToQ_m_vectors_size_x * sizeof(unsigned long);
    size_t total_size = data_size + modulus_size;
    host_partPSwitchedToQ_m_vectors = (unsigned long*) malloc (total_size);

    // Ensure allocation was successful
    if (host_partPSwitchedToQ_m_vectors == nullptr) {
        std::cerr << "Memory allocation failed!" << std::endl;
        exit(1);
    }

}


// Data Marshaling Functions

void cudaPortalForApproxModDownData::marshalCTilda(const std::vector<PolyImpl<NativeVector>>& cTilda_m_vectors) {
    size_t cTilda_modulus_offset = cTilda_m_vectors_size_x * cTilda_m_vectors_size_y;
    for (uint32_t p = 0; p < cTilda_m_vectors_size_x; ++p) {
        for (uint32_t rd = 0; rd < cTilda_m_vectors_size_y; ++rd) {
            host_cTilda_m_vectors[p * cTilda_m_vectors_size_y + rd] = cTilda_m_vectors[p][rd].ConvertToInt<>();
        }
        host_cTilda_m_vectors[cTilda_modulus_offset + p] = cTilda_m_vectors[p].GetModulus().ConvertToInt<>();
    }
}


void cudaPortalForApproxModDownData::marshalWorkData(const std::vector<PolyImpl<NativeVector>>& partP_m_vectors,
                                                     const std::vector<PolyImpl<NativeVector>>& partPSwitchedToQ_m_vectors) {
    // partP
    //this->partP_m_vectors_size_x = sizeP;//partP_m_vectors.size();
    //this->partP_m_vectors_size_y = ringDim;//partP_m_vectors[0].GetLength();

    assert(partP_m_vectors_size_x == partP_m_vectors.size() && "Error: partP_m_vectors size does not match sizeP");
    assert(partP_m_vectors_size_y == partP_m_vectors[0].GetLength() && "Error: partP_m_vectors_size_y does not match ringDim");

    size_t partP_modulus_offset = partP_m_vectors_size_x * partP_m_vectors_size_y;
    for (uint32_t p = 0; p < partP_m_vectors_size_x; ++p) {
        for (uint32_t rd = 0; rd < partP_m_vectors_size_y; ++rd) {
            host_partP_m_vectors[p * partP_m_vectors_size_y + rd] = partP_m_vectors[p][rd].ConvertToInt<>();
            //host_partP_m_vectors[p].data[rd] = partP_m_vectors[p][rd].ConvertToInt<>();
        }
        host_partP_m_vectors[partP_modulus_offset + p] = partP_m_vectors[p].GetModulus().ConvertToInt<>();
        //host_partP_m_vectors[p].modulus = partP_m_vectors[p].GetModulus().ConvertToInt<>();
    }

    // partPSwitchedToQ
    //this->partPSwitchedToQ_m_vectors_size_x = sizeQ;//partPSwitchedToQ_m_vectors.size();
    //this->partPSwitchedToQ_m_vectors_size_y = ringDim;//partPSwitchedToQ_m_vectors[0].GetLength();

    assert(partPSwitchedToQ_m_vectors_size_x == sizeQ && "Error: partPSwitchedToQ_m_vectors_size_x does not match sizeQ");
    assert(partPSwitchedToQ_m_vectors_size_y == ringDim && "Error: partPSwitchedToQ_m_vectors_size_y does not match ringDim");
    // only modulus for partPSwitchedToQ
    size_t data_size_bytes = partPSwitchedToQ_m_vectors_size_x * partPSwitchedToQ_m_vectors_size_y * sizeof(unsigned long);
    size_t modulus_offset = partPSwitchedToQ_m_vectors_size_x * partPSwitchedToQ_m_vectors_size_y;
    // Set the data part to 0s
    memset(host_partPSwitchedToQ_m_vectors, 0, data_size_bytes);
    // Set modulus part
    //unsigned long* modulus_part = host_partPSwitchedToQ_m_vectors + modulus_offset;
    for (uint32_t p = 0; p < partPSwitchedToQ_m_vectors_size_x; ++p) {
        // Set the modulus value at the appropriate index
        host_partPSwitchedToQ_m_vectors[modulus_offset + p] = partPSwitchedToQ_m_vectors[p].GetModulus().ConvertToInt();
        std::cout << "(marshalWorkData) host_partPSwitchedToQ_m_vectors[" << modulus_offset + p << "] = " << host_partPSwitchedToQ_m_vectors[modulus_offset + p] << std::endl;
    }

}

void cudaPortalForApproxModDownData::unmarshalWorkData(std::vector<PolyImpl<NativeVector>>& partPSwitchedToQ_m_vectors) {
    // make sure stream has finished all queued tasks before touching the results from host
    CUDA_CHECK(cudaStreamSynchronize(stream));
    //std::cout << "==> UNMARSHAL START" << std::endl;

    for (usint x = 0; x < partPSwitchedToQ_m_vectors_size_x; x++) {
        for(usint y = 0; y < partPSwitchedToQ_m_vectors_size_y; y++) {
            partPSwitchedToQ_m_vectors[x][y] = NativeInteger(host_partPSwitchedToQ_m_vectors[x * partPSwitchedToQ_m_vectors_size_y + y]);
        }
    }
}

// Data Transfer Functions

void cudaPortalForApproxModDownData::copyInCTilda() {
    // cTilda
    size_t cTilda_data_size = cTilda_m_vectors_size_x * cTilda_m_vectors_size_y * sizeof(unsigned long);
    size_t cTilda_modulus_size = cTilda_m_vectors_size_x * sizeof(unsigned long);
    CUDA_CHECK(cudaMallocAsync((void**)&device_cTilda_m_vectors, cTilda_data_size + cTilda_modulus_size, stream));
    CUDA_CHECK(cudaMemcpyAsync(device_cTilda_m_vectors, host_cTilda_m_vectors, cTilda_data_size + cTilda_modulus_size, cudaMemcpyHostToDevice, stream));
}

void cudaPortalForApproxModDownData::copyInPartP_Empty() {
    // partP_empty
    // allocation only
    size_t partP_empty_data_size = partP_empty_m_vectors_size_x * partP_empty_m_vectors_size_y * sizeof(unsigned long);
    size_t partP_empty_modulus_size = partP_empty_m_vectors_size_x * sizeof(unsigned long);
    CUDA_CHECK(cudaMallocAsync((void**)&device_partP_empty_m_vectors, partP_empty_data_size + partP_empty_modulus_size, stream));
}

void cudaPortalForApproxModDownData::copyInWorkData() {

    // partP
    size_t partP_data_size = partP_m_vectors_size_x * partP_m_vectors_size_y * sizeof(unsigned long);
    size_t partP_modulus_size = partP_m_vectors_size_x * sizeof(unsigned long);
    CUDA_CHECK(cudaMallocAsync((void**)&device_partP_m_vectors, partP_data_size + partP_modulus_size, stream));
    CUDA_CHECK(cudaMemcpyAsync(device_partP_m_vectors, host_partP_m_vectors, partP_data_size + partP_modulus_size, cudaMemcpyHostToDevice, stream));
    /*CUDA_CHECK(cudaMallocAsync((void**)&device_partP_m_vectors, partP_m_vectors_size_x * sizeof(m_vectors_struct), stream));
    CUDA_CHECK(cudaMemcpyAsync(device_partP_m_vectors, host_partP_m_vectors, partP_m_vectors_size_x * sizeof(m_vectors_struct), cudaMemcpyHostToDevice, stream));

    this->device_partP_m_vectors_data_ptr = (unsigned long**)malloc(partP_m_vectors_size_x * sizeof(unsigned long*));
    for (uint32_t i = 0; i < partP_m_vectors_size_x; ++i) {
        CUDA_CHECK(cudaMallocAsync((void**)&(device_partP_m_vectors_data_ptr[i]), partP_m_vectors_size_y * sizeof(unsigned long), stream));
        CUDA_CHECK(cudaMemcpyAsync(&(device_partP_m_vectors[i].data), &(device_partP_m_vectors_data_ptr[i]), sizeof(unsigned long*), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(device_partP_m_vectors_data_ptr[i], host_partP_m_vectors[i].data, partP_m_vectors_size_y * sizeof(unsigned long), cudaMemcpyHostToDevice, stream));
    }*/

    // sum
    CUDA_CHECK(cudaMallocAsync((void**)&device_sum, sizeQ * ringDim * sizeof(uint128_t), stream));
    CUDA_CHECK(cudaMemsetAsync(device_sum, 0, sizeQ * ringDim * sizeof(uint128_t), stream));

    // partPSwitchedToQ  TODO: copy in only modulus part
    size_t data_size = partPSwitchedToQ_m_vectors_size_x * partPSwitchedToQ_m_vectors_size_y * sizeof(unsigned long);
    size_t modulus_size = partPSwitchedToQ_m_vectors_size_x * sizeof(unsigned long);
    CUDA_CHECK(cudaMallocAsync((void**)&device_partPSwitchedToQ_m_vectors, data_size + modulus_size, stream));
    CUDA_CHECK(cudaMemcpyAsync(device_partPSwitchedToQ_m_vectors, host_partPSwitchedToQ_m_vectors, data_size + modulus_size, cudaMemcpyHostToDevice, stream));
    /*CUDA_CHECK(cudaMallocAsync((void**)&device_partPSwitchedToQ_m_vectors, partPSwitchedToQ_m_vectors_size_x * sizeof(m_vectors_struct), stream));
    CUDA_CHECK(cudaMemcpyAsync(device_partPSwitchedToQ_m_vectors, host_partPSwitchedToQ_m_vectors, partPSwitchedToQ_m_vectors_size_x * sizeof(m_vectors_struct), cudaMemcpyHostToDevice, stream));
    this->device_partPSwitchedToQ_m_vectors_data_ptr = (unsigned long**)malloc(partPSwitchedToQ_m_vectors_size_x * sizeof(unsigned long*));
    for (uint32_t p = 0; p < partPSwitchedToQ_m_vectors_size_x; ++p) {
        CUDA_CHECK(cudaMallocAsync((void**)&(device_partPSwitchedToQ_m_vectors_data_ptr[p]), partPSwitchedToQ_m_vectors_size_y * sizeof(unsigned long), stream));
        CUDA_CHECK(cudaMemcpyAsync(&(device_partPSwitchedToQ_m_vectors[p].data), &(device_partPSwitchedToQ_m_vectors_data_ptr[p]), sizeof(unsigned long*), cudaMemcpyHostToDevice, stream));
    }*/
}

void cudaPortalForApproxModDownData::copyOutResult() {
    //printf("(copyOut): SizeP= %d, sizeQ= %d, copy out size = %d\n", sizeP, sizeQ, partPSwitchedToQ_m_vectors_size_x);

    size_t data_size = partPSwitchedToQ_m_vectors_size_x * partPSwitchedToQ_m_vectors_size_y * sizeof(unsigned long);
    CUDA_CHECK(cudaMemcpyAsync(host_partPSwitchedToQ_m_vectors, device_partPSwitchedToQ_m_vectors, data_size, cudaMemcpyDeviceToHost, stream));
    /*for(uint32_t p = 0; p < partPSwitchedToQ_m_vectors_size_x; p++) {
        std::cout << "==> COPY OUT (" << whoAmI() << ")" << std::endl;
        //CUDA_CHECK(cudaMemcpyAsync(host_partPSwitchedToQ_m_vectors[p].data, device_partPSwitchedToQ_m_vectors_data_ptr[p], partPSwitchedToQ_m_vectors_size_y * sizeof(unsigned long), cudaMemcpyDeviceToHost, stream));
    }*/

    //std::cout << "==> COPY OUT FINISHED" << std::endl;
}

// Kernel Invocation Function

void cudaPortalForApproxModDownData::invokePartPFillKernel(int gpuBlocks, int gpuThreads) {
    dim3 blocks = dim3(gpuBlocks, 1U, 1U); // Set the grid dimensions
    dim3 threads = dim3(gpuThreads, 1U, 1U); // Set the block dimensions

    void *args[] = {
        // scalar values
        &sizeQP, &sizeQ,
        // work data along with their column size
        &device_cTilda_m_vectors, &cTilda_m_vectors_size_x, &cTilda_m_vectors_size_y,
        &device_partP_empty_m_vectors, &partP_empty_m_vectors_size_x, &partP_empty_m_vectors_size_y
        };

    fillPartPKernelWrapper(blocks, threads, args, stream);
}

void cudaPortalForApproxModDownData::invokeKernelOfApproxModDown(int gpuBlocks, int gpuThreads) {

    dim3 blocks = dim3(gpuBlocks, 1U, 1U); // Set the grid dimensions
    dim3 threads = dim3(gpuThreads, 1U, 1U); // Set the block dimensions

    ulong*              device_tInvModp             = paramsData->get_device_tInvModp();
    ulong*              device_tInvModpPrecon       = paramsData->get_device_tInvModpPrecon();
    ulong*              device_QHatInvModq          = paramsData->get_device_PHatInvModp();
    ulong*              device_QHatInvModqPrecon    = paramsData->get_device_PHatInvModpPrecon();
    uint128_t*          device_QHatModp             = paramsData->get_device_PHatModq();
    uint32_t            QHatModP_sizeY              = paramsData->get_PHatModq_sizeY();
    uint128_t*          device_modpBarrettMu        = paramsData->get_device_modqBarrettMu();

    void *args[] = {
        // scalar values
        &ringDim, &sizeQP, &sizeP, &sizeQ,
        // work data along with their column size
        &device_partP_empty_m_vectors, &partP_empty_m_vectors_size_x, &partP_empty_m_vectors_size_y,
        &device_sum,
        &device_partPSwitchedToQ_m_vectors, &partPSwitchedToQ_m_vectors_size_y,
        // params data along with their column size (where applicable)
        &device_tInvModp,
        &device_tInvModpPrecon,
        &device_QHatInvModq,
        &device_QHatInvModqPrecon,
        &device_QHatModp, &QHatModP_sizeY,
        &device_modpBarrettMu,
    };

    approxModDownKernelWrapper(blocks, threads, args, stream);
}

// Resources Deallocation - Error Handling - Misc Functions

void cudaPortalForApproxModDownData::handleFreeError(const std::string& operation, void* ptr) {
    if (ptr == nullptr) {
        throw std::runtime_error("Memory error during " + operation + ": null pointer passed for freeing.");
    } else {
        free(ptr);  // Actual free operation happens here
        ptr = nullptr;  // Reset to nullptr after free
    }
}

void cudaPortalForApproxModDownData::freeHostMemory() {

    /*for (uint32_t q = 0; q < partP_m_vectors_size_x; ++q) {
        //handleFreeError("host_partP_m_vectors[" + std::to_string(q) + "].data", host_partP_m_vectors[q].data);
        safeFree(host_partP_m_vectors[q].data);
    }*/
    //handleFreeError("host_partP_m_vectors", host_partP_m_vectors);
    safeFree(host_partP_m_vectors);

    /*for (uint32_t q = 0; q < partPSwitchedToQ_m_vectors_size_x; ++q) {
        safeFree(host_partPSwitchedToQ_m_vectors[q].data);
    }*/
    safeFree(host_partPSwitchedToQ_m_vectors);
}

void cudaPortalForApproxModDownData::freeDeviceMemory() {
    printf("freeDeviceMemory\n");
    CUDA_CHECK(cudaFreeAsync(device_partP_m_vectors, stream));
    /*// Free the 'data' array in each partP vector on the device
    for (uint32_t i = 0; i < partP_m_vectors_size_x; ++i) {
        if (device_partP_m_vectors[i].data != nullptr) {
            // Synchronize the stream before freeing
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // Free device memory using cudaFreeAsync
            CUDA_CHECK(cudaFreeAsync(device_partP_m_vectors[i].data, stream));
            device_partP_m_vectors[i].data = nullptr;
        }
    }

    printf("3\n");
    if (device_partP_m_vectors_data_ptr) {
        free(device_partP_m_vectors_data_ptr);
        device_partP_m_vectors_data_ptr = nullptr;
    }
    if (device_partP_m_vectors) {
        CUDA_CHECK(cudaFreeAsync(device_partP_m_vectors, stream));
        device_partP_m_vectors = nullptr;
    }*/

    // Free sum device memory
    if (device_sum) {
        CUDA_CHECK(cudaFreeAsync(device_sum, stream));
        device_sum = nullptr;
    }

    // Free partPSwitchedToQ device memory
    CUDA_CHECK(cudaFreeAsync(device_partPSwitchedToQ_m_vectors, stream));
    /*for (uint32_t i = 0; i < partPSwitchedToQ_m_vectors_size_x; ++i) {
        if (device_partPSwitchedToQ_m_vectors[i].data) {
            CUDA_CHECK(cudaFreeAsync(device_partPSwitchedToQ_m_vectors[i].data, stream));
            device_partPSwitchedToQ_m_vectors[i].data = nullptr;
        }
    }
    safeFree(device_partPSwitchedToQ_m_vectors_data_ptr);
    if (device_partPSwitchedToQ_m_vectors) {
        CUDA_CHECK(cudaFreeAsync(device_partPSwitchedToQ_m_vectors, stream));
        device_partPSwitchedToQ_m_vectors = nullptr;
    }*/
}


void cudaPortalForApproxModDownData::handleCUDAError(const std::string& operation, cudaError_t err) {
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error during " + operation + ": " + std::string(cudaGetErrorString(err)));
    }
}


}
