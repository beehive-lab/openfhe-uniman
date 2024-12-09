#include "cuda-utils/approxModDown/cudaPortalForApproxModDownData.h"

#include <cassert>

namespace lbcrypto {

using PolyType = PolyImpl<NativeVector>;

cudaPortalForApproxModDownData::cudaPortalForApproxModDownData(std::shared_ptr<cudaPortalForApproxModDownParams> params_data, cudaStream_t workDataStream, cudaStream_t* pipelineStreams, cudaEvent_t workDataEvent, cudaEvent_t* pipelineEvents, int id) {
    this->id = id;
    this->paramsData = params_data;

    this->ringDim = params_data->get_RingDim();
    this->sizeP = params_data->get_sizeP();
    this->sizeQ = params_data->get_sizeQ();

    // partP: input data dimensions
    // cTilda
    this->cTilda_m_vectors_size_x = sizeP;
    this->cTilda_m_vectors_size_y = ringDim;
    this->partP_empty_m_vectors_size_x = sizeP;
    this->partP_empty_m_vectors_size_y = ringDim;

    // partPSwitchedToQ: work data dimensions
    this->partPSwitchedToQ_m_vectors_size_x = sizeQ;//partPSwitchedToQ_m_vectors.size();
    this->partPSwitchedToQ_m_vectors_size_y = ringDim;//partPSwitchedToQ_m_vectors[0].GetLength();

    this->cTildaQ_m_vectors_size_x = sizeQ;
    this->cTildaQ_m_vectors_size_y = ringDim;

    // ans: output data dimensions
    this->ans_m_vectors_size_x = sizeQ;
    this->ans_m_vectors_size_y = ringDim;

    this->stream = workDataStream;
    this->pipelineStreams = pipelineStreams;

    this->event = workDataEvent;
    this->pipelineEvents = pipelineEvents;

    allocateHostData();
}

cudaPortalForApproxModDownData::~cudaPortalForApproxModDownData() {
    //std::cout << "[DESTRUCTOR] Call destructor for " << this << "(cudaPortalForApproxModDown)" << std::endl;

    freeHostMemory();
    freeDeviceMemory();
}

// TODO: delete
void cudaPortalForApproxModDownData::allocateHostCTilda(uint32_t cTilda_size_x, uint32_t cTilda_size_y) {
    // cTilda
    this->cTilda_m_vectors_size_x = cTilda_size_x;
    this->cTilda_m_vectors_size_y = cTilda_size_y;

    size_t cTilda_data_size     = cTilda_m_vectors_size_x * cTilda_m_vectors_size_y * sizeof(unsigned long);
    size_t cTilda_modulus_size  = cTilda_m_vectors_size_x * sizeof(unsigned long);
    size_t cTilda_total_size    = cTilda_data_size + cTilda_modulus_size;
    //host_cTilda_m_vectors       = (unsigned long*) malloc (cTilda_total_size);
    cudaHostAlloc((void**)&host_cTilda_m_vectors, cTilda_total_size, cudaHostAllocDefault);

    // partP_empty
    this->partP_empty_m_vectors_size_x = sizeP;
    this->partP_empty_m_vectors_size_y = ringDim;
}


void cudaPortalForApproxModDownData::allocateHostData() {

    // host input batched data
    size_t input_data_size     = partP_empty_m_vectors_size_x * partP_empty_m_vectors_size_y * sizeof(unsigned long);
    cudaHostAlloc((void**)&host_cTilda_m_vectors, input_data_size, cudaHostAllocDefault);

    // device input batched data
    CUDA_CHECK(cudaMallocAsync((void**)&device_partP_empty_m_vectors, input_data_size, stream));

    // device work (intermediate) data
    size_t work_data_size = partPSwitchedToQ_m_vectors_size_x * partPSwitchedToQ_m_vectors_size_y * sizeof(unsigned long);
    CUDA_CHECK(cudaMallocAsync((void**)&device_partPSwitchedToQ_m_vectors, work_data_size, stream));

    // sum (intermediate)
    CUDA_CHECK(cudaMallocAsync((void**)&device_sum, sizeQ * ringDim * sizeof(uint128_t), stream));
    CUDA_CHECK(cudaMemsetAsync(device_sum, 0, sizeQ * ringDim * sizeof(uint128_t), stream));

    // cTildaQ
    size_t cTildaQ_data_size = cTildaQ_m_vectors_size_x * cTildaQ_m_vectors_size_y * sizeof(ulong);
    cudaHostAlloc((void**)&host_cTildaQ_m_vectors, cTildaQ_data_size, cudaHostAllocDefault);
    CUDA_CHECK(cudaMallocAsync((void**)&device_cTildaQ_m_vectors, cTildaQ_data_size, stream));

    // ans: output
    size_t ans_size = ans_m_vectors_size_x * ans_m_vectors_size_y * sizeof(ulong);
    cudaMallocHost((void**)&host_ans_m_vectors, ans_size, cudaHostAllocDefault);
    CUDA_CHECK(cudaMallocAsync((void**)&device_ans_m_vectors, ans_size, stream));
}


// Data Marshaling Functions
// TODO: delete
void cudaPortalForApproxModDownData::marshalCTilda(const std::vector<PolyImpl<NativeVector>>& cTilda_m_vectors) {
    size_t cTilda_modulus_offset = cTilda_m_vectors_size_x * cTilda_m_vectors_size_y;
    for (uint32_t p = 0; p < cTilda_m_vectors_size_x; ++p) {
        for (uint32_t rd = 0; rd < cTilda_m_vectors_size_y; ++rd) {
            host_cTilda_m_vectors[p * cTilda_m_vectors_size_y + rd] = cTilda_m_vectors[p][rd].ConvertToInt<>();
        }
        host_cTilda_m_vectors[cTilda_modulus_offset + p] = cTilda_m_vectors[p].GetModulus().ConvertToInt<>();
    }
}

// marshal only the ctilda portion of interest (from sizeQ to sizeQP, so the final size equals to sizeP)
void cudaPortalForApproxModDownData::marshalCTildaBatch(const std::vector<PolyImpl<NativeVector>>& cTilda_m_vectors, uint32_t partP_index, uint32_t cTilda_index) {
    for (uint32_t rd = 0; rd < cTilda_m_vectors_size_y; ++rd) {
        host_cTilda_m_vectors[partP_index * cTilda_m_vectors_size_y + rd] = cTilda_m_vectors[cTilda_index][rd].ConvertToInt<>();
    }
}

void cudaPortalForApproxModDownData::marshalCTildaQBatch(const std::vector<PolyImpl<NativeVector>>& cTilda_m_vectors, uint32_t index) {
    for (uint32_t rd = 0; rd < cTilda_m_vectors_size_y; ++rd) {
        host_cTildaQ_m_vectors[index * cTildaQ_m_vectors_size_y + rd] = cTilda_m_vectors[index][rd].ConvertToInt<>();
    }
}

// TODO: delete
void cudaPortalForApproxModDownData::marshalWorkData(const std::vector<PolyImpl<NativeVector>>& partPSwitchedToQ_m_vectors) {

    // partPSwitchedToQ
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
    }

}

// TODO: delete
void cudaPortalForApproxModDownData::unmarshalWorkData(std::vector<PolyImpl<NativeVector>>& ans_m_vectors) {
    // make sure stream has finished all queued tasks before touching the results from host
    CUDA_CHECK(cudaStreamSynchronize(stream));
    //std::cout << "==> UNMARSHAL START" << std::endl;

    for (usint x = 0; x < ans_m_vectors_size_x; x++) {
        for(usint y = 0; y < ans_m_vectors_size_y; y++) {
            ans_m_vectors[x][y] = NativeInteger(host_ans_m_vectors[x * ans_m_vectors_size_y + y]);
        }
    }
}

void unmarshalWorkDataBatch(void *void_arg) {
    //printf("Unmarshalling\n");
    UnmarshalWorkDataBatchParams* args = (UnmarshalWorkDataBatchParams*)void_arg;
    ulong* host_ans_m_vectors = args->host_ans_m_vectors;
    std::vector<PolyImpl<NativeVector>>* ans_m_vectors = args->ans_m_vectors;
    uint32_t i = args->i;
    uint32_t ptrOffset = args->ptrOffset;
    uint32_t size = args->size;
    ulong* host_ans_ptr = host_ans_m_vectors + ptrOffset;
    for(usint y = 0; y < size; y++) {
        (*ans_m_vectors)[i][y] = NativeInteger(host_ans_ptr[y]);
    }
    //free(void_arg);
}

void cudaPortalForApproxModDownData::unmarshalWorkDataBatchWrapper(std::vector<PolyImpl<NativeVector>>& ans_m_vectors, uint32_t i, uint32_t ptr_offset, cudaStream_t pipelineStream) {
    UnmarshalWorkDataBatchParams* args = new UnmarshalWorkDataBatchParams(
        &ans_m_vectors, host_ans_m_vectors, ans_m_vectors_size_y,i, ptr_offset);
    cudaLaunchHostFunc(pipelineStream, unmarshalWorkDataBatch, args);

    //cudaStreamSynchronize(pipelineStream);
    //delete args;
}

// Data Transfer Functions
// TODO: delete
void cudaPortalForApproxModDownData::copyInCTilda() {
    // cTilda
    size_t cTilda_data_size = cTilda_m_vectors_size_x * cTilda_m_vectors_size_y * sizeof(unsigned long);
    size_t cTilda_modulus_size = cTilda_m_vectors_size_x * sizeof(unsigned long);
    CUDA_CHECK(cudaMallocAsync((void**)&device_cTilda_m_vectors, cTilda_data_size + cTilda_modulus_size, stream));
    CUDA_CHECK(cudaMemcpyAsync(device_cTilda_m_vectors, host_cTilda_m_vectors, cTilda_data_size + cTilda_modulus_size, cudaMemcpyHostToDevice, stream));
}

void cudaPortalForApproxModDownData::copyInCTildaQ_Batch(uint32_t ptrOffset, cudaStream_t stream) {
    auto device_m_vectors_ptr = device_cTildaQ_m_vectors + ptrOffset;
    auto host_m_vectors_ptr = host_cTildaQ_m_vectors + ptrOffset;
    size_t size = cTildaQ_m_vectors_size_y * sizeof(ulong);
    CUDA_CHECK(cudaMemcpyAsync(device_m_vectors_ptr, host_m_vectors_ptr, size, cudaMemcpyHostToDevice, stream));
}

// TODO: delete
void cudaPortalForApproxModDownData::copyInPartP_Empty() {
    // partP_empty
    // allocation only
    size_t partP_empty_data_size = partP_empty_m_vectors_size_x * partP_empty_m_vectors_size_y * sizeof(unsigned long);
    size_t partP_empty_modulus_size = partP_empty_m_vectors_size_x * sizeof(unsigned long);
    CUDA_CHECK(cudaMallocAsync((void**)&device_partP_empty_m_vectors, partP_empty_data_size + partP_empty_modulus_size, stream));
}

void cudaPortalForApproxModDownData::copyInPartP_Batch(uint32_t ptrOffset, cudaStream_t stream) {
    size_t batchSize = partP_empty_m_vectors_size_y * sizeof(unsigned long);
    auto device_m_vectors_ptr   = device_partP_empty_m_vectors         + ptrOffset;
    auto host_m_vectors_ptr     = host_cTilda_m_vectors           + ptrOffset;
    CUDA_CHECK(cudaMemcpyAsync(device_m_vectors_ptr, host_m_vectors_ptr, batchSize, cudaMemcpyHostToDevice, stream));
}

// TODO: delete
void cudaPortalForApproxModDownData::copyInWorkData() {

    // sum
    CUDA_CHECK(cudaMallocAsync((void**)&device_sum, sizeQ * ringDim * sizeof(uint128_t), stream));
    CUDA_CHECK(cudaMemsetAsync(device_sum, 0, sizeQ * ringDim * sizeof(uint128_t), stream));

    size_t data_size = partPSwitchedToQ_m_vectors_size_x * partPSwitchedToQ_m_vectors_size_y * sizeof(unsigned long);
    size_t modulus_size = partPSwitchedToQ_m_vectors_size_x * sizeof(unsigned long);
    CUDA_CHECK(cudaMallocAsync((void**)&device_partPSwitchedToQ_m_vectors, data_size + modulus_size, stream));
    // Copy *only* modulus portion to the appropriate offset in the allocated memory
    unsigned long* device_modulus = device_partPSwitchedToQ_m_vectors + (partPSwitchedToQ_m_vectors_size_x * partPSwitchedToQ_m_vectors_size_y);
    unsigned long* host_modulus = host_partPSwitchedToQ_m_vectors + (partPSwitchedToQ_m_vectors_size_x * partPSwitchedToQ_m_vectors_size_y);
    CUDA_CHECK(cudaMemcpyAsync(device_modulus, host_modulus, modulus_size, cudaMemcpyHostToDevice, stream));
    /*CUDA_CHECK(cudaMallocAsync((void**)&device_partPSwitchedToQ_m_vectors, partPSwitchedToQ_m_vectors_size_x * sizeof(m_vectors_struct), stream));
    CUDA_CHECK(cudaMemcpyAsync(device_partPSwitchedToQ_m_vectors, host_partPSwitchedToQ_m_vectors, partPSwitchedToQ_m_vectors_size_x * sizeof(m_vectors_struct), cudaMemcpyHostToDevice, stream));
    this->device_partPSwitchedToQ_m_vectors_data_ptr = (unsigned long**)malloc(partPSwitchedToQ_m_vectors_size_x * sizeof(unsigned long*));
    for (uint32_t p = 0; p < partPSwitchedToQ_m_vectors_size_x; ++p) {
        CUDA_CHECK(cudaMallocAsync((void**)&(device_partPSwitchedToQ_m_vectors_data_ptr[p]), partPSwitchedToQ_m_vectors_size_y * sizeof(unsigned long), stream));
        CUDA_CHECK(cudaMemcpyAsync(&(device_partPSwitchedToQ_m_vectors[p].data), &(device_partPSwitchedToQ_m_vectors_data_ptr[p]), sizeof(unsigned long*), cudaMemcpyHostToDevice, stream));
    }*/
    // ans - only for data, no need for modulus
    size_t ans_data_size = ans_m_vectors_size_x * ans_m_vectors_size_y * sizeof(unsigned long);
    CUDA_CHECK(cudaMallocAsync((void**)&device_ans_m_vectors, ans_data_size, stream));
}

// TODO: delete
void cudaPortalForApproxModDownData::copyOutResult() {
    size_t ans_data_size = ans_m_vectors_size_x * ans_m_vectors_size_y * sizeof(unsigned long);
    //host_ans_m_vectors = (unsigned long*) malloc(ans_data_size);
    cudaMallocHost((void**)&host_ans_m_vectors, ans_data_size, cudaHostAllocDefault);

    CUDA_CHECK(cudaMemcpyAsync(host_ans_m_vectors, device_ans_m_vectors, ans_data_size, cudaMemcpyDeviceToHost, stream));
}

void cudaPortalForApproxModDownData::copyOutResultBatch(uint32_t ptrOffset, cudaStream_t stream) {
    size_t batchSize = ans_m_vectors_size_y * sizeof(unsigned long);
    auto device_m_vectors_ptr   = device_ans_m_vectors         + ptrOffset;
    auto host_m_vectors_ptr     = host_ans_m_vectors           + ptrOffset;
    CUDA_CHECK(cudaMemcpyAsync(host_m_vectors_ptr, device_m_vectors_ptr, batchSize, cudaMemcpyDeviceToHost, stream));
}


// Kernel Invocation Function
// TODO: delete
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
// TODO: delete
void cudaPortalForApproxModDownData::invokeKernelOfApproxModDown(int gpuBlocks, int gpuThreads) {
    //cudaDeviceSynchronize();
    dim3 blocks = dim3(gpuBlocks, 1U, 1U); // Set the grid dimensions
    dim3 threads = dim3(gpuThreads, 1U, 1U); // Set the block dimensions

    ulong*              device_tInvModp             = paramsData->get_device_tInvModp();
    ulong*              device_tInvModpPrecon       = paramsData->get_device_tInvModpPrecon();
    ulong*              device_QHatInvModq          = paramsData->get_device_PHatInvModp();
    ulong*              device_QHatInvModqPrecon    = paramsData->get_device_PHatInvModpPrecon();
    uint128_t*          device_QHatModp             = paramsData->get_device_PHatModq();
    uint32_t            QHatModP_sizeY              = paramsData->get_PHatModq_sizeY();
    uint128_t*          device_modpBarrettMu        = paramsData->get_device_modqBarrettMu();
    uint32_t            t                           = paramsData->get_t();
    ulong*              device_tModqPrecon          = paramsData->get_device_tModqPrecon();

    void *args[] = {
        // scalar values
        &ringDim, &sizeQP, &sizeP, &sizeQ,
        // work data along with their column size
        &device_partP_empty_m_vectors, &partP_empty_m_vectors_size_x, &partP_empty_m_vectors_size_y,
        &device_sum,
        &device_partPSwitchedToQ_m_vectors, &partPSwitchedToQ_m_vectors_size_x, &partPSwitchedToQ_m_vectors_size_y,
        // params data along with their column size (where applicable)
        &device_tInvModp,
        &device_tInvModpPrecon,
        &device_QHatInvModq,
        &device_QHatInvModqPrecon,
        &device_QHatModp, &QHatModP_sizeY,
        &device_modpBarrettMu,
        //
        &t,
        &device_tModqPrecon
    };

    approxModDownKernelWrapper(blocks, threads, args, stream);
    //cudaStreamSynchronize(stream);
}

void cudaPortalForApproxModDownData::invokeKernelOfApproxModDownBatchPt1(int gpuBlocks, int gpuThreads,
                                                                         ulong modulus, ulong tInvModp, ulong tInvModpPrecon,
                                                                         uint32_t ptr_offset, cudaStream_t stream) {
    dim3 blocks = dim3(gpuBlocks, 1U, 1U); // Set the grid dimensions
    dim3 threads = dim3(gpuThreads, 1U, 1U); // Set the block dimensions

    auto device_partP_ptr   = device_partP_empty_m_vectors + ptr_offset;

    void *args[] = {&ringDim,
                    &device_partP_ptr, &modulus,
                    &tInvModp, &tInvModpPrecon};

    approxModDownBatchPt1KernelWrapper(blocks, threads, args, stream);
}

void cudaPortalForApproxModDownData::invokeKernelOfApproxSwitchCRTBasisPt1Batch(int gpuBlocks, int gpuThreads,
                                                                                uint32_t i, ulong modulus, uint32_t ptr_offset,
                                                                                ulong QHatInvModq, ulong QHatInvModqPrecon,
                                                                                cudaStream_t stream) {
    dim3 blocks = dim3(gpuBlocks, 1U, 1U); // Set the grid dimensions
    dim3 threads = dim3(gpuThreads, 1U, 1U); // Set the block dimensions

    auto device_m_vectors_ptr   = device_partP_empty_m_vectors + ptr_offset;

    uint128_t* QHatModp = paramsData->get_device_PHatModq(i);

    void *args[] = {&sizeQ, &i,
                    &device_m_vectors_ptr, &modulus,
                    &QHatInvModq, &QHatInvModqPrecon,
                    &QHatModp,
                    &device_sum};

    approxSwitchCRTBasisPt1BatchKernelWrapper(blocks, threads, args, stream);
}

void cudaPortalForApproxModDownData::invokeKernelOfApproxSwitchCRTBasisPt2Batch(int gpuBlocks, int gpuThreads,
                                                                                uint32_t i, ulong ans_modulus, uint32_t ptr_offset,
                                                                                uint128_t modpBarrettMu, uint32_t t, ulong tModqPrecon,
                                                                                cudaStream_t stream) {
    dim3 blocks = dim3(gpuBlocks, 1U, 1U); // Set the grid dimensions
    dim3 threads = dim3(gpuThreads, 1U, 1U); // Set the block dimensions

    auto device_partPSwitchedQ_m_vectors_ptr   = device_partPSwitchedToQ_m_vectors + ptr_offset;
    void* args[] = {&sizeQ, &i,
                    &device_sum,
                    &device_partPSwitchedQ_m_vectors_ptr, &ans_modulus,
                    &modpBarrettMu, &t, &tModqPrecon};

    approxSwitchCRTBasisPt2BatchKernelWrapper(blocks, threads, args, stream);
}

// TODO: delete
void cudaPortalForApproxModDownData::invokeAnsFillKernel(int gpuBlocks, int gpuThreads) {
    //cudaDeviceSynchronize();
    dim3 blocks = dim3(gpuBlocks, 1U, 1U); // Set the grid dimensions
    dim3 threads = dim3(gpuThreads, 1U, 1U); // Set the block dimensions

    ulong* device_pInvModq = paramsData->get_device_PInvModq();
    ulong* device_pInvModqPrecon = paramsData->get_device_PInvModqPrecon();

    void *args[] = {
        &sizeQ,
        &device_cTilda_m_vectors, &cTilda_m_vectors_size_x, &cTilda_m_vectors_size_y,
        &device_partPSwitchedToQ_m_vectors, &partPSwitchedToQ_m_vectors_size_x, &partPSwitchedToQ_m_vectors_size_y,
        &device_ans_m_vectors, &ans_m_vectors_size_x, &ans_m_vectors_size_y,
        &device_pInvModq,
        &device_pInvModqPrecon
        };

    ansFillKernelWrapper(blocks, threads, args, stream);
}

void cudaPortalForApproxModDownData::invokeAnsFillBatchKernel(int gpuBlocks, int gpuThreads,
                                                              uint32_t i, ulong modulus, uint32_t ptr_offset,
                                                              ulong pInvModq, ulong pInvModqPrecon,
                                                              cudaStream_t stream) {
    dim3 blocks = dim3(gpuBlocks, 1U, 1U); // Set the grid dimensions
    dim3 threads = dim3(gpuThreads, 1U, 1U); // Set the block dimensions

    auto device_cTildaQ_m_vectors_ptr   = device_cTildaQ_m_vectors + ptr_offset;
    auto device_partPSwitchedQ_m_vectors_ptr = device_partPSwitchedToQ_m_vectors + ptr_offset;
    auto device_ans_m_vectors_ptr   = device_ans_m_vectors + ptr_offset;

    void* args[] = {&i, &device_cTildaQ_m_vectors_ptr, &modulus,
                    &device_partPSwitchedQ_m_vectors_ptr,
                    &device_ans_m_vectors_ptr,
                    &pInvModq,
                    &pInvModqPrecon};
    ansFillBatchKernelWrapper(blocks, threads, args, stream);
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

    safeCudaFreeHost(host_cTilda_m_vectors);
    safeCudaFreeHost(host_cTildaQ_m_vectors);
    safeCudaFreeHost(host_ans_m_vectors);
}

void cudaPortalForApproxModDownData::freeDeviceMemory() {

    CUDA_CHECK(cudaFreeAsync(device_partP_empty_m_vectors, stream));
    CUDA_CHECK(cudaFreeAsync(device_partPSwitchedToQ_m_vectors, stream));
    CUDA_CHECK(cudaFreeAsync(device_sum, stream));
    CUDA_CHECK(cudaFreeAsync(device_cTildaQ_m_vectors, stream));
    CUDA_CHECK(cudaFreeAsync(device_ans_m_vectors, stream));
    device_partP_empty_m_vectors = nullptr;
    device_partPSwitchedToQ_m_vectors = nullptr;
    device_sum = nullptr;
    device_cTildaQ_m_vectors = nullptr;
    device_ans_m_vectors = nullptr;
}

void cudaPortalForApproxModDownData::handleCUDAError(const std::string& operation, cudaError_t err) {
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error during " + operation + ": " + std::string(cudaGetErrorString(err)));
    }
}

}