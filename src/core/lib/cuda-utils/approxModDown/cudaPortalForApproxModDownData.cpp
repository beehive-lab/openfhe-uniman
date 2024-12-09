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
void cudaPortalForApproxModDownData::copyInCTildaQ_Batch(uint32_t ptrOffset, cudaStream_t stream) {
    auto device_m_vectors_ptr = device_cTildaQ_m_vectors + ptrOffset;
    auto host_m_vectors_ptr = host_cTildaQ_m_vectors + ptrOffset;
    size_t size = cTildaQ_m_vectors_size_y * sizeof(ulong);
    CUDA_CHECK(cudaMemcpyAsync(device_m_vectors_ptr, host_m_vectors_ptr, size, cudaMemcpyHostToDevice, stream));
}

void cudaPortalForApproxModDownData::copyInPartP_Batch(uint32_t ptrOffset, cudaStream_t stream) {
    size_t batchSize = partP_empty_m_vectors_size_y * sizeof(unsigned long);
    auto device_m_vectors_ptr   = device_partP_empty_m_vectors         + ptrOffset;
    auto host_m_vectors_ptr     = host_cTilda_m_vectors           + ptrOffset;
    CUDA_CHECK(cudaMemcpyAsync(device_m_vectors_ptr, host_m_vectors_ptr, batchSize, cudaMemcpyHostToDevice, stream));
}


void cudaPortalForApproxModDownData::copyOutResultBatch(uint32_t ptrOffset, cudaStream_t stream) {
    size_t batchSize = ans_m_vectors_size_y * sizeof(unsigned long);
    auto device_m_vectors_ptr   = device_ans_m_vectors         + ptrOffset;
    auto host_m_vectors_ptr     = host_ans_m_vectors           + ptrOffset;
    CUDA_CHECK(cudaMemcpyAsync(host_m_vectors_ptr, device_m_vectors_ptr, batchSize, cudaMemcpyDeviceToHost, stream));
}


// Kernel Invocation Functions

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