#include "cuda-utils/approxModDown/cudaPortalForApproxModDownData.h"

#include <cassert>

namespace lbcrypto {

using PolyType = PolyImpl<NativeVector>;

cudaPortalForApproxModDownData::cudaPortalForApproxModDownData(uint32_t ringDim, uint32_t sizeP, uint32_t sizeQ,
                                                               const std::vector<std::vector<NativeInteger>>& PHatModq, // the only crypto-parameter we need
                                                               cudaStream_t workDataStream, cudaStream_t* pipelineStreams, cudaEvent_t workDataEvent, cudaEvent_t* pipelineEvents, int id) {
    this->id = id;

    this->ringDim = ringDim;
    this->sizeP = sizeP;
    this->sizeQ = sizeQ;

    PHatModq_size_x = PHatModq.size();
    PHatModq_size_y = PHatModq[0].size();

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

    // crypto param
    const size_t phatmodq_size = PHatModq_size_x * PHatModq_size_y * sizeof(uint128_t);
    cudaHostAlloc(reinterpret_cast<void**>(&host_PHatModq) , phatmodq_size, cudaHostAllocDefault);
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&device_PHatModq), phatmodq_size, stream));

    // host input batched data
    const size_t input_data_size     = partP_empty_m_vectors_size_x * partP_empty_m_vectors_size_y * sizeof(unsigned long);
    cudaHostAlloc(reinterpret_cast<void**>(&host_cTilda_m_vectors), input_data_size, cudaHostAllocDefault);

    // device input batched data
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&device_partP_empty_m_vectors), input_data_size, stream));

    // device work (intermediate) data
    const size_t work_data_size = partPSwitchedToQ_m_vectors_size_x * partPSwitchedToQ_m_vectors_size_y * sizeof(unsigned long);
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&device_partPSwitchedToQ_m_vectors), work_data_size, stream));

    // sum (intermediate)
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&device_sum), sizeQ * ringDim * sizeof(uint128_t), stream));
    CUDA_CHECK(cudaMemsetAsync(device_sum, 0, sizeQ * ringDim * sizeof(uint128_t), stream));

    // cTildaQ
    const size_t cTildaQ_data_size = cTildaQ_m_vectors_size_x * cTildaQ_m_vectors_size_y * sizeof(ulong);
    cudaHostAlloc(reinterpret_cast<void**>(&host_cTildaQ_m_vectors), cTildaQ_data_size, cudaHostAllocDefault);
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&device_cTildaQ_m_vectors), cTildaQ_data_size, stream));

    // ans: output
    const size_t ans_size = ans_m_vectors_size_x * ans_m_vectors_size_y * sizeof(ulong);
    cudaMallocHost(reinterpret_cast<void**>(&host_ans_m_vectors), ans_size, cudaHostAllocDefault);
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&device_ans_m_vectors), ans_size, stream));
}


// Data Marshaling Functions

// marshal only the ctilda portion of interest (from sizeQ to sizeQP, so the final size equals to sizeP)
void cudaPortalForApproxModDownData::marshalCTildaBatch(const std::vector<PolyImpl<NativeVector>>& cTilda_m_vectors, const uint32_t partP_index, const uint32_t cTilda_index) const {
    const uint32_t baseIndex = partP_index * cTilda_m_vectors_size_y;
    for (uint32_t rd = 0; rd < cTilda_m_vectors_size_y; ++rd) {
        host_cTilda_m_vectors[baseIndex + rd] = cTilda_m_vectors[cTilda_index][rd].ConvertToInt<>();
    }
}

void cudaPortalForApproxModDownData::marshalCTildaQBatch(const std::vector<PolyImpl<NativeVector>>& cTilda_m_vectors, const uint32_t index) const {
    const uint32_t baseIndex = index * cTildaQ_m_vectors_size_y;
    for (uint32_t rd = 0; rd < cTilda_m_vectors_size_y; ++rd) {
        host_cTildaQ_m_vectors[baseIndex + rd] = cTilda_m_vectors[index][rd].ConvertToInt<>();
    }
}

void cudaPortalForApproxModDownData::marshalPHatModqBatch(const std::vector<std::vector<NativeInteger>>& PHatModq, const uint32_t index) const {
    const uint32_t baseIndex = index * PHatModq_size_y;
    for (uint32_t j = 0; j < PHatModq_size_y; ++j) {
        host_PHatModq[baseIndex + j] = PHatModq[index][j].ConvertToInt<>();
    }
}

void unmarshalWorkDataBatch(void *void_arg) {
    //printf("Unmarshalling\n");
    const auto* args                                   = static_cast<UnmarshalWorkDataBatchParams*>(void_arg);
    const ulong* host_ans_m_vectors                    = args->host_ans_m_vectors;
    std::vector<PolyImpl<NativeVector>>* ans_m_vectors = args->ans_m_vectors;
    const uint32_t i                                   = args->i;
    const uint32_t ptrOffset                           = args->ptrOffset;
    const uint32_t size                                = args->size;
    const ulong* host_ans_ptr                          = host_ans_m_vectors + ptrOffset;
    for(usint y = 0; y < size; y++) {
        (*ans_m_vectors)[i][y] = NativeInteger(host_ans_ptr[y]);
    }
    //free(void_arg);
}

void cudaPortalForApproxModDownData::unmarshalWorkDataBatchWrapper(std::vector<PolyImpl<NativeVector>>& ans_m_vectors,
                                                                   const uint32_t i, const uint32_t ptr_offset,
                                                                   cudaStream_t pipelineStream) const {
    auto* args = new UnmarshalWorkDataBatchParams(&ans_m_vectors, host_ans_m_vectors,
                                                  ans_m_vectors_size_y,i, ptr_offset);
    cudaLaunchHostFunc(pipelineStream, unmarshalWorkDataBatch, args);

    //cudaStreamSynchronize(pipelineStream);
    //delete args;
}

// Data Transfer Functions
void cudaPortalForApproxModDownData::copyInCTildaQ_Batch(const uint32_t ptrOffset, cudaStream_t stream) const {
    const auto device_m_vectors_ptr = device_cTildaQ_m_vectors + ptrOffset;
    const auto host_m_vectors_ptr   = host_cTildaQ_m_vectors + ptrOffset;
    const size_t size = cTildaQ_m_vectors_size_y * sizeof(ulong);
    CUDA_CHECK(cudaMemcpyAsync(device_m_vectors_ptr, host_m_vectors_ptr, size, cudaMemcpyHostToDevice, stream));
}

void cudaPortalForApproxModDownData::copyInPartP_Batch(const uint32_t ptrOffset, cudaStream_t stream) const {
    const size_t batchSize          = partP_empty_m_vectors_size_y * sizeof(unsigned long);
    const auto device_m_vectors_ptr = device_partP_empty_m_vectors + ptrOffset;
    const auto host_m_vectors_ptr   = host_cTilda_m_vectors + ptrOffset;
    CUDA_CHECK(cudaMemcpyAsync(device_m_vectors_ptr, host_m_vectors_ptr, batchSize, cudaMemcpyHostToDevice, stream));
}

void cudaPortalForApproxModDownData::copyInPHatModqBatch(const uint32_t index, cudaStream_t stream) const {
    const uint128_t batchSize = PHatModq_size_y * sizeof(uint128_t);
    const uint32_t ptrOffset  = index * PHatModq_size_y;
    const uint128_t* host_ptr = host_PHatModq + ptrOffset;
    uint128_t* device_ptr     = device_PHatModq + ptrOffset;
    CUDA_CHECK(cudaMemcpyAsync(device_ptr, host_ptr, batchSize, cudaMemcpyHostToDevice, stream));
}

void cudaPortalForApproxModDownData::copyOutResultBatch(const uint32_t ptrOffset, cudaStream_t stream) const {
    const size_t batchSize          = ans_m_vectors_size_y * sizeof(unsigned long);
    const auto device_m_vectors_ptr = device_ans_m_vectors         + ptrOffset;
    const auto host_m_vectors_ptr   = host_ans_m_vectors           + ptrOffset;
    CUDA_CHECK(cudaMemcpyAsync(host_m_vectors_ptr, device_m_vectors_ptr, batchSize, cudaMemcpyDeviceToHost, stream));
}


// Kernel Invocation Functions

void cudaPortalForApproxModDownData::invokeKernelOfApproxModDownBatchPt1(const int gpuBlocks, const int gpuThreads,
                                                                         ulong modulus, ulong tInvModp, ulong tInvModpPrecon,
                                                                         const uint32_t ptr_offset, cudaStream_t stream) {
    const auto blocks  = dim3(gpuBlocks, 1U, 1U); // Set the grid dimensions
    const auto threads = dim3(gpuThreads, 1U, 1U); // Set the block dimensions

    auto device_partP_ptr   = device_partP_empty_m_vectors + ptr_offset;

    void *args[] = {&ringDim,
                    &device_partP_ptr, &modulus,
                    &tInvModp, &tInvModpPrecon};

    approxModDownBatchPt1KernelWrapper(blocks, threads, args, stream);
}

void cudaPortalForApproxModDownData::invokeKernelOfApproxSwitchCRTBasisPt1Batch(const int gpuBlocks, const int gpuThreads,
                                                                                uint32_t i, ulong modulus, const uint32_t ptr_offset,
                                                                                ulong QHatInvModq, ulong QHatInvModqPrecon,
                                                                                cudaStream_t stream) {
    const auto blocks = dim3(gpuBlocks, 1U, 1U); // Set the grid dimensions
    const auto threads = dim3(gpuThreads, 1U, 1U); // Set the block dimensions

    auto device_m_vectors_ptr = device_partP_empty_m_vectors + ptr_offset;
    auto device_QHatModp = device_PHatModq + i * PHatModq_size_y ;

    void *args[] = {&sizeQ, &i,
                    &device_m_vectors_ptr, &modulus,
                    &QHatInvModq, &QHatInvModqPrecon,
                    &device_QHatModp,
                    &device_sum};

    approxSwitchCRTBasisPt1BatchKernelWrapper(blocks, threads, args, stream);
}

void cudaPortalForApproxModDownData::invokeKernelOfApproxSwitchCRTBasisPt2Batch(const int gpuBlocks, const int gpuThreads,
                                                                                uint32_t i, ulong ans_modulus, const uint32_t ptr_offset,
                                                                                uint128_t modpBarrettMu, uint32_t t, ulong tModqPrecon,
                                                                                cudaStream_t stream) {
    const auto blocks = dim3(gpuBlocks, 1U, 1U); // Set the grid dimensions
    const auto threads = dim3(gpuThreads, 1U, 1U); // Set the block dimensions

    auto device_partPSwitchedQ_m_vectors_ptr   = device_partPSwitchedToQ_m_vectors + ptr_offset;
    void* args[] = {&sizeQ, &i,
                    &device_sum,
                    &device_partPSwitchedQ_m_vectors_ptr, &ans_modulus,
                    &modpBarrettMu, &t, &tModqPrecon};

    approxSwitchCRTBasisPt2BatchKernelWrapper(blocks, threads, args, stream);
}

void cudaPortalForApproxModDownData::invokeAnsFillBatchKernel(const int gpuBlocks, const int gpuThreads,
                                                              uint32_t i, ulong modulus, const uint32_t ptr_offset,
                                                              ulong pInvModq, ulong pInvModqPrecon,
                                                              cudaStream_t stream) const {
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
    safeCudaFreeHost(host_PHatModq);
}

void cudaPortalForApproxModDownData::freeDeviceMemory() {

    CUDA_CHECK(cudaFreeAsync(device_partP_empty_m_vectors, stream));
    CUDA_CHECK(cudaFreeAsync(device_partPSwitchedToQ_m_vectors, stream));
    CUDA_CHECK(cudaFreeAsync(device_sum, stream));
    CUDA_CHECK(cudaFreeAsync(device_cTildaQ_m_vectors, stream));
    CUDA_CHECK(cudaFreeAsync(device_ans_m_vectors, stream));
    CUDA_CHECK(cudaFreeAsync(device_PHatModq, stream));
    device_partP_empty_m_vectors = nullptr;
    device_partPSwitchedToQ_m_vectors = nullptr;
    device_sum = nullptr;
    device_cTildaQ_m_vectors = nullptr;
    device_ans_m_vectors = nullptr;
    device_PHatModq = nullptr;
}

void cudaPortalForApproxModDownData::handleCUDAError(const std::string& operation, cudaError_t err) {
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error during " + operation + ": " + std::string(cudaGetErrorString(err)));
    }
}

}