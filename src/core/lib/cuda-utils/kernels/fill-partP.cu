#include "cuda-utils/kernel-headers/fill-partP.cuh"

__global__ void fill_partP(int sizeQP, int sizeQ,
ulong*      cTilda_m_vectors,           uint32_t cTilda_m_vectors_sizeX, uint32_t cTilda_m_vectors_sizeY,
ulong*      partP_empty_m_vectors,      uint32_t partP_empty_m_vectors_sizeX, uint32_t partP_empty_m_vectors_sizeY) {

    int ri = blockIdx.x * blockDim.x + threadIdx.x;

    // fill partP_empty - ok
    for (uint32_t i = sizeQ, j = 0; i < sizeQP; i++, j++) {
        // data - ok
        partP_empty_m_vectors[j * partP_empty_m_vectors_sizeY + ri] = cTilda_m_vectors[i * cTilda_m_vectors_sizeY + ri];
        // modulus - ok
        if (ri <= j)
            partP_empty_m_vectors[partP_empty_m_vectors_sizeX * partP_empty_m_vectors_sizeY + j] = cTilda_m_vectors[cTilda_m_vectors_sizeX * cTilda_m_vectors_sizeY + i];
    }

    // check partP_empty_m_vectors
    if (ri == 0) {
        for(int i = 0; i<partP_empty_m_vectors_sizeX; i++) {
            for(int j = 0; j<10; j++)
                printf("(fill-partP kernel) partP_empty[%d] = %lu\n", i * partP_empty_m_vectors_sizeY + j, partP_empty_m_vectors[i * partP_empty_m_vectors_sizeY + j]);
        }
    }
}

void fillPartPKernelWrapper(dim3 blocks, dim3 threads, void** args, cudaStream_t stream) {
    cudaError_t         cudaStatus;

    //cudaDeviceSynchronize();
    cudaStatus = cudaLaunchKernel((void*)fill_partP, blocks, threads, args, 0U, stream);
    if (cudaStatus != cudaSuccess) {
        printf("fill_partP kernel launch failed: %s (%d) \n", cudaGetErrorString(cudaStatus), cudaStatus);
        //return;
        exit(-1);
    }
}