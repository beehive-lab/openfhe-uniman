/**
 * CUDA kernel for ApproxSwitchCRTBasis() function
 */

#include "cuda-utils/kernel-headers/approx-switch-crt-basis.cuh"

#include <cinttypes>

#include "cuda-utils/m_vectors.h"
#include "cuda-utils/kernel-headers/shared_device_functions.cuh"

__device__ void initSumArray(uint128_t* sum, int sizeP) {
    for(int i=0; i<sizeP; i++) {
        sum[i] = 0;
    }
}

__global__ void approxSwitchCRTBasis(int ringDim, int sizeP, int sizeQ,
                                     m_vectors_struct*  m_vectors,
                                     ulong*             QHatInvModq,
                                     ulong*             QHatInvModqPrecon,
                                     uint128_t*         QHatModp,
                                     uint128_t*         sum,
                                     uint128_t*         modpBarrettMu,
                                     m_vectors_struct*  ans_m_vectors) {

    int ri = blockIdx.x * blockDim.x + threadIdx.x;
    //for(int ri = 0; ri < ringDim; ri++) {
    if (ri < ringDim) {
        //__int128 sum[sizeP];
        //initSumArray(sum, sizeP);
        for(int i = 0; i < sizeQ; i++) {
            //const NativeInteger& xi     = m_vectors[i][ri];
            ulong xi = m_vectors[i].data[ri];
            //const NativeInteger& qi     = m_vectors[i].GetModulus();
            ulong qi = m_vectors[i].modulus;
            //NativeInteger xQHatInvModqi = xi.ModMulFastConst(QHatInvModq[i], qi, QHatInvModqPrecon[i]);
            ulong xQHatInvModqi = ModMulFastConst(xi, QHatInvModq[i], qi, QHatInvModqPrecon[i]);
            // debugging: check ModMulFastConst - ok
            /*if(ri < 2) {
                printf("cuda_ [%d][%d]: ModMulFastConst(%ld, %ld, %ld) = %ld\n", i, ri, xi, QHatInvModq[i],qi, xQHatInvModqi);
            }*/
            for(int j = 0; j < sizeP; j++) {
                // sum[j] += Mul128(xQHatInvModqi.ConvertToInt(), QHatModp[i][j].ConvertToInt());
                sum[ri * sizeP + j] += (uint128_t)xQHatInvModqi * QHatModp[i * sizeP + j];
            }
        }

        for(int j = 0; j < sizeP; j++) {
            //const NativeInteger& pj = ans.m_vectors[j].GetModulus();
            ulong pj = ans_m_vectors[j].modulus;
            //ans.m_vectors[j][ri]    = BarrettUint128ModUint64(sum[j], pj.ConvertToInt(), modpBarrettMu[j]);
            ans_m_vectors[j].data[ri] = BarrettUint128ModUint64(sum[ri * sizeP + j], pj, modpBarrettMu[j]);
            //ans_m_vectors[j].data[ri] = ri;
        }
    }
}

__global__ void approxSwitchCRTBasisV2(int ringDim, int sizeP, int sizeQ, uint32_t pIndex,
                                     m_vectors_struct*  m_vectors,
                                     ulong*             QHatInvModq,
                                     ulong*             QHatInvModqPrecon,
                                     uint128_t*         QHatModp,
                                     uint128_t*         modpBarrettMu,
                                     m_vectors_struct*  ans_m_vectors) {

    //extern __shared__ uint128_t s[]; // Shared memory
    //extern __shared__ uint128_t m_vectors_shared[]; // Shared memory
    extern __shared__ ulong shared_qi[];
    //int t = threadIdx.x;
    int ri = blockIdx.x * blockDim.x + threadIdx.x;

    if (ri < ringDim) {
        // Initialize shared memory
        if (threadIdx.x < sizeQ) {
            shared_qi[threadIdx.x] = m_vectors[threadIdx.x].modulus;
        }
        __syncthreads();
        uint128_t sum = 0;
        //__syncthreads();

        for (int i = 0; i < sizeQ; i++) {
            ulong xi = m_vectors[i].data[ri];
            //ulong qi = shared_qi[i];

            // Modular multiplication
            ulong xQHatInvModqi = ModMulFastConst(xi, QHatInvModq[i], shared_qi[i], QHatInvModqPrecon[i]);

            // Accumulate into shared memory for this `pIndex`
            //s[t] += (uint128_t)xQHatInvModqi * QHatModp[i * sizeP + pIndex];
            sum += (uint128_t)xQHatInvModqi * QHatModp[i * sizeP + pIndex];
        }
        //__syncthreads();

        // Perform modular reduction and write the result
        ulong pj = ans_m_vectors[pIndex].modulus;
        ans_m_vectors[pIndex].data[ri] = BarrettUint128ModUint64(sum, pj, modpBarrettMu[pIndex]);
    }

}

void approxSwitchCRTBasisKernelWrapper(dim3 blocks, dim3 threads, void** args, size_t sharedMemSize, cudaStream_t stream) {
    //std::cout << "New approxSwitchCRTBasisKernelWrapper" << std::endl;
    cudaError_t         cudaStatus;

    //cudaDeviceSynchronize();
    cudaStatus = cudaLaunchKernel((void*)approxSwitchCRTBasis, blocks, threads, args, sharedMemSize, stream);
    if (cudaStatus != cudaSuccess) {
        printf("approxSwitchCRTBasis kernel launch failed: %s (%d) \n", cudaGetErrorString(cudaStatus), cudaStatus);
        //return;
        exit(-1);
    }
    //cudaDeviceSynchronize();

    //std::cout << "End New approxSwitchCRTBasisKernelWrapper" << std::endl;

}

void approxSwitchCRTBasisKernelWrapperV2(dim3 blocks, dim3 threads, void** args, size_t sharedMemSize, cudaStream_t stream) {
    cudaError_t cudaStatus = cudaLaunchKernel((void*)approxSwitchCRTBasisV2, blocks, threads, args, sharedMemSize, stream);
    if (cudaStatus != cudaSuccess) {
        printf("approxSwitchCRTBasisV2 kernel launch failed: %s (%d) \n", cudaGetErrorString(cudaStatus), cudaStatus);
        exit(-1);
    }
}

void callApproxSwitchCRTBasisKernel(int gpuBlocks, int gpuThreads,
                                    int ringDim, int sizeP, int sizeQ,
                                    m_vectors_struct*   host_m_vectors,
                                    ulong*              host_QHatInvModq,
                                    ulong*              host_QHatInvModqPrecon,
                                    uint128_t*          host_QHatModp,
                                    uint128_t*          host_modpBarrettMu,
                                    m_vectors_struct*   host_ans_m_vectors) {

    std::cout << "[callApproxSwitchCRTBasisKernel]: sizeP = " << sizeP << ", sizeQ = " << sizeQ << std::endl;

    // debugging:
    //std::cout << "==> callApproxSwitchCRTBasisKernel" << std::endl;

    cudaError_t         cudaStatus;

    m_vectors_struct*   device_m_vectors;
    ulong*              device_QHatInvModq;
    ulong*              device_QHatInvModqPrecon;
    uint128_t*          device_QHatModp;
    uint128_t*          device_sum;
    uint128_t*          device_modpBarrettMu;
    m_vectors_struct*   device_ans_m_vectors;

    // m_vectors
    // inspired by: https://stackoverflow.com/questions/30082991/memory-allocation-on-gpu-for-dynamic-array-of-structs
    cudaMalloc((void**)&device_m_vectors, sizeQ * sizeof(m_vectors_struct));
    cudaMemcpy(device_m_vectors, host_m_vectors, sizeQ * sizeof(m_vectors_struct), cudaMemcpyHostToDevice);

    unsigned long* tmp_data[sizeQ];

    for (int q = 0; q < sizeQ; ++q) {
        cudaMalloc((void**)&(tmp_data[q]), ringDim * sizeof(unsigned long));
        cudaMemcpy(&(device_m_vectors[q].data), &(tmp_data[q]), sizeof(unsigned long*), cudaMemcpyHostToDevice);
        cudaMemcpy(tmp_data[q], host_m_vectors[q].data, ringDim * sizeof(unsigned long), cudaMemcpyHostToDevice);
    }

    // qhatinvmodq
    cudaMalloc((void**)&device_QHatInvModq, sizeQ * sizeof(unsigned long));
    cudaMemcpy(device_QHatInvModq, host_QHatInvModq, sizeQ * sizeof(unsigned long), cudaMemcpyHostToDevice);

    // QHatInvModqPrecon
    cudaMalloc((void**)&device_QHatInvModqPrecon, sizeQ * sizeof(unsigned long));
    cudaMemcpy(device_QHatInvModqPrecon, host_QHatInvModqPrecon, sizeQ * sizeof(unsigned long), cudaMemcpyHostToDevice);

    // qhatmodp
    cudaMalloc((void**)&device_QHatModp,    sizeQ * sizeP * sizeof(uint128_t));
    cudaMemcpy(device_QHatModp, host_QHatModp, sizeQ * sizeP * sizeof(uint128_t), cudaMemcpyHostToDevice);

    // sum
    cudaMalloc((void**)&device_sum,         sizeP * ringDim * sizeof(uint128_t));
    cudaMemset(device_sum, 0, sizeP * ringDim * sizeof(uint128_t));

    // modpBarrettMu
    cudaMalloc((void**)&device_modpBarrettMu, sizeP * sizeof(uint128_t));
    cudaMemcpy(device_modpBarrettMu, host_modpBarrettMu, sizeP * sizeof(uint128_t), cudaMemcpyHostToDevice);

    // ans_m_vectors
    cudaMalloc((void**)&device_ans_m_vectors, sizeP * sizeof(m_vectors_struct));
    cudaMemcpy(device_ans_m_vectors, host_ans_m_vectors, sizeP * sizeof(m_vectors_struct), cudaMemcpyHostToDevice);

    unsigned long* tmp_device_ans_m_vectors_data[sizeP];

    for (int p = 0; p < sizeP; ++p) {
        cudaMalloc((void**)&(tmp_device_ans_m_vectors_data[p]), ringDim * sizeof(unsigned long));
        cudaMemcpy(&(device_ans_m_vectors[p].data), &(tmp_device_ans_m_vectors_data[p]), sizeof(unsigned long*), cudaMemcpyHostToDevice);
        //cudaMemcpy(tmp_data[q], host_m_vectors[q].data, ringDim * sizeof(unsigned long), cudaMemcpyHostToDevice);
    }


    // cudaLaunchKernel
    //dim3 blocks = dim3(1U, 1U, 1U); // Set the grid dimensions
    //cudaOccupancyMaxActiveBlocksPerMultiprocessor
    dim3 blocks = dim3(gpuBlocks, 1U, 1U); // Set the grid dimensions
    dim3 threads = dim3(gpuThreads, 1U, 1U); // Set the block dimensions
    void *args[] = {&ringDim, &sizeP, &sizeQ, &device_m_vectors, &device_QHatInvModq, &device_QHatInvModqPrecon, &device_QHatModp, &device_sum, &device_modpBarrettMu, &device_ans_m_vectors};
    // debugging:
    // printf("Before kernel launch\n");
    cudaStatus = cudaLaunchKernel((void*)approxSwitchCRTBasis, blocks, threads, args, 0U, nullptr);
    if (cudaStatus != cudaSuccess) {
        printf("approxSwitchCRTBasis kernel launch failed: %s (%d) \n", cudaGetErrorString(cudaStatus), cudaStatus);
        return;
    }
    cudaDeviceSynchronize();
    // debugging:
    //printf("After kernel launch\n");

    // copy out the result ans vector
    for(int p = 0; p < sizeP; p++) {
        cudaMemcpy(host_ans_m_vectors[p].data, tmp_device_ans_m_vectors_data[p], ringDim * sizeof(unsigned long), cudaMemcpyDeviceToHost);
    }

    // debugging: print sum result
    /*printf("gpu_sum size = %d\n", sizeP);
    for(int i = 0; i < sizeP; i++) {
        //printf("host_sum[%d] = %llx\n", i, host_sum[i]);
        uint64_t lo = (uint64_t) host_sum[i];
        uint64_t hi = (uint64_t) (host_sum[i] >> 64);
        printf("gpu_sum[%d] = 0x%016llx%016llx\n", i, (unsigned long long)hi, (unsigned long long)lo);
    }*/

    // debugging: print ans_m_vectors result -ok
    /*int tmp_ri = ringDim-1;
    for(int p = 0; p < sizeP; p++) {
        std::cout << "gpu_ans_m_vectors[" << p << ", " << tmp_ri << "] = " << host_ans_m_vectors[p].data[tmp_ri] << std::endl;
    }*/

    cudaFree(device_m_vectors);
    cudaFree(device_QHatInvModq);
    cudaFree(device_QHatInvModqPrecon);
    cudaFree(device_QHatModp);
    cudaFree(device_sum);
    cudaFree(device_modpBarrettMu);
    cudaFree(device_ans_m_vectors);

    //std::cout << "END Old callApproxSwitchCRTBasisKernel" << std::endl;

}

void printMemoryInfo() {
    size_t freeMem;
    size_t totalMem;

    cudaError_t err = cudaMemGetInfo(&freeMem, &totalMem);
    if (err != cudaSuccess) {
        std::cerr << "Error getting memory info: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    std::cout << "Total device memory: " << totalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Free device memory: " << freeMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Used device memory: " << (totalMem - freeMem) / (1024 * 1024) << " MB" << std::endl;
}


/**
 * A dummy CUDA kernel.
 */
static __global__ void myKernel(int* data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        data[tid] += 1;
    }
}

/**
 * Call the dummy kernel.
 */
void callMyKernel(uint32_t ringDim, uint32_t sizeQ, uint32_t sizeP) {

    // here
    int n = 32;
    int i;
    int* h_data;   // host data
    int* d_data;   // device data
    cudaError_t cudaStatus;

    h_data = (int*)malloc(n * sizeof(int));
    for(i=0; i < n; i++) {
        h_data[i] = 1;
    }
    ///////////////

    ///////////////

    cudaStatus = cudaMalloc((void **)&d_data, n * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s (%d) \n", cudaGetErrorString(cudaStatus), cudaStatus);
        return;
    }

    cudaStatus = cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s (%d) \n", cudaGetErrorString(cudaStatus), cudaStatus);
        return;
    }

    dim3 blocks = dim3(1U, 1U, 1U); // Set the grid dimensions
    dim3 threads = dim3(n, 1U, 1U); // Set the block dimensions
    void *args[] = { &d_data, &n};
    printf("Before kernel launch\n");
    cudaStatus = cudaLaunchKernel((void*)myKernel, blocks, threads, args, 0U, nullptr);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s (%d) \n", cudaGetErrorString(cudaStatus), cudaStatus);
        return;
    }
    cudaDeviceSynchronize();
    printf("After kernel launch\n");
    cudaMemcpy(h_data, d_data, n * sizeof(int), cudaMemcpyDeviceToHost);

    // print result
    for(i = 0; i < n; i++) {
        printf("h_data[%d] = %d\n", i, h_data[i]);
    }

    cudaFree(d_data);
    free(h_data);
}