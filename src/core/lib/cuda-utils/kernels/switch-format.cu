#include "cuda-utils/kernel-headers/switch-format.cuh"

#include "cuda-utils/kernel-headers/shared_device_functions.cuh"

// NOTE: check whether if condition is reduntant
__device__ inline uint32_t getMSB64(uint64_t x) {
    if (x == 0)
        return 0;
    return 64 - __clzll(x);
}

// ok
__device__ int calculateM(int ri) {
    if (ri == 0) return 1; // Special case for ri = 0
    return 1 << (31 - __clz(ri + 1)); // Adjusting for the range

}

// ok
__device__ int calculateI(int ri, int m) {
    // Calculate i so that it starts from 0
    return ri - m + 1; // Adjusting the calculation
}


__global__ void inverseNTT_Part1(
    uint32_t p, uint32_t m, uint32_t n, uint32_t step,
    ulong* element, uint32_t sizeX, uint32_t sizeY,
    ulong* rootOfUnityInverseTable,
    ulong* preconRootOfUnityInverseTable
    ) {

    // Use thread ID to handle indices
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    //uint32_t blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
    uint32_t p_offset = p * sizeY;
    ulong modulus = element[sizeX * sizeY + p];

    if (tid < (n >> 1)) {
        //uint32_t step        = (n/m) >> 1;
        uint32_t psi_step    = tid/step; // i -> middle loop
        uint32_t indexOmega  = m + psi_step;
        ulong omega          = rootOfUnityInverseTable[p_offset + indexOmega]; // ok
        ulong preconOmega    = preconRootOfUnityInverseTable[p_offset + indexOmega];
        uint32_t target_idx  = (psi_step * step << 1) + (tid % step); // indexLo -> inner loop
        uint32_t indexLo     = target_idx;
        uint32_t indexHi     = target_idx + step;

        ulong hiVal = element[p_offset + indexHi]; // ok
        ulong loVal = element[p_offset + indexLo]; // ok

        ulong omegaFactor = loVal;
        // conditional expression instead of if statement?
        //omegaFactor += (modulus * (omegaFactor < hiVal));

        if (omegaFactor < hiVal)
            omegaFactor += modulus;

        omegaFactor -= hiVal;

        loVal += hiVal;
        if (loVal >= modulus)
            loVal -= modulus;

        //if (m == 1 && target_idx <5)
        //printf("(omegaFactor kernel): omegaFactor[%u] = %lu, omega=%lu, modulus=%lu, preconOmega=%lu ",indexHi, omegaFactor, omega, modulus, preconOmega);
        omegaFactor = ModMulFastConst(omegaFactor, omega, modulus, preconOmega);
        //if (m == 1 && target_idx <5)
        //printf("result[%u]=%lu\n", indexHi, omegaFactor);

        element[p_offset + indexLo] = loVal;
        element[p_offset + indexHi] = omegaFactor;
        //if (tid < 16)
        //printf("m=%d, tid=%d, i=%d, indexOmega=%d, omega=%llu, [indexLo=%d, loVal=%llu], [indexHi=%d, hiVal=%llu] \n", m, tid, psi_step, indexOmega, omega, target_idx, loVal, target_idx+step, hiVal);
    }
}

__global__ void inverseNTT_Part2(
    int p,
    ulong* element, uint32_t sizeX, uint32_t sizeY,
    ulong* cycloOrderInv,
    ulong* preconCycloOrderInv) {

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t p_offset = p * sizeY;
    ulong modulus = element[sizeX * sizeY + p];

    //if (tid < 5)
        //printf("(pt2 kernel): p=%d, element[%d] = %lu, cycloOrderInv=%lu, modulus=%lu, preconCycloOrderInv=%lu ", p, tid, element[p_offset + tid], cycloOrderInv[p], modulus, preconCycloOrderInv[p]);
    element[p_offset + tid] = ModMulFastConst(element[p_offset + tid], cycloOrderInv[p], modulus, preconCycloOrderInv[p]);
    //if (tid < 5)
        //printf("result[%d] = %lu\n", tid, element[p_offset + tid]);
}



void iNTTKernelWrapper(dim3 blocks, dim3 threads, void** args, cudaStream_t stream) {
    cudaError_t         cudaStatus;

    // Check if kernel configuration is valid
    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, 0);
    if (threads.x > deviceProps.maxThreadsPerBlock) {
        std::cerr << "threadsPerBlock exceeds maxThreadsPerBlock!" << std::endl;
        return;
    }
    if (blocks.y > deviceProps.maxGridSize[0]) {
        std::cerr << "Grid size exceeds maximum!" << std::endl;
        return;
    }

    cudaStatus = cudaLaunchKernel((void*)inverseNTT_Part1, blocks, threads, args, 0U, stream);

    if (cudaStatus != cudaSuccess) {
        printf("inverseNTTKernel kernel launch failed: %s (%d) \n", cudaGetErrorString(cudaStatus), cudaStatus);
        exit(-1);
    }
}

void iNTTPart2Wrapper(dim3 blocksPt2, dim3 threads, void** args, cudaStream_t stream) {
    cudaError_t         cudaStatus;

    // Check if kernel configuration is valid
    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, 0);
    if (threads.x > deviceProps.maxThreadsPerBlock) {
        std::cerr << "threadsPerBlock exceeds maxThreadsPerBlock!" << std::endl;
        return;
    }
    if (blocksPt2.y > deviceProps.maxGridSize[0]) {
        std::cerr << "Grid size exceeds maximum!" << std::endl;
        return;
    }

    cudaStatus = cudaLaunchKernel((void*)inverseNTT_Part2, blocksPt2, threads, args, 0U, stream);

    if (cudaStatus != cudaSuccess) {
        printf("inverseNTTKernelPt2 kernel launch failed: %s (%d) \n", cudaGetErrorString(cudaStatus), cudaStatus);
        exit(-1);
    }
}

__device__ inline void switchToEvaluationFormat() {}