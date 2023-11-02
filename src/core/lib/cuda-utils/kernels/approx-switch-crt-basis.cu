/**
 * CUDA kernel for ApproxSwitchCRTBasis() function
 */

#include <iostream>
#include <cstdint> // for uint32_t type
#include "cuda-utils/kernel-headers/approx-switch-crt-basis.cuh"

__device__ void initSumArray(uint128_t* sum, int sizeP) {
    for(int i=0; i<sizeP; i++) {
        sum[i] = 0;
    }
}

/**
 * CUDA implementation of:
 * NativeIntegerT ModMulFastConst(const NativeIntegerT& b, const NativeIntegerT& modulus, const NativeIntegerT& bInv)
 * from core/include/math/hal/intnat/ubintnat.h
 *
 * Validated
 */
__device__ ulong ModMulFastConst(ulong a, ulong b, ulong modulus, ulong bInv) {
    //NativeInt q      = MultDHi(this->m_value, bInv.m_value);
    ulong q = __umul64hi(a, bInv);
    //NativeInt yprime = this->m_value * b.m_value - q * modulus.m_value;
    ulong yprime = a * b - q * modulus;
    //return SignedNativeInt(yprime) - SignedNativeInt(modulus.m_value) >= 0 ? yprime - modulus.m_value : yprime;
    return (long)yprime - (long)modulus >=0 ? yprime - modulus : yprime;
}

__device__ uint128_t Mul128(ulong a, ulong b) {
    return (uint128_t)a * (uint128_t)b;
}

__global__ void approxSwitchCRTBasis(int ringDim, int sizeP, int sizeQ,
                                     m_vectors_struct*  m_vectors,
                                     ulong*             QHatInvModq,
                                     ulong*             QHatInvModqPrecon,
                                     uint128_t*         QHatModp,
                                     uint128_t*         sum) {

    for(int ri = 0; ri < ringDim; ri++) {
        //__int128 sum[sizeP];
        initSumArray(sum, sizeP);
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
                sum[j] += (uint128_t)xQHatInvModqi * QHatModp[i * sizeP + j];
            }
        }
    }
}

void callApproxSwitchCRTBasisKernel(int ringDim, int sizeP, int sizeQ,
                                    m_vectors_struct*   host_m_vectors,
                                    ulong*              host_QHatInvModq,
                                    ulong*              host_QHatInvModqPrecon,
                                    uint128_t*          host_QHatModp,
                                    uint128_t*          host_sum) {

    // debugging:
    //std::cout << "==> callApproxSwitchCRTBasisKernel" << std::endl;

    cudaError_t         cudaStatus;

    m_vectors_struct*   device_m_vectors;
    ulong*              device_QHatInvModq;
    ulong*              device_QHatInvModqPrecon;
    uint128_t*          device_QHatModp;
    uint128_t*          device_sum;

    // m_vectors
    // inspired by: https://stackoverflow.com/questions/30082991/memory-allocation-on-gpu-for-dynamic-array-of-structs
    cudaMalloc((void**)&device_m_vectors, sizeQ * sizeof(m_vectors_struct));
    cudaMemcpy(device_m_vectors, host_m_vectors, sizeQ * sizeof(m_vectors_struct), cudaMemcpyHostToDevice);

    unsigned long* tmp_data[ringDim];

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
    cudaMalloc((void**)&device_sum,         sizeP * sizeof(uint128_t));
    cudaMemset(device_sum, 0, sizeP * sizeof(uint128_t));


    // cudaLaunchKernel
    dim3 blocks = dim3(1U, 1U, 1U); // Set the grid dimensions
    dim3 threads = dim3(1U, 1U, 1U); // Set the block dimensions
    void *args[] = {&ringDim, &sizeP, &sizeQ, &device_m_vectors, &device_QHatInvModq, &device_QHatInvModqPrecon, &device_QHatModp, &device_sum};
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

    cudaMemcpy(host_sum, device_sum, sizeP * sizeof(uint128_t), cudaMemcpyDeviceToHost);

    // debugging: print sum result
    /*printf("gpu_sum size = %d\n", sizeP);
    for(int i = 0; i < sizeP; i++) {
        //printf("host_sum[%d] = %llx\n", i, host_sum[i]);
        uint64_t lo = (uint64_t) host_sum[i];
        uint64_t hi = (uint64_t) (host_sum[i] >> 64);
        printf("gpu_sum[%d] = 0x%016llx%016llx\n", i, (unsigned long long)hi, (unsigned long long)lo);
    }*/

    cudaFree(device_m_vectors);
    cudaFree(device_QHatInvModq);
    cudaFree(device_QHatInvModqPrecon);
    cudaFree(device_QHatModp);
    cudaFree(device_sum);

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