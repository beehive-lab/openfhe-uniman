#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda-utils/kernel-headers/approx-mod-down.cuh"

#include <cinttypes>
#include <cuda-utils/cuda_util_macros.h>

#include "cuda-utils/m_vectors.h"
#include "cuda-utils/kernel-headers/shared_device_functions.cuh"

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

__device__ inline void approxSwitchCRTBasisFunc(int ri, int ringDim, int sizeP, int sizeQ,
                                     ulong*             m_vectors, uint32_t m_vectors_sizeY,
                                     ulong*             QHatInvModq,
                                     ulong*             QHatInvModqPrecon,
                                     uint128_t*         QHatModp, uint32_t QHatModp_sizeY,
                                     uint128_t*         sum,
                                     uint128_t*         modpBarrettMu,
                                     ulong*             ans_m_vectors, uint32_t ans_sizeY) {

        //if (ri == 0)
            //printf("(kernel) m_vectors size = %d\n", sizeQ*ringDim);
        for(int i = 0; i < sizeQ; i++) {
            // access the data part - index ok
            //ulong xi = m_vectors[i * ringDim + ri];
            ulong xi = m_vectors[i * m_vectors_sizeY + ri];
            // access the modulus part - index ok
            //ulong qi = m_vectors[sizeQ * ringDim + i];
            ulong qi = m_vectors[sizeQ * m_vectors_sizeY + i];
            // ok
            ulong xQHatInvModqi = ModMulFastConst(xi, QHatInvModq[i], qi, QHatInvModqPrecon[i]);
            for(int j = 0; j < sizeP; j++) {
                /*if(ri == 0) {
                    uint128_t value = (uint128_t)QHatModp[i * QHatModp_sizeY + j];
                    uint64_t lo = (uint64_t) value;
                    uint64_t hi = (uint64_t) (value >> 64);
                    uint128_t value2 = (uint128_t)sum[ri * sizeP + j];
                    uint64_t lo2 = (uint64_t) value2;
                    uint64_t hi2 = (uint64_t) (value2 >> 64);
                    printf("cuda_ before xQHatInvModqi=%ld, QHatModp[%d]=0x%016llx%016llx, sum[%d]=0x%016llx%016llx \n", xQHatInvModqi, i * sizeP + j, (unsigned long long)hi, (unsigned long long)lo, ri * sizeP + j, (unsigned long long)hi2, (unsigned long long)lo2);
                }*/
                sum[ri * sizeP + j] += (uint128_t)xQHatInvModqi * QHatModp[i * QHatModp_sizeY + j];
                /*if(ri == 0) {
                    uint128_t value = (uint128_t)sum[ri * sizeP + j];
                    uint64_t lo2 = (uint64_t) value;
                    uint64_t hi2 = (uint64_t) (value >> 64);
                    printf("cuda_ after  sum[%d]=0x%016llx%016llx \n", ri * sizeP + j, (unsigned long long)hi2, (unsigned long long)lo2);
                }*/
            }
        }
        /*if (ri==0) {
            for (uint32_t p = 0; p < sizeP; p++) {
                uint128_t value = (uint128_t)sum[p];
                uint64_t lo = (uint64_t) value;
                uint64_t hi = (uint64_t) (value >> 64);
                printf("gpu_sum[%d] = 0x%016llx%016llx\n", p, (unsigned long long)hi, (unsigned long long)lo);
            }
        }*/
        for(int j = 0; j < sizeP; j++) {
            //if (ri == 0)
            //    printf("(kernel) modulus ans_m_vectors[%d] = %llu\n", sizeP * ringDim + j, ans_m_vectors[sizeP * ringDim + j]);
            // get the modulus
            ulong pj = ans_m_vectors[sizeP * ans_sizeY + j];
            ans_m_vectors[j * ringDim + ri] = BarrettUint128ModUint64(sum[ri * sizeP + j], pj, modpBarrettMu[j]);
        }
}

__global__ void approxSwitchCRTBasisPt1Batch(int sizeP, uint32_t i,
                                             ulong*             m_vectors, ulong modulus,
                                             ulong QHatInvModq, ulong QHatInvModqPrecon,
                                             uint128_t*         QHatModp,
                                             uint128_t*         sum) {

    int ri = blockIdx.x * blockDim.x + threadIdx.x;

    ulong xi = m_vectors[ri];
    ulong xQHatInvModqi = ModMulFastConst(xi, QHatInvModq, modulus, QHatInvModqPrecon);
    for(int j = 0; j < sizeP; j++) {
        sum[ri * sizeP + j] += (uint128_t)xQHatInvModqi * QHatModp[j];
    }
}

__global__ void approxSwitchCRTBasisPt2Batch(int sizeP, uint32_t j,
                                             uint128_t* sum,
                                             ulong* ans_m_vectors, ulong modulus,
                                             uint128_t modpBarrettMu, uint32_t t, ulong tModqPrecon) {
    int ri = blockIdx.x * blockDim.x + threadIdx.x;

    ans_m_vectors[ri] = BarrettUint128ModUint64(sum[ri * sizeP + j], modulus, modpBarrettMu);

    // kernel fusion with the next step instead of approxModDownKernelPt2 (multiply everything by t mod Q)
    ans_m_vectors[ri] = ModMulFastConst(ans_m_vectors[ri], t, modulus, tModqPrecon);
}

void approxSwitchCRTBasisPt1BatchKernelWrapper(dim3 blocks, dim3 threads, void** args, cudaStream_t stream) {
    CUDA_CHECK(cudaLaunchKernel((void*)approxSwitchCRTBasisPt1Batch, blocks, threads, args, 0U, stream));
}

void approxSwitchCRTBasisPt2BatchKernelWrapper(dim3 blocks, dim3 threads, void** args, cudaStream_t stream) {
    CUDA_CHECK(cudaLaunchKernel((void*)approxSwitchCRTBasisPt2Batch, blocks, threads, args, 0U, stream));
}

__global__ void approxModDown(
    //scalar values
    int ringDim, int sizeQP, int sizeP, int sizeQ,
    // work data along with their column size
    ulong*      partP_empty_m_vectors,      uint32_t partP_empty_m_vectors_sizeX, uint32_t partP_empty_m_vectors_sizeY,
    uint128_t*  sum,
    ulong*      partPSwitchedToQ_m_vectors, uint32_t partPSwitchedToQ_sizeX, uint32_t partPSwitchedToQ_sizeY,
    // params data along with their column size (where applicable)
    ulong*      tInvModp,
    ulong*      tInvModpPrecon,
    ulong*      QHatInvModq,
    ulong*      QHatInvModqPrecon,
    uint128_t*  QHatModp,                   uint32_t QHatModp_sizeY,
    uint128_t*  modpBarrettMu,
    //
    uint32_t t,
    ulong* tModqPrecon) {

    int ri = blockIdx.x * blockDim.x + threadIdx.x;
    if (ri < ringDim) {

        for (uint32_t j = 0; j < sizeP; j++) {
            partP_empty_m_vectors[j * partP_empty_m_vectors_sizeY + ri] =
                ModMulFastConst(partP_empty_m_vectors[j * partP_empty_m_vectors_sizeY + ri],
                                tInvModp[j],
                                partP_empty_m_vectors[partP_empty_m_vectors_sizeX * partP_empty_m_vectors_sizeY + j],
                                tInvModpPrecon[j]);
            //if (ri <2)
                //printf("(kernel) {[%d]: %lu, %lu, %lu, %lu, %lu}\n", j, a, b, m, inv, res);
            //partP_empty_m_vectors[j * partP_empty_m_vectors_sizeY + ri] = res;
        }

        // swap sizeP with sizeQ
        approxSwitchCRTBasisFunc(ri, ringDim, sizeQ, sizeP, partP_empty_m_vectors, partP_empty_m_vectors_sizeY, QHatInvModq, QHatInvModqPrecon, QHatModp, QHatModp_sizeY, sum, modpBarrettMu, partPSwitchedToQ_m_vectors, partPSwitchedToQ_sizeY);

        // Multiply everything by t mod Q (BGVrns only)
        if (t > 0) {
            for (uint32_t q = 0; q < sizeQ; q++) {
                partPSwitchedToQ_m_vectors[q * partPSwitchedToQ_sizeY + ri] =
                    ModMulFastConst(partPSwitchedToQ_m_vectors[q * partPSwitchedToQ_sizeY + ri],
                                    t,
                                    partPSwitchedToQ_m_vectors[partPSwitchedToQ_sizeX * partPSwitchedToQ_sizeY + q],
                                    tModqPrecon[q]);
            }
        }
    }
}

__global__ void approxModDownBatchPt1(
    //scalar values
    int ringDim,
    // work data along with their column size
    ulong*      partP_empty_m_vectors, ulong partP_empty_modulus,
    // params data along with their column size (where applicable)
    ulong       tInvModp,
    ulong      tInvModpPrecon) {

    int ri = blockIdx.x * blockDim.x + threadIdx.x;
    if (ri < ringDim) {
        partP_empty_m_vectors[ri] =
            ModMulFastConst(partP_empty_m_vectors[ri],
                            tInvModp,
                            partP_empty_modulus,
                            tInvModpPrecon);
    }
}

void approxModDownKernelWrapper(dim3 blocks, dim3 threads, void** args, cudaStream_t stream) {
    //std::cout << "New approxSwitchCRTBasisKernelWrapper" << std::endl;
    cudaError_t         cudaStatus;

    //cudaDeviceSynchronize();
    cudaStatus = cudaLaunchKernel((void*)approxModDown, blocks, threads, args, 0U, stream);
    if (cudaStatus != cudaSuccess) {
        printf("approxModDown kernel launch failed: %s (%d) \n", cudaGetErrorString(cudaStatus), cudaStatus);
        //return;
        exit(-1);
    }
    //cudaDeviceSynchronize();

    //std::cout << "End New approxSwitchCRTBasisKernelWrapper" << std::endl;
}

void approxModDownBatchPt1KernelWrapper(dim3 blocks, dim3 threads, void** args, cudaStream_t stream) {
    CUDA_CHECK(cudaLaunchKernel((void*)approxModDownBatchPt1, blocks, threads, args, 0U, stream));
}

__global__ void ansFill(int sizeQ,
                        ulong*      cTilda_m_vectors,           uint32_t cTilda_m_vectors_sizeX, uint32_t cTilda_m_vectors_sizeY,
                        ulong*      partPSwitchedToQ_m_vectors, uint32_t partPSwitchedToQ_sizeX, uint32_t partPSwitchedToQ_sizeY,
                        ulong*      ans_m_vectors, uint32_t ans_sizeX, uint32_t ans_sizeY,
                        ulong*      pInvModq,
                        ulong*      pInvModqPrecon) {

    int ri = blockIdx.x * blockDim.x + threadIdx.x;

    for (uint32_t q = 0; q < sizeQ; q++) {
        ulong cTildaModulus = cTilda_m_vectors[cTilda_m_vectors_sizeX * cTilda_m_vectors_sizeY + q];

        ulong diff = ModSubFast(cTilda_m_vectors[q * cTilda_m_vectors_sizeY + ri], partPSwitchedToQ_m_vectors[q * partPSwitchedToQ_sizeY + ri], cTildaModulus); // ok
        ans_m_vectors[q * ans_sizeY + ri] = ModMulFastConst(diff, pInvModq[q], cTildaModulus, pInvModqPrecon[q]);
    }
}

void ansFillKernelWrapper(dim3 blocks, dim3 threads, void** args, cudaStream_t stream) {
    cudaError_t         cudaStatus;
    cudaStatus = cudaLaunchKernel((void*)ansFill, blocks, threads, args, 0U, stream);
    if (cudaStatus != cudaSuccess) {
        printf("ansFill kernel launch failed: %s (%d) \n", cudaGetErrorString(cudaStatus), cudaStatus);
        exit(-1);
    }
}

__global__ void ansFillBatch(uint32_t i, ulong*      cTildaQ_m_vectors, ulong cTildaQ_modulus,
                             ulong*      partPSwitchedToQ_m_vectors,
                             ulong*      ans_m_vectors,
                             ulong       pInvModq,
                             ulong       pInvModqPrecon) {
    int ri = blockIdx.x * blockDim.x + threadIdx.x;

    ulong diff = ModSubFast(cTildaQ_m_vectors[ri], partPSwitchedToQ_m_vectors[ri], cTildaQ_modulus);
    ans_m_vectors[ri] = ModMulFastConst(diff, pInvModq, cTildaQ_modulus, pInvModqPrecon);
}

void ansFillBatchKernelWrapper(dim3 blocks, dim3 threads, void** args, cudaStream_t stream) {
    CUDA_CHECK(cudaLaunchKernel((void*)ansFillBatch, blocks, threads, args, 0U, stream));
}
