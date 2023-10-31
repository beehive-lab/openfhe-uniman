/**
 * CUDA kernel for ApproxSwitchCRTBasis() function
 */

#include <iostream>
#include <cstdint> // for uint32_t type

static __global__ void myKernel(int* data, int n) {
    __int128 a = 0;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        data[tid] += 1;
    }
}

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