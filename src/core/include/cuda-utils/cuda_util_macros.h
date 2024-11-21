#ifndef CUDA_UTIL_MACROS_H
#define CUDA_UTIL_MACROS_H

#include <cuda_runtime.h>
#include <iostream>

/**
 *  A macro that checks the return value of CUDA calls and reports errors.
 * @param call
 */
#define CUDA_CHECK(call)                                             \
    do {                                                             \
        cudaError_t err = call;                                      \
        if (err != cudaSuccess) {                                    \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    } while (0)

#define CUDA_SAFE_FREE(ptr, stream)                                  \
    do {                                                             \
        if ((ptr) != nullptr) {                                      \
            cudaError_t err = cudaFreeAsync((ptr), (stream));        \
            if (err != cudaSuccess) {                                \
                fprintf(stderr, "CUDA Error at %s:%d - %s\n",        \
                        __FILE__, __LINE__, cudaGetErrorString(err));\
                exit(EXIT_FAILURE);                                  \
            }                                                        \
            (ptr) = nullptr;                                         \
        }                                                            \
    } while (0)

#define SAFE_FREE(ptr) \
    do {               \
        free(ptr);     \
        ptr = NULL;    \
    } while (0)

#define SAFE_CUDA_FREE_HOST(ptr) \
    do {                         \
        cudaFreeHost(ptr);       \
        ptr = NULL;              \
    } while (0)


#endif //CUDA_UTIL_MACROS_H