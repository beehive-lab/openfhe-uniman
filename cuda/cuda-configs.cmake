if(${CMAKE_VERSION} VERSION_LESS "3.22.0")
    message(FATAL_ERROR "Build with CUDA requires at least cmake 3.22.0")
endif ()

# inspired by: https://github.com/pytorch/pytorch/blob/59281d563154b82d60b03702095a3fe3cdd45e98/cmake/public/cuda.cmake
find_package(CUDA)
# Enable CUDA language support
set(CUDAToolkit_ROOT "${CUDA_TOOLKIT_ROOT_DIR}")
enable_language(CUDA)
include_directories("${CUDA_INCLUDE_DIRS}")

cmake_policy(SET CMP0074 NEW)

find_package(CUDAToolkit REQUIRED)
message(STATUS "GPU ACCELERATION: CUDA detected: " ${CUDA_VERSION})
message(STATUS "GPU ACCELERATION: CUDA nvcc is: " ${CUDA_NVCC_EXECUTABLE})
message(STATUS "GPU ACCELERATION: CUDA toolkit directory: " ${CUDA_TOOLKIT_ROOT_DIR})
if(CUDA_VERSION VERSION_LESS 11.5)
    message(FATAL_ERROR "GPU ACCELERATION requires CUDA 11.5 or above.")
endif()

# explicit configurations for nvcc flags
set(CMAKE_CUDA_ARCHITECTURES 61 CACHE STRING "CUDA architectures" FORCE)
# specify gpu architecture automatically
cuda_select_nvcc_arch_flags(ARCH_FLAGS Auto)
#message("==ARCH_FLAGS: " ${ARCH_FLAGS})
list( APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
# compatibility with cxx 17
list( APPEND CUDA_NVCC_FLAGS -std=c++17)
list( APPEND CUDA_NVCC_FLAGS --verbose )
#set(CUDA_PROPAGATE_HOST_FLAGS off)
set(CUDA_SEPARABLE_COMPILATION ON)