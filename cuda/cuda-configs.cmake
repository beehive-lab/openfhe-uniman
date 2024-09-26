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

# Automatically detect GPU architecture
cuda_detect_installed_gpus(INSTALLED_GPU_CCS_1)
if(NOT INSTALLED_GPU_CCS_1)
    message(FATAL_ERROR "No GPU architectures detected.")
else()
    string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
    string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
    string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
    SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
    message(STATUS "Detected CUDA architecture(s): ${CUDA_ARCH_LIST}")
endif()

# explicit configurations for nvcc flags
#message("==ARCH_FLAGS: " ${ARCH_FLAGS})
#list( APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
# compatibility with cxx 17
list( APPEND CUDA_NVCC_FLAGS -std=c++17)
list( APPEND CUDA_NVCC_FLAGS --verbose )
#set(CUDA_PROPAGATE_HOST_FLAGS off)
set(CUDA_SEPARABLE_COMPILATION ON)
# enable debug information
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")