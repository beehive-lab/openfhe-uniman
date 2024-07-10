include_directories(${CUDA_INCLUDE_DIRS})

# find all cuda source (.cu) at core/lib/cuda-utils
file(GLOB_RECURSE CUDA_SRC_FILES lib/cuda-utils/kernels/*.cu include/cuda-utils/kernel-headers/*.cuh)

set_source_files_properties(${CUDA_SRC_FILES} PROPERTIES LANGUAGE CUDA)

foreach(cu_file ${CUDA_SRC_FILES})
    message(STATUS "Found CUDA source file: ${cu_file}")
endforeach()

# create a target for the cuda library
add_library(CUDALibrary SHARED ${CUDA_SRC_FILES})
# STATIC = libraries are archives of object files for use when linking other targets
# SHARED = libraries are linked dynamically and loaded at runtime
set_target_properties(CUDALibrary PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(CUDALibrary PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(CUDALibrary PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

target_compile_features(CUDALibrary PRIVATE cuda_std_17)

# CUDA libraries to include
#message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}") # /usr/local/cuda-12.2/include
#message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}") #/usr/local/cuda-12.2/lib64/libcudart_static.a;Threads::Threads;dl;/usr/lib/x86_64-linux-gnu/librt.so
target_link_libraries(CUDALibrary ${CUDA_LIBRARIES})
target_include_directories(CUDALibrary PUBLIC ${CUDA_INCLUDE_DIRS})

install(TARGETS CUDALibrary
        EXPORT OpenFHETargets
        DESTINATION lib)