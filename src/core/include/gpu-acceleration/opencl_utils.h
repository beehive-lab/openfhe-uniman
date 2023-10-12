//
// Created by orion on 12/09/23.
//

#ifndef OPENFHE_OPENCL_UTILS_H
#define OPENFHE_OPENCL_UTILS_H

#define CL_HPP_TARGET_OPENCL_VERSION 210  // For OpenCL 2.2

#include <CL/cl2.hpp>
#include <lattice/hal/default/dcrtpoly.h>

#define ARRAY_SIZE 2048//64

#undef HAVE_INT128
#define HAVE_INT128 FALSE

#define USE_GPU_ACCELERATION 1

extern int              isInitialized;

extern cl_int 		status;
//cl_int          i, j;

extern size_t          str_info_size;
extern char *          str_info;
extern cl_uint         uint_info;

//cl_uint         num_platforms;
extern cl_platform_id 	*platform;

extern cl_uint       	num_devices;
extern cl_device_id 	*devices;

extern cl_context context;
extern cl_program program;

//cl_command_queue_properties properties[3];
extern cl_command_queue queue;

extern cl_kernel kernel;

extern size_t local_size, global_size;

/* Data and buffers    */
extern float data[ARRAY_SIZE];
//extern cl_ulong2* sum;
extern unsigned long long* sum;
extern float total, actual_sum;
//extern cl_mem input_buffer, sum_buffer, output_buffer;
extern cl_mem mvectors_buffer, qhatinvmodq_buffer, qhatmodp_buffer, ans_mvectors_buffer;
extern cl_mem sum_buffer, output_buffer;
extern cl_int num_groups;

//void initializeOpenCL(const char* str);
void initializeKernel(const char* PROGRAM_FILE, const char* KERNEL_FUNC);
//cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename);
//std::vector<lbcrypto::DCRTPolyImpl::PolyType>
//void configKernel(std::vector<lbcrypto::DCRTPolyImpl::PolyType> m_vectors, size_t global, size_t local);
void enqueueKernel(size_t sum_length);
void deallocateBuffers();
void deallocateResources();
#endif  //OPENFHE_OPENCL_UTILS_H
