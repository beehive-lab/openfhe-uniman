//
// Created by orion on 12/09/23.
//

#include "gpu-acceleration/opencl_utils.h"
#include <math.h>
#include <iostream>
#include <thread>
#include <vector>

#define ARRAY_SIZE 2048//64

cl_int 		status;
//cl_int          i, j;

size_t          str_info_size;
char *          str_info;
cl_uint         uint_info;

//cl_uint         num_platforms;
cl_platform_id 	*platform;

cl_uint       	num_devices;
cl_device_id 	*devices;

cl_context context;
cl_program program;

//cl_command_queue_properties properties[3];
cl_command_queue queue;

cl_kernel kernel;

size_t local_size, global_size;

int call_count = 0;

/* Data and buffers    */
float data[ARRAY_SIZE];
//float sum[2], total, actual_sum;
// output data
//cl_ulong2* sum;
unsigned long long* sum;
float total, actual_sum;
// input buffers
cl_mem mvectors_buffer, qhatinvmodq_buffer, qhatmodp_buffer, ans_mvectors_buffer;
// output buffers
cl_mem sum_buffer, output_buffer;
cl_int num_groups;

/* Create program from a file and compile it */
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename) {
    printf("[host] build program start\n");

    cl_program prog;
    FILE *program_handle;
    char *program_buffer, *program_log;
    size_t program_size, log_size;
    int err;

    /* Read program file and place content into buffer */
    program_handle = fopen(filename, "r");
    if(program_handle == NULL) {
        perror("Couldn't find the program file");
        exit(1);
    }
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char*)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    size_t v = fread(program_buffer, sizeof(char), program_size, program_handle);
    if (v == 0) {
        perror("Couldn't read the program.");
        exit(1);
    }
    fclose(program_handle);

    /* Create program from file

   Creates a program from the source code in the add_numbers.cl file.
   Specifically, the code reads the file's content into a char array
   called program_buffer, and then calls clCreateProgramWithSource.
   */
    prog = clCreateProgramWithSource(ctx, 1,
                                        (const char**)&program_buffer, &program_size, &err);
    if(err < 0) {
        perror("Couldn't create the program");
        exit(1);
    }
    free(program_buffer);

    /* Build program

   The fourth parameter accepts options that configure the compilation.
   These are similar to the flags used by gcc. For example, you can
   define a macro with the option -DMACRO=VALUE and turn off optimization
   with -cl-opt-disable.
   */
    err = clBuildProgram(prog, 0, NULL, NULL, NULL, NULL);
    if(err < 0) {

        /* Find size of log and print to std output */
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG,
                              0, NULL, &log_size);
        program_log = (char*) malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG,
                              log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }
    printf("[host] build program end\n");
    return prog;
}

void initializeKernel(const char* PROGRAM_FILE, const char* KERNEL_FUNC) {
    printf("malloc\n");
    platform = (cl_platform_id*)malloc(sizeof(cl_platform_id) * 1);
    printf("clGetPlatformIDs\n");
    clGetPlatformIDs(1, platform, NULL);

    printf("clGetDeviceIDs\n");
    status = clGetDeviceIDs(*platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
    if (status != CL_SUCCESS) {
        printf("Failed to get the number of devices in the platform. [%d]\n", status);
        exit(-1);
    }

    if (num_devices > 0) {
        printf("  * The number of devices: %d\n", num_devices);
    }
    else {
        printf("Platform has no devices.\n");
        exit(-1);
    }

    devices = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);

    printf("clGetDeviceIDs\n");
    clGetDeviceIDs(*platform, CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);

    printf("    Device:\n");

    printf("clGetDeviceInfo\n");
    clGetDeviceInfo(*devices, CL_DEVICE_NAME, 0, NULL, &str_info_size);
    str_info = (char*)malloc(str_info_size);

    clGetDeviceInfo(*devices, CL_DEVICE_NAME, str_info_size, str_info, NULL);
    printf("    * name: %s\n", str_info);
    free(str_info);

    clGetDeviceInfo(*devices, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(uint_info), &uint_info, NULL);
    printf("    * Max Compute Units: %d\n", uint_info);

    clGetDeviceInfo(*devices, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(uint_info), &uint_info, NULL);
    printf("    * Max Clock freq: %d\n", uint_info);

    context = clCreateContext(NULL, 1, devices, NULL, NULL, &status);
    printf("clCreateContext\n");

    if (status < 0) {
        printf("Couldn't create a context\n");
        exit(-1);
    }

    printf("build_program\n");
    program = build_program(context, *devices, PROGRAM_FILE);

    /* Create a command queue

    Does not support profiling or out-of-order-execution
    */
    cl_command_queue_properties properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    printf("clCreateCommandQueueWithProperties\n");
    queue = clCreateCommandQueueWithProperties(context, *devices, properties, &status);
    if (status < 0) {
        perror("Couldn't create a command queue\n");
        exit(1);
    }

    /* Create a kernel */
    kernel = clCreateKernel(program, KERNEL_FUNC, &status);
    printf("clCreateKernel\n");
    if (status < 0) {
        perror("Couldn't create a kernel\n");
        exit(1);
    }
}


/* Enqueue kernel
 * At this point, the application has created all the data structures
 * (device, kernel, program, command queue, and context) needed by an
 * OpenCL host application. Now, it deploys the kernel to a device.
 *
 * Of the OpenCL functions that run on the host, clEnqueueNDRangeKernel
 * is probably the most important to understand. Not only does it deploy
 * kernels to devices, it also identifies how many work-items should
 * be generated to execute the kernel (global_size) and the number of
 * work-items in each work-group (local_size).
 * */
void enqueueKernel(size_t sum_length) {
    printf("[host] enqueueKernel start\n");
    if (call_count >= 1) {
        printf("[host] enqueueKernel abort\n");
        //deallocateResources();
        return ;
    }
    call_count++;

    std::cout << "enqueueKernel: Current Thread ID: " << std::this_thread::get_id() << std::endl;

    //status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);#
    status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
        printf("Couldn't enqueue the kernel\n");
        //deallocateResources();
        return;
    }

    //
    //size_t size_of_sum = sum_length * sizeof(cl_ulong2) * global_size;
    size_t size_of_sum = sum_length * sizeof(unsigned long long) * global_size;
    //sum = (cl_ulong2*) malloc(size_of_sum);
    sum = (unsigned long long*) malloc(size_of_sum);
    std::cout << "sizeof(sum) = " << sizeof(sum) << ", size_of_sum = " << size_of_sum << "sizeof(cl_ulong2) = " << sizeof(cl_ulong2) << std::endl;

    /* Read the kernel's output    */
    /*status = clEnqueueReadBuffer(queue, sum_buffer, CL_TRUE, 0, size_of_sum, sum, 0, NULL, NULL); // <=====GET OUTPUT
    switch (status) {
        case CL_SUCCESS:
            printf("Read operation successful.\n");
            for (size_t i = 0; i < global_size; i++) {
                for (size_t j = 0; j < sum_length; j++) {
                    //std::cout << "(" << sum[i * sum_length + j].lo << ", "<< sum[i * sum_length + j].hi << ") ";
                    std::cout << sum[i * sum_length + j] ;
                }
                std::cout << std::endl; // Start a new line for each row
            }
            break;
        case CL_INVALID_COMMAND_QUEUE:
            printf("Error: CL_INVALID_COMMAND_QUEUE - Invalid command queue.\n");
            break;
        case CL_INVALID_CONTEXT:
            printf("Error: CL_INVALID_CONTEXT - Invalid context.\n");
            break;
        case CL_INVALID_MEM_OBJECT:
            printf("Error: CL_INVALID_MEM_OBJECT - Invalid memory object (buffer).\n");
            break;
        case CL_INVALID_VALUE:
            printf("Error: CL_INVALID_VALUE - Invalid value (e.g., out-of-bounds region or NULL pointer).\n");
            break;
        case CL_INVALID_EVENT_WAIT_LIST:
            printf("Error: CL_INVALID_EVENT_WAIT_LIST - Invalid event wait list.\n");
            break;
        case CL_MISALIGNED_SUB_BUFFER_OFFSET:
            printf("Error: CL_MISALIGNED_SUB_BUFFER_OFFSET - Misaligned sub-buffer offset.\n");
            break;
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
            printf("Error: CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST - Execution status error in event wait list.\n");
            break;
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            printf("Error: CL_MEM_OBJECT_ALLOCATION_FAILURE - Memory object allocation failure.\n");
            break;
        case CL_INVALID_OPERATION:
            printf("Error: CL_INVALID_OPERATION - Invalid operation (e.g., buffer has wrong access flags).\n");
            break;
        case CL_OUT_OF_RESOURCES:
            printf("Error: CL_OUT_OF_RESOURCES - Out of OpenCL resources on the device.\n");
            break;
        case CL_OUT_OF_HOST_MEMORY:
            printf("Error: CL_OUT_OF_HOST_MEMORY - Out of host memory.\n");
            break;
        default:
            printf("Error: Unknown error code.\n");
            break;
    }*/
    free(sum);
    //float *output_data = (float *)malloc(sizeof(float) * global_size);
    //status = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0,
    //                                         sizeof(float) * global_size, output_data, 0, NULL, NULL);

    // Print the values
    //for (size_t i = 0; i < global_size; i++) {
    //    printf("get_global_id(0) value for work item %zu: %f\n", i, output_data[i]);
    //}


    /* Check result */
    /*total = 0.0f;
    for(int j=0; j<num_groups; j++) {
        total += sum[j];
    }
    actual_sum = 1.0f * ARRAY_SIZE/2*(ARRAY_SIZE-1);
    printf("Computed sum = %.1f.\n", total);
    printf("total=%f\n",total);
    printf("actual_sum=%f\n",actual_sum);
    if(fabs(total - actual_sum) > 0.01*fabs(actual_sum))
        printf("Check failed.\n");
    else
        printf("Check passed.\n");*/
    //deallocateResources();
    //printf("success\n");
    printf("[host] enqueueKernel end\n");
    return ;
}

void deallocateBuffers() {
    // deallocate input buffers
    clReleaseMemObject(mvectors_buffer);
    clReleaseMemObject(qhatinvmodq_buffer);
    clReleaseMemObject(qhatmodp_buffer);
    //clReleaseMemObject(ans_mvectors_buffer);
    clReleaseMemObject(sum_buffer);
}

void deallocateResources() {

    printf("deallocate resources\n");
    /* Deallocate resources */
    clReleaseKernel(kernel);

    //
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);
    return ;
}
