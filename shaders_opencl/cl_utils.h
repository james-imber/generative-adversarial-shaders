// Copyright:    Copyright (c) Imagination Technologies Ltd 2023
// License:      MIT (refer to the accompanying LICENSE file)
// Author:       AI Research, Imagination Technologies Ltd
// Paper:        Generative Adversarial Shaders for Real-Time Realism Enhancement


#include "CL/cl.h"
#include <stdio.h>
#include <stdlib.h>

void clewErrorString(cl_int error)
{
    static const char* strings[] =
    {
        // Error Codes
          "CL_SUCCESS"                                  //   0
        , "CL_DEVICE_NOT_FOUND"                         //  -1
        , "CL_DEVICE_NOT_AVAILABLE"                     //  -2
        , "CL_COMPILER_NOT_AVAILABLE"                   //  -3
        , "CL_MEM_OBJECT_ALLOCATION_FAILURE"            //  -4
        , "CL_OUT_OF_RESOURCES"                         //  -5
        , "CL_OUT_OF_HOST_MEMORY"                       //  -6
        , "CL_PROFILING_INFO_NOT_AVAILABLE"             //  -7
        , "CL_MEM_COPY_OVERLAP"                         //  -8
        , "CL_IMAGE_FORMAT_MISMATCH"                    //  -9
        , "CL_IMAGE_FORMAT_NOT_SUPPORTED"               //  -10
        , "CL_BUILD_PROGRAM_FAILURE"                    //  -11
        , "CL_MAP_FAILURE"                              //  -12

        , ""    //  -13
        , ""    //  -14
        , ""    //  -15
        , ""    //  -16
        , ""    //  -17
        , ""    //  -18
        , ""    //  -19

        , ""    //  -20
        , ""    //  -21
        , ""    //  -22
        , ""    //  -23
        , ""    //  -24
        , ""    //  -25
        , ""    //  -26
        , ""    //  -27
        , ""    //  -28
        , ""    //  -29

        , "CL_INVALID_VALUE"                            //  -30
        , "CL_INVALID_DEVICE_TYPE"                      //  -31
        , "CL_INVALID_PLATFORM"                         //  -32
        , "CL_INVALID_DEVICE"                           //  -33
        , "CL_INVALID_CONTEXT"                          //  -34
        , "CL_INVALID_QUEUE_PROPERTIES"                 //  -35
        , "CL_INVALID_COMMAND_QUEUE"                    //  -36
        , "CL_INVALID_HOST_PTR"                         //  -37
        , "CL_INVALID_MEM_OBJECT"                       //  -38
        , "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR"          //  -39
        , "CL_INVALID_IMAGE_SIZE"                       //  -40
        , "CL_INVALID_SAMPLER"                          //  -41
        , "CL_INVALID_BINARY"                           //  -42
        , "CL_INVALID_BUILD_OPTIONS"                    //  -43
        , "CL_INVALID_PROGRAM"                          //  -44
        , "CL_INVALID_PROGRAM_EXECUTABLE"               //  -45
        , "CL_INVALID_KERNEL_NAME"                      //  -46
        , "CL_INVALID_KERNEL_DEFINITION"                //  -47
        , "CL_INVALID_KERNEL"                           //  -48
        , "CL_INVALID_ARG_INDEX"                        //  -49
        , "CL_INVALID_ARG_VALUE"                        //  -50
        , "CL_INVALID_ARG_SIZE"                         //  -51
        , "CL_INVALID_KERNEL_ARGS"                      //  -52
        , "CL_INVALID_WORK_DIMENSION"                   //  -53
        , "CL_INVALID_WORK_GROUP_SIZE"                  //  -54
        , "CL_INVALID_WORK_ITEM_SIZE"                   //  -55
        , "CL_INVALID_GLOBAL_OFFSET"                    //  -56
        , "CL_INVALID_EVENT_WAIT_LIST"                  //  -57
        , "CL_INVALID_EVENT"                            //  -58
        , "CL_INVALID_OPERATION"                        //  -59
        , "CL_INVALID_GL_OBJECT"                        //  -60
        , "CL_INVALID_BUFFER_SIZE"                      //  -61
        , "CL_INVALID_MIP_LEVEL"                        //  -62
        , "CL_INVALID_GLOBAL_WORK_SIZE"                 //  -63
        , "CL_UNKNOWN_ERROR_CODE"
    };

    if (error >= -63 && error <= 0){
        printf("%s\n", strings[-error]);
        return;
    } else {
        printf("%s\n", strings[64]);
        return;
    }
}

void check_cl_errors(cl_int err){
    if (err > 0){
        clewErrorString(err);
        exit(1);
    }

    return;
} 

cl_program build_program(cl_context cxt, cl_device_id dev, const char* filename){

    cl_program program;
    FILE *program_handle;
    char *program_buffer, *program_log;
    size_t program_size, log_size;
    int err;

    program_handle = fopen(filename, "r");
    if (program_handle == NULL){
        perror("Couldn't find the program file.");
        exit(1);
    }

    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char*)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);

    program = clCreateProgramWithSource(cxt, 1, 
      (const char**)&program_buffer, &program_size, &err);
    if(err < 0) {
        perror("Couldn't create the program");
        exit(1);
    }
    free(program_buffer);

    // err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    char* option = "-cl-fast-relaxed-math -cl-single-precision-constant";
    err = clBuildProgram(program, 0, NULL, option, NULL, NULL);

    if (err < 0){
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 
            0, NULL, &log_size);
        program_log = (char*) malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 
            log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }

    // clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_STATUS, 
    //     0, NULL, &log_size);
    // program_log = (char*) malloc(log_size + 1);
    // program_log[log_size] = '\0';
    // clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_STATUS, 
    //     log_size + 1, program_log, NULL);
    // printf("%s\n", program_log);
    // free(program_log);

    return program;
}

void max_work_group_size(cl_device_id device){

    size_t max_work_group;
    int err;

    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, 
        sizeof(size_t), &max_work_group, NULL);

    if (err < 0){
        clewErrorString(err);
        exit(1);
    }

    printf("Max work group size: %zu \n", max_work_group);
}

void set_kernel_arguments(
    cl_kernel kernel,
    cl_mem* mem_bufs,
    int num_args
){
    int err; 

    for (int i=0; i < num_args; i++){
        err = clSetKernelArg(kernel, i, sizeof(cl_mem), &mem_bufs[i]);
        check_cl_errors(err);
    }

    return;
}

void get_runtime(cl_event event){
    cl_ulong time_start;
    cl_ulong time_end;

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

    double nanoSeconds = time_end-time_start;
    printf("\nRuntime: ");
    printf("%0.3f \n",nanoSeconds / 1000000.0);
    printf("\n");
}

void print_kernel_runtime(cl_event event, char* kernel_type){
    cl_ulong time_start;
    cl_ulong time_end;

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

    double nanoSeconds = time_end-time_start;
    printf("\n%s runtime: ", kernel_type);
    printf("%0.3f \n",nanoSeconds / 1000000.0);
}

void profile_single_kernel(
    cl_event event_1, 
    cl_event event_2, 
    const int numIterations,
    const char kernel_type[]
){
    cl_int err;
    cl_ulong start = 0, end = 0;

    err = clGetEventProfilingInfo(event_1, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &start, NULL);
    check_cl_errors(err);
    err = clGetEventProfilingInfo(event_2, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    check_cl_errors(err);

    printf("(device) %s OpenCL time: %.5f ms\n\n", kernel_type, 1.0e-6 * ((double)end - (double)start) / (double)numIterations);
}

void get_runtime_n_events(cl_event *event_list, int n_events, int verbose){
    double tot_time = 0.0;
    cl_ulong time_start;
    cl_ulong time_end;

    for (int i=0; i < n_events; i++){
        clGetEventProfilingInfo(event_list[i], CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
        clGetEventProfilingInfo(event_list[i], CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

        tot_time += time_end-time_start;
    }

    // printf("Total runtime: %0.3f\n\n", tot_time / 1000000.0);
    if (verbose == 1){
        printf("\nTotal runtime: %0.3f\n", tot_time / 1000000.0);
    } else {
        printf("\n%0.3f\n", tot_time / 1000000.0);
    }

    return;
}

void save_image_from_cl_buf(
    cl_mem image_buf,
    cl_command_queue queue,
    int width,
    int height, 
    int channels,
    char* file_name
){
    size_t frame_size = channels * height * width * sizeof(unsigned char);
    unsigned char* frame_out = (unsigned char*)malloc(frame_size);

    clEnqueueReadBuffer(queue, image_buf, CL_TRUE, 0, frame_size, frame_out, 0, NULL, NULL);
    float* frame_out_float = convert_to_float(frame_out, width, height, channels);
    channel_last_reorder(frame_out_float, width, height);

    write_png_file(frame_out_float, file_name, height, width);

    free(frame_out);
    free(frame_out_float);

    return;
}
