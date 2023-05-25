// Copyright:    Copyright (c) Imagination Technologies Ltd 2023
// License:      MIT (refer to the accompanying LICENSE file)
// Author:       AI Research, Imagination Technologies Ltd
// Paper:        Generative Adversarial Shaders for Real-Time Realism Enhancement


#define LENS_BLUR "./cl_files/lens_blur.cl"
#define BLOOM "./cl_files/bloom.cl"
#define IMG_H 1080
#define IMG_W 1920
#define TILE_SIZE_COL 4
#define TILE_SIZE_ROW 4
#define TILE_SIZE_COL_DW 2
#define TILE_SIZE_ROW_DW 2

#include "CL/cl.h"
#include <stdio.h>
#include <stdlib.h>
#include <png.h>
#include <errno.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>

#include "utils.h"
#include "cl_utils.h"

int main(){
    const char * png_file = "input.png";
    int height, width;

    float* img_array = read_png_image(png_file, &height, &width);
    printf("%d, %d \n", height, width);
    channel_first_reorder(img_array, width, height);
    unsigned char* img_array_uint = convert_to_uint(img_array, width, height, 3);

    size_t frame_size = height*width*3*sizeof(unsigned char);

    cl_int err;

    // Get a platform.
    cl_platform_id platform;
    err = clGetPlatformIDs( 1, &platform, NULL);

    check_cl_errors(err);

    // Find GPU Device.
    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    check_cl_errors(err);

    // Create a context and command queue on device.
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    check_cl_errors(err);
    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    check_cl_errors(err);

    cl_program lens_blur_program = build_program(context, device, LENS_BLUR);
    cl_program bloom_program = build_program(context, device, BLOOM);

    // Define compute kernels.
    cl_kernel lens_blur = clCreateKernel(lens_blur_program, "lens_blur", &err);
    check_cl_errors(err);
    cl_kernel downsample = clCreateKernel(bloom_program, "downsample", &err);
    check_cl_errors(err);
    cl_kernel bloom_2 = clCreateKernel(bloom_program, "bloom_2", &err);
    check_cl_errors(err);
    cl_kernel bloom_4 = clCreateKernel(bloom_program, "bloom_4", &err);
    check_cl_errors(err);
    cl_kernel bloom_8 = clCreateKernel(bloom_program, "bloom_8", &err);
    check_cl_errors(err);
    cl_kernel blend_bloom = clCreateKernel(bloom_program, "blend_multiscale_bloom", &err);
    check_cl_errors(err);

    // Lens blur
    cl_mem input_buf  = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, frame_size, img_array_uint, &err);
    check_cl_errors(err);
    cl_mem dw_2_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, frame_size/4, NULL, &err);
    check_cl_errors(err);
    cl_mem lb_out_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, frame_size*2, NULL, &err);
    check_cl_errors(err);
    // cl_mem lb_out_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, frame_size, NULL, &err);
    // check_cl_errors(err);
    
    // Downscales
    cl_mem dw_4_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, frame_size/16, NULL, &err);
    check_cl_errors(err);
    cl_mem dw_8_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, frame_size/64, NULL, &err);
    check_cl_errors(err);

    // Bloom bufs
    cl_mem bloom_out_2_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, frame_size/4, NULL, &err);
    check_cl_errors(err);
    cl_mem bloom_out_4_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, frame_size/16, NULL, &err);
    check_cl_errors(err);
    cl_mem bloom_out_8_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, frame_size/64, NULL, &err);
    check_cl_errors(err);

    // Blend bufs 
    cl_mem blend = clCreateBuffer(context, CL_MEM_READ_WRITE, frame_size, NULL, &err);
    check_cl_errors(err);

    cl_mem lens_blur_args_bufs[3] = {input_buf, dw_2_buf, lb_out_buf};
    set_kernel_arguments(lens_blur, lens_blur_args_bufs, 3);

    cl_mem downsample_args_bufs[3] = {dw_2_buf, dw_4_buf, dw_8_buf};
    set_kernel_arguments(downsample, downsample_args_bufs, 3);

    cl_mem bloom_2_args_bufs[2] = {
        dw_2_buf, bloom_out_2_buf
    };
    set_kernel_arguments(bloom_2, bloom_2_args_bufs, 2);
    
    cl_mem bloom_4_args_bufs[2] = {
        dw_4_buf, bloom_out_4_buf
    };
    set_kernel_arguments(bloom_4, bloom_4_args_bufs, 2);

    cl_mem bloom_8_args_bufs[2] = {
        dw_8_buf, bloom_out_8_buf
    };
    set_kernel_arguments(bloom_8, bloom_8_args_bufs, 2);

    cl_mem blend_args_bufs[5] = {
        lb_out_buf, bloom_out_2_buf, bloom_out_4_buf, bloom_out_8_buf, blend};
    set_kernel_arguments(blend_bloom, blend_args_bufs, 5);

    size_t lb_gl_sz[2] = {(width + TILE_SIZE_COL - 1)/TILE_SIZE_COL + 32, (height + TILE_SIZE_ROW - 1)/TILE_SIZE_ROW + 42};
    size_t lb_lo_sz[2] = {32, 8};

    size_t dw_gl_sz[2] = {480, 272};
    size_t dw_lo_sz[2] = {32, 8};

    size_t half_gl_sz[2] = {(width/2 + TILE_SIZE_COL - 1)/TILE_SIZE_COL + 16, (height/2)/TILE_SIZE_ROW + 25};
    size_t half_lo_sz[2] = {32, 8};

    size_t quarter_gl_sz[2] = {(width/4 + TILE_SIZE_COL - 1)/TILE_SIZE_COL + 8, (height/4)/TILE_SIZE_ROW + 13};
    size_t quarter_lo_sz[2] = {32, 8};

    size_t eighth_gl_sz[2] = {(width/8 + TILE_SIZE_COL - 1)/TILE_SIZE_COL + 36, (height/8 + TILE_SIZE_ROW - 1)/TILE_SIZE_ROW + 6};
    size_t eighth_lo_sz[2] = {32, 8};

    size_t blend_gl_sz[2] = {(width + TILE_SIZE_COL - 1)/TILE_SIZE_COL, (height + TILE_SIZE_ROW - 1)/TILE_SIZE_ROW };
    size_t blend_lo_sz[2] = {32, 1};

    cl_event event[2];
    const int numIterations = 1000;
    float tot = 0.0;
    cl_ulong start = 0, end = 0;
    float measurements[numIterations];
    float runtime;

    for (int i=0; i < numIterations; i++){
        clEnqueueMarker(queue, &event[0]);
        err = clEnqueueNDRangeKernel(queue, lens_blur, 2, 0, lb_gl_sz, lb_lo_sz, 0, NULL, NULL);
        err = clEnqueueNDRangeKernel(queue, downsample, 2, 0, dw_gl_sz, dw_lo_sz, 0, NULL, NULL); 
        err = clEnqueueNDRangeKernel(queue, bloom_2, 2, 0, half_gl_sz, half_lo_sz, 0, NULL, NULL); 
        err = clEnqueueNDRangeKernel(queue, bloom_4, 2, 0, quarter_gl_sz, quarter_lo_sz, 0, NULL, NULL); 
        err = clEnqueueNDRangeKernel(queue, bloom_8, 2, 0, eighth_gl_sz, eighth_lo_sz, 0, NULL, NULL);  
        err = clEnqueueNDRangeKernel(queue, blend_bloom, 2, 0, blend_gl_sz, blend_lo_sz, 0, NULL, NULL);
        clEnqueueMarker(queue, &event[1]);

        clWaitForEvents(2, event);

        err = clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &start, NULL);
        check_cl_errors(err);
        err = clGetEventProfilingInfo(event[1], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
        check_cl_errors(err);

        runtime = (1.0e-6 * ((double)end - (double)start));
        tot += runtime;
        measurements[i] = runtime;
    }

    float average;
    average = tot / numIterations;
    printf("Average: %.6f ms\n", average);

    tot = 0.0;
    for (int i=0; i < numIterations; i++){
        tot += pow((measurements[i] - average), 2);
    }

    float std;
    std = sqrt(tot / numIterations);
    printf("Std dev: %.6f \n", std);
    
    save_image_from_cl_buf(
        blend, queue, width, height, 3, "./output.png");

    return 0;
}

// int main(){
//     const char * png_file = "input.png";
//     int height, width;

//     float* img_array = read_png_image(png_file, &height, &width);
//     printf("%d, %d \n", height, width);
//     channel_first_reorder(img_array, width, height);
//     unsigned char* img_array_uint = convert_to_uint(img_array, width, height, 3);

//     size_t frame_size = height*width*3*sizeof(unsigned char);

//     cl_int err;

//     // Get a platform.
//     cl_platform_id platform;
//     err = clGetPlatformIDs( 1, &platform, NULL);

//     check_cl_errors(err);

//     // Find GPU Device.
//     cl_device_id device;
//     err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
//     check_cl_errors(err);

//     // Create a context and command queue on device.
//     cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
//     check_cl_errors(err);
//     cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE|CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
//     check_cl_errors(err);

//     cl_program lens_blur_program = build_program(context, device, LENS_BLUR);
//     cl_program bloom_program = build_program(context, device, BLOOM);

//     // Define compute kernels.
//     cl_kernel lens_blur = clCreateKernel(lens_blur_program, "lens_blur", &err);
//     check_cl_errors(err);
//     cl_kernel downsample = clCreateKernel(bloom_program, "downsample", &err);
//     check_cl_errors(err);
//     cl_kernel bloom_2 = clCreateKernel(bloom_program, "bloom_2", &err);
//     check_cl_errors(err);
//     cl_kernel bloom_4 = clCreateKernel(bloom_program, "bloom_4", &err);
//     check_cl_errors(err);
//     cl_kernel bloom_8 = clCreateKernel(bloom_program, "bloom_8", &err);
//     check_cl_errors(err);
//     cl_kernel blend_bloom = clCreateKernel(bloom_program, "blend_multiscale_bloom", &err);
//     check_cl_errors(err);

//     // Lens blur
//     cl_mem input_buf  = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, frame_size, img_array_uint, &err);
//     check_cl_errors(err);
//     cl_mem dw_2_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, frame_size/4, NULL, &err);
//     check_cl_errors(err);
//     cl_mem lb_out_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, frame_size*2, NULL, &err);
//     check_cl_errors(err);
//     // cl_mem lb_out_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, frame_size, NULL, &err);
//     // check_cl_errors(err);
    
//     // Downscales
//     cl_mem dw_4_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, frame_size/16, NULL, &err);
//     check_cl_errors(err);
//     cl_mem dw_8_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, frame_size/64, NULL, &err);
//     check_cl_errors(err);

//     // Bloom bufs
//     cl_mem bloom_out_2_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, frame_size/4, NULL, &err);
//     check_cl_errors(err);
//     cl_mem bloom_out_4_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, frame_size/16, NULL, &err);
//     check_cl_errors(err);
//     cl_mem bloom_out_8_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, frame_size/64, NULL, &err);
//     check_cl_errors(err);

//     // Blend bufs 
//     cl_mem blend = clCreateBuffer(context, CL_MEM_READ_WRITE, frame_size, NULL, &err);
//     check_cl_errors(err);

//     cl_mem lens_blur_args_bufs[3] = {input_buf, dw_2_buf, lb_out_buf};
//     set_kernel_arguments(lens_blur, lens_blur_args_bufs, 3);

//     cl_mem downsample_args_bufs[3] = {dw_2_buf, dw_4_buf, dw_8_buf};
//     set_kernel_arguments(downsample, downsample_args_bufs, 3);

//     cl_mem bloom_2_args_bufs[2] = {
//         dw_2_buf, bloom_out_2_buf
//     };
//     set_kernel_arguments(bloom_2, bloom_2_args_bufs, 2);
    
//     cl_mem bloom_4_args_bufs[2] = {
//         dw_4_buf, bloom_out_4_buf
//     };
//     set_kernel_arguments(bloom_4, bloom_4_args_bufs, 2);

//     cl_mem bloom_8_args_bufs[2] = {
//         dw_8_buf, bloom_out_8_buf
//     };
//     set_kernel_arguments(bloom_8, bloom_8_args_bufs, 2);

//     cl_mem blend_args_bufs[5] = {
//         lb_out_buf, bloom_out_2_buf, bloom_out_4_buf, bloom_out_8_buf, blend};
//     set_kernel_arguments(blend_bloom, blend_args_bufs, 5);
//     // cl_mem blend_args_bufs[3] = {
//     //     lb_out_buf, bloom_out_2_buf, blend};
//     // set_kernel_arguments(blend_bloom, blend_args_bufs, 3);
//     // cl_mem blend_args_bufs[4] = {
//     //     lb_out_buf, bloom_out_2_buf, bloom_out_4_buf, blend};
//     // set_kernel_arguments(blend_bloom, blend_args_bufs, 4);

//     size_t lb_gl_sz[2] = {(width + TILE_SIZE_COL - 1)/TILE_SIZE_COL + 32, (height + TILE_SIZE_ROW - 1)/TILE_SIZE_ROW + 42};
//     size_t lb_lo_sz[2] = {32, 8};

//     size_t dw_gl_sz[2] = {480, 272};
//     size_t dw_lo_sz[2] = {32, 8};

//     size_t half_gl_sz[2] = {(width/2 + TILE_SIZE_COL - 1)/TILE_SIZE_COL + 16, (height/2)/TILE_SIZE_ROW + 25};
//     size_t half_lo_sz[2] = {32, 8};

//     size_t quarter_gl_sz[2] = {(width/4 + TILE_SIZE_COL - 1)/TILE_SIZE_COL + 8, (height/4)/TILE_SIZE_ROW + 13};
//     size_t quarter_lo_sz[2] = {32, 8};

//     size_t eighth_gl_sz[2] = {(width/8 + TILE_SIZE_COL - 1)/TILE_SIZE_COL + 36, (height/8 + TILE_SIZE_ROW - 1)/TILE_SIZE_ROW + 6};
//     size_t eighth_lo_sz[2] = {32, 8};

//     // size_t blend_gl_sz[2] = {(width + 16 - 1)/16, height};
//     // size_t blend_lo_sz[2] = {1, 4};
//     size_t blend_gl_sz[2] = {(width + TILE_SIZE_COL - 1)/TILE_SIZE_COL, (height + TILE_SIZE_ROW - 1)/TILE_SIZE_ROW };
//     size_t blend_lo_sz[2] = {32, 1};
//     // size_t blend_gl_sz[2] = {(width + TILE_SIZE_COL - 1)/TILE_SIZE_COL, (height + TILE_SIZE_ROW - 1)/TILE_SIZE_ROW};
//     // size_t blend_lo_sz[2] = {1, 1};

//     cl_event sync_event[6];
//     cl_event event[2];
//     const int numIterations = 100;

//     // err = clEnqueueNDRangeKernel(queue, lens_blur, 2, 0, lb_gl_sz, lb_lo_sz, 0, NULL, &sync_event[0]);
//     // cl_event dw_waitlist[1] = {sync_event[0]};
//     // err = clEnqueueNDRangeKernel(queue, downsample, 2, 0, dw_gl_sz, dw_lo_sz, 1, dw_waitlist, &sync_event[1]);
//     // err = clEnqueueNDRangeKernel(queue, bloom_2, 2, 0, half_gl_sz, half_lo_sz, 1, dw_waitlist, &sync_event[2]);
//     // cl_event bloom_waitlist[1] = {sync_event[1]};
//     // err = clEnqueueNDRangeKernel(queue, bloom_4, 2, 0, quarter_gl_sz, quarter_lo_sz, 1, bloom_waitlist, NULL);
//     // err = clEnqueueNDRangeKernel(queue, bloom_8, 2, 0, eighth_gl_sz, eighth_lo_sz, 1, bloom_waitlist, NULL);
//     // clEnqueueBarrier(queue);
//     // // cl_event blend_waitlist[3] = {&sync_event[2], &sync_event[3], &sync_event[4]};
//     // err = clEnqueueNDRangeKernel(queue, blend_bloom, 2, 0, blend_gl_sz, blend_lo_sz, 0, NULL, NULL);

//     // clFinish(queue);

//     clEnqueueMarker(queue, &event[0]);
//     for (int i=0; i < numIterations; i++){
//         err = clEnqueueNDRangeKernel(queue, lens_blur, 2, 0, lb_gl_sz, lb_lo_sz, 0, NULL, &sync_event[0]);
//         cl_event dw_waitlist[1] = {sync_event[0]};
//         err = clEnqueueNDRangeKernel(queue, downsample, 2, 0, dw_gl_sz, dw_lo_sz, 1, dw_waitlist, &sync_event[1]);
//         err = clEnqueueNDRangeKernel(queue, bloom_2, 2, 0, half_gl_sz, half_lo_sz, 1, dw_waitlist, &sync_event[2]);
//         cl_event bloom_waitlist[1] = {sync_event[1]};
//         err = clEnqueueNDRangeKernel(queue, bloom_4, 2, 0, quarter_gl_sz, quarter_lo_sz, 1, bloom_waitlist, NULL);
//         err = clEnqueueNDRangeKernel(queue, bloom_8, 2, 0, eighth_gl_sz, eighth_lo_sz, 1, bloom_waitlist, NULL);
//         clEnqueueBarrier(queue);
//         // cl_event blend_waitlist[3] = {&sync_event[2], &sync_event[3], &sync_event[4]};
//         err = clEnqueueNDRangeKernel(queue, blend_bloom, 2, 0, blend_gl_sz, blend_lo_sz, 0, NULL, NULL);
//         clEnqueueBarrier(queue);
//     }
//     clEnqueueMarker(queue, &event[1]);

//     clFinish(queue);
//     // cl_event event[7];
//     // const int numIterations = 100;

//     // err = clEnqueueNDRangeKernel(queue, lens_blur, 2, 0, lb_gl_sz, lb_lo_sz, 0, NULL, NULL);

//     // clEnqueueMarker(queue, &event[0]);
//     // for (int i=0; i < numIterations; i++){
//     //     err = clEnqueueNDRangeKernel(queue, lens_blur, 2, 0, lb_gl_sz, lb_lo_sz, 0, NULL, NULL);
//     // }
//     // clEnqueueMarker(queue, &event[1]);
//     // for (int i=0; i < numIterations; i++){
//     //     err = clEnqueueNDRangeKernel(queue, downsample, 2, 0, dw_gl_sz, dw_lo_sz, 0, NULL, NULL);
//     // }
//     // clEnqueueMarker(queue, &event[2]);
//     // for (int i=0; i < numIterations; i++){    
//     //     err = clEnqueueNDRangeKernel(queue, bloom_2, 2, 0, half_gl_sz, half_lo_sz, 0, NULL, NULL);
//     // }
//     // clEnqueueMarker(queue, &event[3]);
//     // for (int i=0; i < numIterations; i++){    
//     //     err = clEnqueueNDRangeKernel(queue, bloom_4, 2, 0, quarter_gl_sz, quarter_lo_sz, 0, NULL, NULL);
//     // }
//     // clEnqueueMarker(queue, &event[4]);
//     // for (int i=0; i < numIterations; i++){    
//     //     err = clEnqueueNDRangeKernel(queue, bloom_8, 2, 0, eighth_gl_sz, eighth_lo_sz, 0, NULL, NULL);
//     // }
//     // clEnqueueMarker(queue, &event[5]);
//     // for (int i=0; i < numIterations; i++){    
//     //     err = clEnqueueNDRangeKernel(queue, blend_bloom, 2, 0, blend_gl_sz, blend_lo_sz, 0, NULL, NULL);
//     // }
//     // clEnqueueMarker(queue, &event[6]);

//     // clWaitForEvents(7, event);
//     clFinish(queue);
    
//     profile_single_kernel(event[0], event[1], numIterations, "Lens Blur + Half Downsample + Full Res Bloom");
//     // profile_single_kernel(event[1], event[2], numIterations, "Quarter and Eighth Downsample");
//     // profile_single_kernel(event[2], event[3], numIterations, "Half Res Bloom");
//     // profile_single_kernel(event[3], event[4], numIterations, "Quarter Res Bloom");
//     // profile_single_kernel(event[4], event[5], numIterations, "Eighth Res Bloom");
//     // profile_single_kernel(event[5], event[6], numIterations, "Blend Bloom");

//     // profile_single_kernel(event[0], event[6], numIterations, "FULL PIPELINE");

//     save_image_from_cl_buf(
//         lb_out_buf, queue, width, height, 3, "./lb_cmap_pvr_uint_v5.png");
    
//     save_image_from_cl_buf(
//         dw_2_buf, queue, width/2, height/2, 3, "./dw_2_uint_v5.png");

//     save_image_from_cl_buf(
//         dw_4_buf, queue, width/4, height/4, 3, "./dw_4_uint_v5.png");

//     save_image_from_cl_buf(
//         dw_8_buf, queue, width/8, height/8, 3, "./dw_8_uint_v5.png");

//     save_image_from_cl_buf(
//         bloom_out_2_buf, queue, width/2, height/2, 3, "./bloom_2_uint_v5.png");
    
//     save_image_from_cl_buf(
//         bloom_out_4_buf, queue, width/4, height/4, 3, "./bloom_4_uint_v5.png");

//     save_image_from_cl_buf(
//         bloom_out_8_buf, queue, width/8, height/8, 3, "./bloom_8_uint_v5.png");
    
//     save_image_from_cl_buf(
//         blend, queue, width, height, 3, "./blend_uint_v5.png");

//     return 0;
// }
