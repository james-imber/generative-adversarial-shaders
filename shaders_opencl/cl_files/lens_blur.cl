// Copyright:    Copyright (c) Imagination Technologies Ltd 2023
// License:      MIT (refer to the accompanying LICENSE file)
// Author:       AI Research, Imagination Technologies Ltd
// Paper:        Generative Adversarial Shaders for Real-Time Realism Enhancement


#define KERNEL_SIZE 5
#define ARRAY_SIZE 3
#define IMG_H 1080
#define IMG_W 1920
#define TILE_SIZE_ROW 4
#define TILE_SIZE_COL 4
#define X_WORK_DIM 32
#define Y_WORK_DIM 8
#define RGB 3

#include "./cl_files/params_dump.h"

__constant float colour_transform[ARRAY_SIZE][ARRAY_SIZE] = COLOUR_TRANSFORM;

__constant float colour_translation[ARRAY_SIZE] = COLOUR_TRANSLATION;

__kernel void lens_blur(
    const __global uchar2 in_frame[3][IMG_H][IMG_W/2],
    __global uchar out_frame_2[3][IMG_H/2][IMG_W/2],
    __global ushort2 out_frame[3][IMG_H][IMG_W/2]
){
    const int2 gl_id = (int2)(get_global_id(0), get_global_id(1));
    const int2 lo_id = (int2)(get_local_id(0),  get_local_id(1));

    int x_offset = (gl_id.x/X_WORK_DIM)*TILE_SIZE_COL;
    int y_offset = (gl_id.y/Y_WORK_DIM)*TILE_SIZE_ROW;

    float kernel_h[KERNEL_SIZE] = KERNEL_X_LENS_BLUR;
    float kernel_v[KERNEL_SIZE] = KERNEL_Y_LENS_BLUR;
    // float kernel_h[KERNEL_SIZE] = {-0.06546026468276978, 0.402292937040329, 0.39708980917930603, 0.31741082668304443, -0.05133328214287758};
    // float kernel_v[KERNEL_SIZE] = {-0.01068652793765068, 0.18955382704734802, 0.6720903515815735, 0.117972731590271, 0.031069636344909668};

    __local float tile_shared[TILE_SIZE_ROW*Y_WORK_DIM][X_WORK_DIM + 1]; // [TILE_SIZE_ROW*Y_WORK_DIM][X_WORK_DIM + 1]
    __local float h_blur_shared[X_WORK_DIM + 1];

    float blurred[RGB][TILE_SIZE_ROW][TILE_SIZE_COL];
    float buf[TILE_SIZE_ROW][TILE_SIZE_COL];

    if (((gl_id.x*TILE_SIZE_COL - 6 - x_offset) > IMG_W) || ((gl_id.y*TILE_SIZE_ROW - 6 - y_offset) > IMG_H)){
        return;
    }

    int x_1 = clamp(lo_id.x - 1, 0, X_WORK_DIM - 1);
    int x_2 = clamp(lo_id.x + 1, 0, X_WORK_DIM - 1);

    int y_1 = clamp(lo_id.y*TILE_SIZE_ROW - 2, 0, TILE_SIZE_ROW*Y_WORK_DIM - 1);
    int y_2 = clamp(lo_id.y*TILE_SIZE_ROW - 1, 0, TILE_SIZE_ROW*Y_WORK_DIM - 1);
    int y_3 = clamp(lo_id.y*TILE_SIZE_ROW + 4, 0, TILE_SIZE_ROW*Y_WORK_DIM - 1);
    int y_4 = clamp(lo_id.y*TILE_SIZE_ROW + 5, 0, TILE_SIZE_ROW*Y_WORK_DIM - 1);

    /******************************* LENS BLUR *******************************/

    float conv_val[4];
    
    #pragma unroll
    for (int c=0; c < 3; c++){
        #pragma unroll
        for (int j=-6; j < TILE_SIZE_ROW - 6; j++){
            int y = clamp(gl_id.y*TILE_SIZE_ROW + j - y_offset, 0, IMG_H - 1);
            #pragma unroll
            for (int i=-6; i < TILE_SIZE_COL - 6; i+=2){
                int x = clamp(gl_id.x*TILE_SIZE_COL + i - x_offset, 0, IMG_W - 1);

                uchar2 val = in_frame[c][y][x/2];

                blurred[c][j + 6][i + 6] = val.x / 255.0;
                blurred[c][j + 6][i + 7] = val.y / 255.0;
            }
        }

        /**** HORIZONTAL CONVOLUTION PASS ****/

        #pragma unroll
        for (int j=0; j < TILE_SIZE_ROW; j++){
            tile_shared[lo_id.y*TILE_SIZE_ROW + j][lo_id.x] = blurred[c][j][0];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int i=0; i < TILE_SIZE_ROW; i++){
            conv_val[i] = tile_shared[lo_id.y*TILE_SIZE_ROW + i][x_2];

            buf[i][2] = blurred[c][i][0]*kernel_h[0] + blurred[c][i][1]*kernel_h[1] + 
                        blurred[c][i][2]*kernel_h[2] + blurred[c][i][3]*kernel_h[3] +
                        conv_val[i]*kernel_h[4];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int j=0; j < TILE_SIZE_ROW; j++){
            tile_shared[lo_id.y*TILE_SIZE_ROW + j][lo_id.x] = blurred[c][j][1];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int i=0; i < TILE_SIZE_ROW; i++){
            buf[i][3] = blurred[c][i][1]*kernel_h[0] + blurred[c][i][2]*kernel_h[1] + blurred[c][i][3]*kernel_h[2] +
                        conv_val[i]*kernel_h[3] +
                        tile_shared[lo_id.y*TILE_SIZE_ROW + i][x_2]*kernel_h[4];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int j=0; j < TILE_SIZE_ROW; j++){
            tile_shared[lo_id.y*TILE_SIZE_ROW + j][lo_id.x] = blurred[c][j][3];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int i=0; i < TILE_SIZE_ROW; i++){
            conv_val[i] = tile_shared[lo_id.y*TILE_SIZE_ROW + i][x_1];

            buf[i][1] = conv_val[i]*kernel_h[0] + 
                        blurred[c][i][0]*kernel_h[1] + blurred[c][i][1]*kernel_h[2] + 
                        blurred[c][i][2]*kernel_h[3] + blurred[c][i][3]*kernel_h[4];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int j=0; j < TILE_SIZE_ROW; j++){
            tile_shared[lo_id.y*TILE_SIZE_ROW + j][lo_id.x] = blurred[c][j][2];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int i=0; i < TILE_SIZE_ROW; i++){
            buf[i][0] = tile_shared[lo_id.y*TILE_SIZE_ROW + i][x_1]*kernel_h[0] + 
                        conv_val[i]*kernel_h[1] + 
                        blurred[c][i][0]*kernel_h[2] + blurred[c][i][1]*kernel_h[3] + blurred[c][i][2]*kernel_h[4];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        /**** VERTICAL CONVOLUTION PASS ****/

        #pragma unroll
        for (int i=0; i < TILE_SIZE_COL; i++){
            #pragma unroll
            for (int j=0; j < TILE_SIZE_ROW; j++){
                tile_shared[lo_id.y*TILE_SIZE_ROW + j][lo_id.x] = buf[j][i];
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            blurred[c][0][i] = tile_shared[y_1][lo_id.x]*kernel_v[0] + 
                               tile_shared[y_2][lo_id.x]*kernel_v[1] + 
                               buf[0][i]*kernel_v[2] + buf[1][i]*kernel_v[3] + buf[2][i]*kernel_v[4];

            blurred[c][1][i] = tile_shared[y_2][lo_id.x]*kernel_v[0] + 
                               buf[0][i]*kernel_v[1] + buf[1][i]*kernel_v[2] + 
                               buf[2][i]*kernel_v[3] + buf[3][i]*kernel_v[4];

            blurred[c][2][i] = buf[0][i]*kernel_v[0] + buf[1][i]*kernel_v[1] + 
                               buf[2][i]*kernel_v[2] + buf[3][i]*kernel_v[3] +
                               tile_shared[y_3][lo_id.x]*kernel_v[4];

            blurred[c][3][i] = buf[1][i]*kernel_v[0] + buf[2][i]*kernel_v[1] + buf[3][i]*kernel_v[2] +
                               tile_shared[y_3][lo_id.x]*kernel_v[3] +
                               tile_shared[y_4][lo_id.x]*kernel_v[4];

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    /***************************** COLOUR MAPPING ****************************/

    float R_transform[3] = {colour_transform[0][0], colour_transform[0][1], colour_transform[0][2]};
    float G_transform[3] = {colour_transform[1][0], colour_transform[1][1], colour_transform[1][2]};
    float B_transform[3] = {colour_transform[2][0], colour_transform[2][1], colour_transform[2][2]};
    float translation[3] = {colour_translation[0], colour_translation[1], colour_translation[2]};
    float R_acc, G_acc, B_acc;

    #pragma unroll
    for (int j=0; j < TILE_SIZE_ROW; j++){
        #pragma unroll
        for (int i=0; i < TILE_SIZE_COL; i++){
            R_acc = blurred[0][j][i]*R_transform[0] + blurred[1][j][i]*R_transform[1] + blurred[2][j][i]*R_transform[2];
            G_acc = blurred[0][j][i]*G_transform[0] + blurred[1][j][i]*G_transform[1] + blurred[2][j][i]*G_transform[2];
            B_acc = blurred[0][j][i]*B_transform[0] + blurred[1][j][i]*B_transform[1] + blurred[2][j][i]*B_transform[2];
            R_acc += translation[0];
            G_acc += translation[1];
            B_acc += translation[2];
            blurred[0][j][i] = R_acc;
            blurred[1][j][i] = G_acc;
            blurred[2][j][i] = B_acc;
        }
    }

    /**************************** HALF DOWNSAMPLE ****************************/

    for (int c=0; c < 3; c++){
        for (int j=0; j < TILE_SIZE_ROW/2; j++){
            for (int i=0; i < TILE_SIZE_COL/2; i++){
                buf[j][i] = (
                    blurred[c][j*2 + 0][i*2 + 0] +
                    blurred[c][j*2 + 0][i*2 + 1] +
                    blurred[c][j*2 + 1][i*2 + 0] +
                    blurred[c][j*2 + 1][i*2 + 1]
                ) * 0.25;
            }
        }

        for (int j=0; j < TILE_SIZE_ROW; j+=2){
            int y_i = clamp(lo_id.y*TILE_SIZE_ROW + j, 2, TILE_SIZE_ROW*Y_WORK_DIM - 3);
            y_i -= lo_id.y*TILE_SIZE_ROW;
            int y = clamp(gl_id.y*TILE_SIZE_ROW + y_i - 6 - y_offset, 0, IMG_H - 1);
            for (int i=0; i < TILE_SIZE_COL; i+=2){
                int x_i = clamp(lo_id.x*TILE_SIZE_COL + i, 2, TILE_SIZE_COL*X_WORK_DIM - 3);
                x_i -= lo_id.x*TILE_SIZE_COL;
                int x = clamp(gl_id.x*TILE_SIZE_COL + x_i - 6 - x_offset, 0, IMG_W - 1);

                out_frame_2[c][y/2][x/2] = (uchar)min(buf[y_i/2][x_i/2] * 255.0, 255.0);
            }
        }
    }

    /***************************** BLOOM FULL RES ****************************/

    float Y_weight[3] = Y_WEIGHT_FULL_RES;
    float sig_shift = SIG_SHIFT_FULL_RES;
    float sig_scale = SIG_SCALE_FULL_RES;
    
    float kernel_h_b[KERNEL_SIZE] = KERNEL_X_BLOOM_FULL_RES;
    float kernel_v_b[KERNEL_SIZE] = KERNEL_Y_BLOOM_FULL_RES;

    // kernel_h[0] = 0;
    // kernel_h[1] = 0;
    // kernel_h[2] = 1.1532641649246216;
    // kernel_h[3] = 0;
    // kernel_h[4] = 0;

    // kernel_h[0] = 3.3911422836660573e-43;
    // kernel_h[1] = 2.686375254190576e-11;
    // kernel_h[2] = 1.1532641649246216;
    // kernel_h[3] = 2.686375254190576e-11;
    // kernel_h[4] = 3.3911422836660573e-43;

    // kernel_v[0] = 0.12202727794647217;
    // kernel_v[1] = 0.27464398741722107;
    // kernel_v[2] = 0.35992154479026794;
    // kernel_v[3] = 0.27464398741722107;
    // kernel_v[4] = 0.12202727794647217;
    
    float acc;

    float luma_buf[TILE_SIZE_ROW][TILE_SIZE_COL];

    #pragma unroll
    for (int j=0; j < TILE_SIZE_ROW; j++){
        #pragma unroll
        for (int i=0; i < TILE_SIZE_COL; i++){
            acc = (blurred[0][j][i]*Y_weight[0] + blurred[1][j][i]*Y_weight[1] + blurred[2][j][i]*Y_weight[2] + 16)/255; 
            luma_buf[j][i] = 1/(1 + native_exp(-(sig_scale*(acc - sig_shift))));
        }
    }

    // #pragma unroll
    // for (int j=0; j < TILE_SIZE_ROW; j++){
    //     #pragma unroll
    //     for (int i=0; i < TILE_SIZE_COL; i++){
    //         acc = (blurred[0][j][i]*Y_weight[0] + blurred[1][j][i]*Y_weight[1] + blurred[2][j][i]*Y_weight[2] + 16)/255;
    //         luma_buf[j][i] = max(0, acc - sig_shift);
    //     }
    // }

    float luma[TILE_SIZE_COL];
    float bloom_conv_val;

    #pragma unroll
    for (int c=0; c < 3; c++){

        /**** HORIZONTAL CONVOLUTION PASS ****/
        #pragma unroll
        for (int j=0; j < TILE_SIZE_ROW; j++){
            #pragma unroll
            for (int i=0; i < TILE_SIZE_COL; i++){
                luma[i] = blurred[c][j][i] * luma_buf[j][i];
            }

            h_blur_shared[lo_id.x] = luma[0];

            barrier(CLK_LOCAL_MEM_FENCE);

            bloom_conv_val = h_blur_shared[x_2];

            buf[j][2] = luma[0]*kernel_h_b[0] + luma[1]*kernel_h_b[1] + 
                        luma[2]*kernel_h_b[2] + luma[3]*kernel_h_b[3] +
                        bloom_conv_val*kernel_h_b[4];

            barrier(CLK_LOCAL_MEM_FENCE);

            h_blur_shared[lo_id.x] = luma[1];

            barrier(CLK_LOCAL_MEM_FENCE);

            buf[j][3] = luma[1]*kernel_h_b[0] + luma[2]*kernel_h_b[1] + luma[3]*kernel_h_b[2] +
                        bloom_conv_val*kernel_h_b[3] +
                        h_blur_shared[x_2]*kernel_h_b[4];

            barrier(CLK_LOCAL_MEM_FENCE);

            h_blur_shared[lo_id.x] = luma[3];

            barrier(CLK_LOCAL_MEM_FENCE);

            bloom_conv_val = h_blur_shared[x_1];

            buf[j][1] = bloom_conv_val*kernel_h_b[0] + 
                        luma[0]*kernel_h_b[1] + luma[1]*kernel_h_b[2] + 
                        luma[2]*kernel_h_b[3] + luma[3]*kernel_h_b[4];

            barrier(CLK_LOCAL_MEM_FENCE);

            h_blur_shared[lo_id.x] = luma[2];

            barrier(CLK_LOCAL_MEM_FENCE);

            buf[j][0] = h_blur_shared[x_1]*kernel_h_b[0] + 
                        bloom_conv_val*kernel_h_b[1] + 
                        luma[0]*kernel_h_b[2] + luma[1]*kernel_h_b[3] + luma[2]*kernel_h_b[4];

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        /**** VERTICAL CONVOLUTION PASS ****/

        #pragma unroll
        for (int i=0; i < TILE_SIZE_COL; i++){
            #pragma unroll
            for (int j=0; j < TILE_SIZE_ROW; j++){
                tile_shared[lo_id.y*TILE_SIZE_ROW + j][lo_id.x] = buf[j][i];
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            blurred[c][0][i] += tile_shared[y_1][lo_id.x]*kernel_v_b[0] + 
                                tile_shared[y_2][lo_id.x]*kernel_v_b[1] + 
                                buf[0][i]*kernel_v_b[2] + buf[1][i]*kernel_v_b[3] + buf[2][i]*kernel_v_b[4];

            blurred[c][1][i] += tile_shared[y_2][lo_id.x]*kernel_v_b[0] + 
                                buf[0][i]*kernel_v_b[1] + buf[1][i]*kernel_v_b[2] + 
                                buf[2][i]*kernel_v_b[3] + buf[3][i]*kernel_v_b[4];

            blurred[c][2][i] += buf[0][i]*kernel_v_b[0] + buf[1][i]*kernel_v_b[1] + 
                                buf[2][i]*kernel_v_b[2] + buf[3][i]*kernel_v_b[3] +
                                tile_shared[y_3][lo_id.x]*kernel_v_b[4];

            blurred[c][3][i] += buf[1][i]*kernel_v_b[0] + buf[2][i]*kernel_v_b[1] + buf[3][i]*kernel_v_b[2] +
                                tile_shared[y_3][lo_id.x]*kernel_v_b[3] +
                                tile_shared[y_4][lo_id.x]*kernel_v_b[4];

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    ushort2 out_val;

    for (int c=0; c < RGB; c++){
        for (int j=0; j < TILE_SIZE_ROW; j++){
            int y_i = clamp(lo_id.y*TILE_SIZE_ROW + j, 2, TILE_SIZE_ROW*Y_WORK_DIM - 3);
            y_i -= lo_id.y*TILE_SIZE_ROW;
            int y = clamp(gl_id.y*TILE_SIZE_ROW + y_i - 6 - y_offset, 0, IMG_H - 1);
            for (int i=0; i < TILE_SIZE_COL; i+=2){
                int x_i = clamp(lo_id.x*TILE_SIZE_COL + i, 2, TILE_SIZE_COL*X_WORK_DIM - 4);
                x_i -= lo_id.x*TILE_SIZE_COL;
                int x = clamp(gl_id.x*TILE_SIZE_COL + x_i - 6 - x_offset, 0, IMG_W - 1);

                out_val.x = blurred[c][y_i][x_i + 0] * 255.0;
                out_val.y = blurred[c][y_i][x_i + 1] * 255.0;

                out_frame[c][y][x/2] = out_val;
            }
        }
    }

    return;
}
