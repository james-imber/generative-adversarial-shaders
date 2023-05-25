// Copyright:    Copyright (c) Imagination Technologies Ltd 2023
// License:      MIT (refer to the accompanying LICENSE file)
// Author:       AI Research, Imagination Technologies Ltd
// Paper:        Generative Adversarial Shaders for Real-Time Realism Enhancement


#define ARRAY_SIZE 3
#define KERNEL_SIZE 5
#define RES_SCALES 4
#define LUMA_WEIGHTS 3
#define IMG_H 1080
#define IMG_W 1920
#define TILE_SIZE_ROW 4
#define TILE_SIZE_COL 4
#define X_WORK_DIM_DW 32
#define Y_WORK_DIM_DW 8
#define X_WORK_DIM 32
#define Y_WORK_DIM 8
#define X_WORK_DIM_UP 32
#define Y_WORK_DIM_UP 1
#define RGB 3
#define Y_WORK_DIM_BLEND 128

#include "./cl_files/params_dump.h"

__constant float gain_expanded = GAIN;

inline uint MWC64X(uint2 *state)
{
    enum { A=4294883355U};
    uint x=(*state).x, c=(*state).y;  // Unpack the state
    uint res=x^c;                     // Calculate the result
    uint hi=mul_hi(x,A);              // Step the RNG
    x=x*A+c;
    c=hi+(x<c);
    *state=(uint2)(x,c);               // Pack the state back up
    return res;                       // Return the next result
}

#define mwc64x_FLOAT_MULTI 2.3283064365386963e-10f
#define MWC64X_float(state) (MWC64X(state)*mwc64x_FLOAT_MULTI)

inline uint rand_xorshift(uint rng_state){
    rng_state ^= (rng_state << 13);
    rng_state ^= (rng_state >> 17);
    rng_state ^= (rng_state << 5);
    return rng_state;
}

// inline float2 normal_dist(uint2 *state){
//     float u1, u2, w, mag;
//     float2 rand;
//     u1 = max(MWC64X_float(state), 1e-8f);
//     u2 = MWC64X_float(state);
//     mag = sqrt (-2 * log(u1));
//     rand.x = mag*cos(2*M_PI_F*u2);
//     rand.y = mag*sin(2*M_PI_F*u2);
//     return rand;
// }

inline float2 normal_dist(uint2 *state){
    float u1, u2, w, mag;
    float2 rand;
    u1 = max(MWC64X_float(state), 1e-8f);
    u2 = MWC64X_float(state);
    mag = sqrt (-2 * log(u1));
    rand.x = mag*cos(2*M_PI_F*u2);
    rand.y = mag*sin(2*M_PI_F*u2);
    return rand;
}

__kernel void downsample(
    const __global uchar2 frame_size_2[3][IMG_H/2][IMG_W/4],
    __global uchar frame_size_4[3][IMG_H/4][IMG_W/4],
    __global uchar frame_size_8[3][IMG_H/8][IMG_W/8]
){
    const int2 gl_id = (int2)(get_global_id(0), get_global_id(1));
    const int2 lo_id = (int2)(get_local_id(0), get_local_id(1));

    uchar buf[2][2];
    __local uchar tile_shared[Y_WORK_DIM_DW][X_WORK_DIM_DW + 1];

    if ((gl_id.x*2 > (IMG_W/2)) || (gl_id.y*2 > (IMG_H/2))){
        return;
    }

    #pragma unroll
    for (int c=0; c < 3; c++){
        #pragma unroll
        for (int j=0; j < 2; j++){
            #pragma unroll
            for (int i=0; i < 2; i+=2){
                uchar2 val = frame_size_2[c][gl_id.y*2 + j][(gl_id.x*2 + i)/2];

                buf[j][i + 0] = val.x;
                buf[j][i + 1] = val.y;
            }
        }

        uchar out_4 = (buf[0][0] + buf[0][1] + buf[1][0] + buf[1][1]) * 0.25;

        frame_size_4[c][gl_id.y][gl_id.x] = out_4;

        tile_shared[lo_id.y][lo_id.x] = out_4;

        barrier(CLK_LOCAL_MEM_FENCE);

        if ((gl_id.y % 2 == 0) && (gl_id.x % 2 == 0)){
            ushort out_8 = (
                out_4 + 
                tile_shared[lo_id.y + 0][lo_id.x + 1] +
                tile_shared[lo_id.y + 1][lo_id.x + 0] +
                tile_shared[lo_id.y + 1][lo_id.x + 1]
            ) * 0.25;

            frame_size_8[c][gl_id.y/2][gl_id.x/2] = out_8;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    return;
}

inline void bloom_conv_h(
    __local float tile_shared[TILE_SIZE_ROW*Y_WORK_DIM][X_WORK_DIM + 1],
    float blurred[3][TILE_SIZE_ROW][TILE_SIZE_COL],
    float buf[TILE_SIZE_ROW][TILE_SIZE_COL],
    float conv_val[4], 
    const float kernel_h[KERNEL_SIZE],
    const int2 lo_id, int x_1, int x_2, int c
){

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

    return;
}

inline void bloom_conv_v(
    __local float tile_shared[TILE_SIZE_ROW*Y_WORK_DIM][X_WORK_DIM + 1],
    float blurred[3][TILE_SIZE_ROW][TILE_SIZE_COL],
    float buf[TILE_SIZE_ROW][TILE_SIZE_COL],
    const float kernel_v[KERNEL_SIZE],
    const int2 lo_id, int c, 
    int y_1, int y_2, int y_3, int y_4
){

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

    return;
}

__kernel void bloom_2(
    const __global uchar2 in_frame[3][IMG_H/2][IMG_W/4],
    __global uchar2 out_frame[3][IMG_H/2][IMG_W/4] 
){
    const int2 gl_id = (int2)(get_global_id(0), get_global_id(1));
    const int2 lo_id = (int2)(get_local_id(0), get_local_id(1));

    int x_offset = (gl_id.x/X_WORK_DIM)*TILE_SIZE_COL;
    int y_offset = (gl_id.y/Y_WORK_DIM)*TILE_SIZE_ROW;

    int2 size = (int2)(IMG_W/2, IMG_H/2);

    float blurred[3][TILE_SIZE_ROW][TILE_SIZE_COL];
    float buf[TILE_SIZE_ROW][TILE_SIZE_COL];
    __local float tile_shared[TILE_SIZE_ROW*Y_WORK_DIM][X_WORK_DIM + 1];
    
    float Y_weight[3] = Y_WEIGHT_HALF_RES;

    float sig_shift = SIG_SHIFT_HALF_RES;
    float sig_scale = SIG_SCALE_HALF_RES;

    const float kernel_h[KERNEL_SIZE] = KERNEL_X_BLOOM_HALF_RES;
    const float kernel_v[KERNEL_SIZE] = KERNEL_Y_BLOOM_HALF_RES;

    for (int c=0; c < 3; c++){
        #pragma unroll
        for (int j=-4; j < TILE_SIZE_ROW - 4; j++){
            int y = clamp(gl_id.y*TILE_SIZE_ROW + j - y_offset, 0, (IMG_H/2) - 1);
            #pragma unroll
            for (int i=-4; i < TILE_SIZE_COL - 4; i+=2){
                int x = clamp(gl_id.x*TILE_SIZE_COL + i - x_offset, 0, (IMG_W/2) - 1);

                uchar2 val = in_frame[c][y][x/2];

                blurred[c][j + 4][i + 4] = val.x / 255.0;
                blurred[c][j + 4][i + 5] = val.y / 255.0;
            }
        }
    }

    float acc;

    #pragma unroll
    for (int j=0; j < TILE_SIZE_ROW; j++){
        #pragma unroll
        for (int i=0; i < TILE_SIZE_COL; i++){
            acc = (blurred[0][j][i]*Y_weight[0] + blurred[1][j][i]*Y_weight[1] + blurred[2][j][i]*Y_weight[2] + 16)/255; 
            buf[j][i] = 1/(1 + native_exp(-(sig_scale*(acc - sig_shift))));
        }
    }

    for (int c=0; c < 3; c++){
        for (int j=0; j < TILE_SIZE_ROW; j++){
            for (int i=0; i < TILE_SIZE_COL; i++){
                blurred[c][j][i] *= buf[j][i];
            }
        }
    }

    int x_1 = clamp(lo_id.x - 1, 0, X_WORK_DIM - 1);
    int x_2 = clamp(lo_id.x + 1, 0, X_WORK_DIM - 1);

    int y_1 = clamp(lo_id.y*TILE_SIZE_ROW - 2, 0, TILE_SIZE_ROW*Y_WORK_DIM - 1);
    int y_2 = clamp(lo_id.y*TILE_SIZE_ROW - 1, 0, TILE_SIZE_ROW*Y_WORK_DIM - 1);
    int y_3 = clamp(lo_id.y*TILE_SIZE_ROW + 4, 0, TILE_SIZE_ROW*Y_WORK_DIM - 1);
    int y_4 = clamp(lo_id.y*TILE_SIZE_ROW + 5, 0, TILE_SIZE_ROW*Y_WORK_DIM - 1);

    float conv_val[4];

    for (int c=0; c < 3; c++){

        /**** HORIZONTAL CONVOLUTION PASS ****/

        bloom_conv_h(tile_shared, blurred, buf, conv_val, kernel_h, lo_id, x_1, x_2, c);

        /**** VERTICAL CONVOLUTION PASS ****/

        bloom_conv_v(tile_shared, blurred, buf, kernel_v, lo_id, c, y_1, y_2, y_3, y_4);
    }

    uchar2 out_val;

    for (int c=0; c < RGB; c++){
        for (int j=0; j < TILE_SIZE_ROW; j++){
            int y_i = clamp(lo_id.y*TILE_SIZE_ROW + j, 2, TILE_SIZE_ROW*Y_WORK_DIM - 3);
            y_i -= lo_id.y*TILE_SIZE_ROW;
            int y = clamp(gl_id.y*TILE_SIZE_ROW + y_i - 4 - y_offset, 0, (IMG_H/2) - 1);
            for (int i=0; i < TILE_SIZE_COL; i+=2){
                int x_i = clamp(lo_id.x*TILE_SIZE_COL + i, 2, TILE_SIZE_COL*X_WORK_DIM - 4);
                x_i -= lo_id.x*TILE_SIZE_COL;
                int x = clamp(gl_id.x*TILE_SIZE_COL + x_i - 4 - x_offset, 0, (IMG_W/2) - 1);

                out_val.x = blurred[c][y_i][x_i + 0] * 255.0;
                out_val.y = blurred[c][y_i][x_i + 1] * 255.0;

                out_frame[c][y][x/2] = out_val;
            }
        }
    }

    return;
}

__kernel void bloom_4(
    const __global uchar2 in_frame[3][IMG_H/4][IMG_W/8],
    __global uchar2 out_frame[3][IMG_H/4][IMG_W/8] 
){
    const int2 gl_id = (int2)(get_global_id(0), get_global_id(1));
    const int2 lo_id = (int2)(get_local_id(0), get_local_id(1));

    int x_offset = (gl_id.x/X_WORK_DIM)*TILE_SIZE_COL;
    int y_offset = (gl_id.y/Y_WORK_DIM)*TILE_SIZE_ROW;

    int2 size = (int2)(IMG_W/2, IMG_H/2);

    float blurred[3][TILE_SIZE_ROW][TILE_SIZE_COL];
    float buf[TILE_SIZE_ROW][TILE_SIZE_COL];
    __local float tile_shared[TILE_SIZE_ROW*Y_WORK_DIM][X_WORK_DIM + 1];
    
    float Y_weight[3] = Y_WEIGHT_QUARTER_RES;

    float sig_shift = SIG_SHIFT_QUARTER_RES;
    float sig_scale = SIG_SCALE_QUARTER_RES;

    float kernel_h[KERNEL_SIZE] = KERNEL_X_BLOOM_QUARTER_RES;
    float kernel_v[KERNEL_SIZE] = KERNEL_Y_BLOOM_QUARTER_RES;

    for (int c=0; c < 3; c++){
        #pragma unroll
        for (int j=-4; j < TILE_SIZE_ROW - 4; j++){
            int y = clamp(gl_id.y*TILE_SIZE_ROW + j - y_offset, 0, (IMG_H/4) - 1);
            #pragma unroll
            for (int i=-4; i < TILE_SIZE_COL - 4; i+=2){
                int x = clamp(gl_id.x*TILE_SIZE_COL + i - x_offset, 0, (IMG_W/4) - 1);

                uchar2 val = in_frame[c][y][x/2];

                blurred[c][j + 4][i + 4] = val.x / 255.0;
                blurred[c][j + 4][i + 5] = val.y / 255.0;
            }
        }
    }

    float acc;

    #pragma unroll
    for (int j=0; j < TILE_SIZE_ROW; j++){
        #pragma unroll
        for (int i=0; i < TILE_SIZE_COL; i++){
            acc = (blurred[0][j][i]*Y_weight[0] + blurred[1][j][i]*Y_weight[1] + blurred[2][j][i]*Y_weight[2] + 16)/255; 
            buf[j][i] = 1/(1 + native_exp(-(sig_scale*(acc - sig_shift))));
        }
    }

    for (int c=0; c < 3; c++){
        for (int j=0; j < TILE_SIZE_ROW; j++){
            for (int i=0; i < TILE_SIZE_COL; i++){
                blurred[c][j][i] *= buf[j][i];
            }
        }
    }

    int x_1 = clamp(lo_id.x - 1, 0, X_WORK_DIM - 1);
    int x_2 = clamp(lo_id.x + 1, 0, X_WORK_DIM - 1);

    int y_1 = clamp(lo_id.y*TILE_SIZE_ROW - 2, 0, TILE_SIZE_ROW*Y_WORK_DIM - 1);
    int y_2 = clamp(lo_id.y*TILE_SIZE_ROW - 1, 0, TILE_SIZE_ROW*Y_WORK_DIM - 1);
    int y_3 = clamp(lo_id.y*TILE_SIZE_ROW + 4, 0, TILE_SIZE_ROW*Y_WORK_DIM - 1);
    int y_4 = clamp(lo_id.y*TILE_SIZE_ROW + 5, 0, TILE_SIZE_ROW*Y_WORK_DIM - 1);

    float conv_val[4];

    for (int c=0; c < 3; c++){

        /**** HORIZONTAL CONVOLUTION PASS ****/

        bloom_conv_h(tile_shared, blurred, buf, conv_val, kernel_h, lo_id, x_1, x_2, c);

        /**** VERTICAL CONVOLUTION PASS ****/

        bloom_conv_v(tile_shared, blurred, buf, kernel_v, lo_id, c, y_1, y_2, y_3, y_4);
    }

    uchar2 out_val;

    for (int c=0; c < RGB; c++){
        for (int j=0; j < TILE_SIZE_ROW; j++){
            int y_i = clamp(lo_id.y*TILE_SIZE_ROW + j, 2, TILE_SIZE_ROW*Y_WORK_DIM - 3);
            y_i -= lo_id.y*TILE_SIZE_ROW;
            int y = clamp(gl_id.y*TILE_SIZE_ROW + y_i - 4 - y_offset, 0, (IMG_H/4) - 1);
            for (int i=0; i < TILE_SIZE_COL; i+=2){
                int x_i = clamp(lo_id.x*TILE_SIZE_COL + i, 2, TILE_SIZE_COL*X_WORK_DIM - 4);
                x_i -= lo_id.x*TILE_SIZE_COL;
                int x = clamp(gl_id.x*TILE_SIZE_COL + x_i - 4 - x_offset, 0, (IMG_W/4) - 1);

                out_val.x = blurred[c][y_i][x_i + 0] * 255.0;
                out_val.y = blurred[c][y_i][x_i + 1] * 255.0;

                out_frame[c][y][x/2] = out_val;
            }
        }
    }

    return;
}

__kernel void bloom_8(
    const __global uchar2 in_frame[3][IMG_H/8][IMG_W/16],
    __global uchar2 out_frame[3][IMG_H/8][IMG_W/16] 
){
    const int2 gl_id = (int2)(get_global_id(0), get_global_id(1));
    const int2 lo_id = (int2)(get_local_id(0), get_local_id(1));

    int x_offset = (gl_id.x/X_WORK_DIM)*TILE_SIZE_COL;
    int y_offset = (gl_id.y/Y_WORK_DIM)*TILE_SIZE_ROW;

    int2 size = (int2)(IMG_W/2, IMG_H/2);

    float blurred[3][TILE_SIZE_ROW][TILE_SIZE_COL];
    float buf[TILE_SIZE_ROW][TILE_SIZE_COL];
    __local float tile_shared[TILE_SIZE_ROW*Y_WORK_DIM][X_WORK_DIM + 1];
    
    float Y_weight[3] = Y_WEIGHT_EIGHTH_RES;

    float sig_shift = SIG_SHIFT_EIGHTH_RES;
    float sig_scale = SIG_SCALE_EIGHTH_RES;

    float kernel_h[KERNEL_SIZE] = KERNEL_X_BLOOM_EIGHTH_RES;
    float kernel_v[KERNEL_SIZE] = KERNEL_Y_BLOOM_EIGHTH_RES;

    for (int c=0; c < 3; c++){
        #pragma unroll
        for (int j=-4; j < TILE_SIZE_ROW - 4; j++){
            int y = clamp(gl_id.y*TILE_SIZE_ROW + j - y_offset, 0, (IMG_H/8) - 1);
            #pragma unroll
            for (int i=-4; i < TILE_SIZE_COL - 4; i+=2){
                int x = clamp(gl_id.x*TILE_SIZE_COL + i - x_offset, 0, (IMG_W/8) - 1);

                uchar2 val = in_frame[c][y][x/2];

                blurred[c][j + 4][i + 4] = val.x / 255.0;
                blurred[c][j + 4][i + 5] = val.y / 255.0;
            }
        }
    }

    float acc;

    #pragma unroll
    for (int j=0; j < TILE_SIZE_ROW; j++){
        #pragma unroll
        for (int i=0; i < TILE_SIZE_COL; i++){
            acc = (blurred[0][j][i]*Y_weight[0] + blurred[1][j][i]*Y_weight[1] + blurred[2][j][i]*Y_weight[2] + 16)/255; 
            buf[j][i] = 1/(1 + native_exp(-(sig_scale*(acc - sig_shift))));
        }
    }

    for (int c=0; c < 3; c++){
        for (int j=0; j < TILE_SIZE_ROW; j++){
            for (int i=0; i < TILE_SIZE_COL; i++){
                blurred[c][j][i] *= buf[j][i];
            }
        }
    }

    int x_1 = clamp(lo_id.x - 1, 0, X_WORK_DIM - 1);
    int x_2 = clamp(lo_id.x + 1, 0, X_WORK_DIM - 1);

    int y_1 = clamp(lo_id.y*TILE_SIZE_ROW - 2, 0, TILE_SIZE_ROW*Y_WORK_DIM - 1);
    int y_2 = clamp(lo_id.y*TILE_SIZE_ROW - 1, 0, TILE_SIZE_ROW*Y_WORK_DIM - 1);
    int y_3 = clamp(lo_id.y*TILE_SIZE_ROW + 4, 0, TILE_SIZE_ROW*Y_WORK_DIM - 1);
    int y_4 = clamp(lo_id.y*TILE_SIZE_ROW + 5, 0, TILE_SIZE_ROW*Y_WORK_DIM - 1);

    float conv_val[4];

    for (int c=0; c < 3; c++){

        /**** HORIZONTAL CONVOLUTION PASS ****/

        bloom_conv_h(tile_shared, blurred, buf, conv_val, kernel_h, lo_id, x_1, x_2, c);

        /**** VERTICAL CONVOLUTION PASS ****/

        bloom_conv_v(tile_shared, blurred, buf, kernel_v, lo_id, c, y_1, y_2, y_3, y_4);
    }

    uchar2 out_val;

    for (int c=0; c < RGB; c++){
        for (int j=0; j < TILE_SIZE_ROW; j++){
            int y_i = clamp(lo_id.y*TILE_SIZE_ROW + j, 2, TILE_SIZE_ROW*Y_WORK_DIM - 3);
            y_i -= lo_id.y*TILE_SIZE_ROW;
            int y = clamp(gl_id.y*TILE_SIZE_ROW + y_i - 4 - y_offset, 0, (IMG_H/8) - 1);
            for (int i=0; i < TILE_SIZE_COL; i+=2){
                int x_i = clamp(lo_id.x*TILE_SIZE_COL + i, 2, TILE_SIZE_COL*X_WORK_DIM - 4);
                x_i -= lo_id.x*TILE_SIZE_COL;
                int x = clamp(gl_id.x*TILE_SIZE_COL + x_i - 4 - x_offset, 0, (IMG_W/8) - 1);

                out_val.x = blurred[c][y_i][x_i + 0] * 255.0;
                out_val.y = blurred[c][y_i][x_i + 1] * 255.0;

                out_frame[c][y][x/2] = out_val;
            }
        }
    }

    return;
}

inline float cubic_convolution1(float x){
    return ((-0.75f + 2.0f) * x - (-0.75f + 3.0f)) * x * x + 1.0f;
}

inline float cubic_convolution2(float x){
    return ((-0.75f * x - 5.0f * -0.75f) * x + 8.0f * -0.75f) * x - 4.0f * -0.75f;
}

inline void bicubic_upsample_8(
    float src_tile[TILE_SIZE_ROW + 2][TILE_SIZE_COL + 2],
    const int2 gl_id,
    float dest[TILE_SIZE_ROW][TILE_SIZE_COL]
){

    for (int v=0; v < TILE_SIZE_ROW; v++){
        for (int u=0; u < TILE_SIZE_COL; u++){

            float2 src_idx = (float2)((gl_id.x*TILE_SIZE_COL + u + 0.5f)/8.0f - 0.5f, (gl_id.y*TILE_SIZE_ROW + v + 0.5f)/8.0f - 0.5f);
            int2 isrc_idx = convert_int2(src_idx); 

            float t_x = src_idx.x - isrc_idx.x;
            float t_y = src_idx.y - isrc_idx.y;

            float x_coeffs[4], y_coeffs[4];

            x_coeffs[0] = cubic_convolution2(t_x + 1);
            x_coeffs[1] = cubic_convolution1(t_x);
            x_coeffs[2] = cubic_convolution1(1.0 - t_x);
            x_coeffs[3] = cubic_convolution2((1.0 - t_x) + 1);

            y_coeffs[0] = cubic_convolution2(t_y + 1);
            y_coeffs[1] = cubic_convolution1(t_y);
            y_coeffs[2] = cubic_convolution1(1.0 - t_y);
            y_coeffs[3] = cubic_convolution2((1.0 - t_y) + 1);

            for (int j=0; j < TILE_SIZE_ROW; j++){
                for (int i=0; i < TILE_SIZE_COL; i++){
                    dest[v][u] += src_tile[j][i] * x_coeffs[i] * y_coeffs[j];
                }
            }
        }
    }

    return;
}

// inline void bicubic_upsample_8(
//     float src_tile[TILE_SIZE_ROW + 2][TILE_SIZE_COL + 2],
//     const int2 gl_id,
//     const int c,
//     float dest[TILE_SIZE_ROW][TILE_SIZE_COL]
// ){

//     float coeffs[8][4] = {
//         {-0.080750, 0.510559, 0.674011, -0.103821},
//         {-0.050354, 0.342712, 0.818420, -0.110779},
//         {-0.021423, 0.185120, 0.929138, -0.092834},
//         {-0.002747, 0.052429, 0.991516, -0.041199},
//         {-0.041199, 0.991516, 0.052429, -0.002747},
//         {-0.092834, 0.929138, 0.185120, -0.021423},
//         {-0.110779, 0.818420, 0.342712, -0.050354},
//         {-0.103821, 0.674011, 0.510559, -0.080750}
//     };

//     int odd_x = (gl_id.x % 2)*4;
//     int odd_y = (gl_id.y % 2)*4;

//     #pragma unroll
//     for (int v=0; v < TILE_SIZE_ROW; v++){
//         #pragma unroll
//         for (int u=0; u < TILE_SIZE_COL; u++){
//             #pragma unroll
//             for (int j=0; j < TILE_SIZE_ROW; j++){
//                 #pragma unroll
//                 for (int i=0; i < TILE_SIZE_COL; i++){
//                     // dest[v][u] += src_tile[j][i] * coeffs[u % 4][i] * coeffs[v % 4][j];
//                     dest[v][u] += src_tile[j][i] * coeffs[(odd_x + u) % 8][i] * coeffs[(odd_y + v) % 8][j];
//                 }
//             }
//         }
//     }

//     return;
// }

__kernel void blend_multiscale_bloom(
    const __global ushort4 full_size_frame[3][IMG_H][IMG_W/4],
    const __global uchar2 half_size_frame[3][IMG_H/2][IMG_W/4],
    const __global uchar quarter_size_frame[3][IMG_H/4][IMG_W/4],
    const __global uchar eighth_size_frame[3][IMG_H/8][IMG_W/8],
    __global uchar4 blend[3][IMG_H][IMG_W/4]
){
    int2 gl_id = (int2)(get_global_id(0), get_global_id(1));
    int2 lo_id = (int2)(get_local_id(0),  get_local_id(1));

    float exposure = 1.170825719833374;

    float out_tile[TILE_SIZE_ROW][TILE_SIZE_COL];
    float src_tile[TILE_SIZE_ROW + 2][TILE_SIZE_COL + 2];

    __local float src_shared[X_WORK_DIM*2 + 1];

    float gain = native_powr(2, gain_expanded);
    float sigma = SIGMA;
    float mu = -0.0105;

    uint2 state = (uint2)(rand_xorshift(rand_xorshift((gl_id.x + 1)*(gl_id.y + 1))), rand_xorshift(gl_id.y + 1));

    float2 start_idx;

    start_idx = (float2)((gl_id.x*TILE_SIZE_COL + 0.5f)/2.0f - 0.5f, (gl_id.y*TILE_SIZE_ROW + 0.5f)/2.0f - 0.5f);
    const int2 istart_idx_2 = convert_int2(start_idx);

    start_idx = (float2)((gl_id.x*TILE_SIZE_COL + 0.5f)/4.0f - 0.5f, (gl_id.y*TILE_SIZE_ROW + 0.5f)/4.0f - 0.5f);
    const int2 istart_idx_4 = convert_int2(start_idx);

    start_idx = (float2)((gl_id.x*TILE_SIZE_COL + 0.5f)/8.0f - 0.5f, (gl_id.y*TILE_SIZE_ROW + 0.5f)/8.0f - 0.5f);
    const int2 istart_idx_8 = convert_int2(start_idx);

    const float scale_2_coeffs[4] = {-0.03515625, 0.26171875, 0.87890625, -0.10546875};

    const float scale_4_coeffs[2][4] = {
        {-0.065918, 0.426270, 0.749512, -0.109863},
        {-0.010254, 0.114746, 0.967285, -0.071777},
    };

    // const float scale_8_coeffs[4][4] = {
    //     {-0.080750, 0.510559, 0.674011, -0.103821},
    //     {-0.050354, 0.342712, 0.818420, -0.110779},
    //     {-0.021423, 0.185120, 0.929138, -0.092834},
    //     {-0.002747, 0.052429, 0.991516, -0.041199}
    // };

    const int loc_offset = (lo_id.x % 2)*2;
    
    for (int c=0; c < 3; c++){

        ushort4 row_val;
        #pragma unroll
        for (int j=0; j < TILE_SIZE_ROW; j++){
            #pragma unroll
            for (int i=0; i < TILE_SIZE_COL; i+=4){
                row_val = full_size_frame[c][gl_id.y*TILE_SIZE_ROW + j][(gl_id.x*TILE_SIZE_COL + i)/4];

                out_tile[j][i + 0] = row_val.x / 255.0;
                out_tile[j][i + 1] = row_val.y / 255.0;
                out_tile[j][i + 2] = row_val.z / 255.0;
                out_tile[j][i + 3] = row_val.w / 255.0;
            }
        }

        float row[6];
        
        uchar2 sample_val;

        /*** BICUBIC UPSAMPLE x2 ***/
        
        #pragma unroll
        for (int j=0; j < TILE_SIZE_ROW + 2; j++){
            int y = clamp(istart_idx_2.y - 1 + j, 0, (IMG_H/2) - 1);

            #pragma unroll
            for (int i=0; i < TILE_SIZE_COL + 2; i+=2){
                int x = clamp(istart_idx_2.x - 1 + i, 0, (IMG_W/2) - 1);

                sample_val = half_size_frame[c][y][x/2]; 

                row[i + 0] = sample_val.x / 255.0;
                row[i + 1] = sample_val.y / 255.0;
            }
            
            src_tile[j][0] = row[0]*scale_2_coeffs[0];
            src_tile[j][1] = row[1]*scale_2_coeffs[3];
            src_tile[j][2] = row[1]*scale_2_coeffs[0];
            src_tile[j][3] = row[2]*scale_2_coeffs[3];

            #pragma unroll
            for (int i=1; i < TILE_SIZE_COL; i++){
                src_tile[j][0] += row[i + 0]*scale_2_coeffs[i];
                src_tile[j][1] += row[i + 1]*scale_2_coeffs[3 - i];
                src_tile[j][2] += row[i + 1]*scale_2_coeffs[i];
                src_tile[j][3] += row[i + 2]*scale_2_coeffs[3 - i];
            }
        }
        
        #pragma unroll
        for (int i=0; i < TILE_SIZE_COL; i++){
            out_tile[0][i] += src_tile[0][i]*scale_2_coeffs[0] + 
                              src_tile[1][i]*scale_2_coeffs[1] +
                              src_tile[2][i]*scale_2_coeffs[2] +
                              src_tile[3][i]*scale_2_coeffs[3];

            out_tile[1][i] += src_tile[1][i]*scale_2_coeffs[3] + 
                              src_tile[2][i]*scale_2_coeffs[2] +
                              src_tile[3][i]*scale_2_coeffs[1] +
                              src_tile[4][i]*scale_2_coeffs[0];

            out_tile[2][i] += src_tile[1][i]*scale_2_coeffs[0] + 
                              src_tile[2][i]*scale_2_coeffs[1] +
                              src_tile[3][i]*scale_2_coeffs[2] +
                              src_tile[4][i]*scale_2_coeffs[3];

            out_tile[3][i] += src_tile[2][i]*scale_2_coeffs[3] + 
                              src_tile[3][i]*scale_2_coeffs[2] +
                              src_tile[4][i]*scale_2_coeffs[1] +
                              src_tile[5][i]*scale_2_coeffs[0];
        }

        /*** BICUBIC UPSAMPLE x4 ***/

        for (int j=0; j < TILE_SIZE_ROW + 1; j++){
            int y = clamp(istart_idx_4.y - 1 + j, 0, (IMG_H/4) - 1);
            
            for (int i=0; i < TILE_SIZE_COL + 1; i++){

                int x = clamp(istart_idx_4.x - 1 + i, 0, (IMG_W/4) - 1);
                row[i] = quarter_size_frame[c][y][x] / 255.0;
            }

            src_tile[j][0] = row[0]*scale_4_coeffs[0][0];
            src_tile[j][1] = row[0]*scale_4_coeffs[1][0];
            src_tile[j][2] = row[1]*scale_4_coeffs[1][3];
            src_tile[j][3] = row[1]*scale_4_coeffs[0][3];

            #pragma unroll
            for (int i=1; i < TILE_SIZE_COL; i++){
                src_tile[j][0] += row[i + 0]*scale_4_coeffs[0][i];
                src_tile[j][1] += row[i + 0]*scale_4_coeffs[1][i];
                src_tile[j][2] += row[i + 1]*scale_4_coeffs[1][3 - i];
                src_tile[j][3] += row[i + 1]*scale_4_coeffs[0][3 - i];
            }
        }

        // #pragma unroll
        for (int i=0; i < TILE_SIZE_COL; i++){
            out_tile[0][i] += src_tile[0][i]*scale_4_coeffs[0][0] + 
                              src_tile[1][i]*scale_4_coeffs[0][1] +
                              src_tile[2][i]*scale_4_coeffs[0][2] +
                              src_tile[3][i]*scale_4_coeffs[0][3];

            out_tile[1][i] += src_tile[0][i]*scale_4_coeffs[1][0] + 
                              src_tile[1][i]*scale_4_coeffs[1][1] +
                              src_tile[2][i]*scale_4_coeffs[1][2] +
                              src_tile[3][i]*scale_4_coeffs[1][3];

            out_tile[2][i] += src_tile[1][i]*scale_4_coeffs[1][3] + 
                              src_tile[2][i]*scale_4_coeffs[1][2] +
                              src_tile[3][i]*scale_4_coeffs[1][1] +
                              src_tile[4][i]*scale_4_coeffs[1][0];

            out_tile[3][i] += src_tile[1][i]*scale_4_coeffs[0][3] + 
                              src_tile[2][i]*scale_4_coeffs[0][2] +
                              src_tile[3][i]*scale_4_coeffs[0][1] +
                              src_tile[4][i]*scale_4_coeffs[0][0];
        }

        /*** BICUBIC UPSAMPLE x8 ***/

        #pragma unroll
        for (int j=0; j < TILE_SIZE_ROW; j++){
            int y = clamp(istart_idx_8.y - 1 + j, 0, (IMG_H/8) - 1);
            // #pragma unroll
            for (int i=0; i < TILE_SIZE_COL; i++){
                int x = clamp(istart_idx_8.x - 1 + i, 0, (IMG_W/8) - 1);
                src_tile[j][i] = eighth_size_frame[c][y][x] / 255.0;
            }
        }

        bicubic_upsample_8(src_tile, gl_id, out_tile);

        float4 out_val;

        #pragma unroll
        for (int j=0; j < TILE_SIZE_ROW; j++){
            #pragma unroll
            for (int i=0; i < TILE_SIZE_COL; i+=4){
                float2 noise_1 = normal_dist(&state);
                float2 noise_2 = normal_dist(&state);
                float4 noise = (float4)(noise_1, noise_2);

                out_val = (float4)(out_tile[j][i + 0], out_tile[j][i + 1], out_tile[j][i + 2], out_tile[j][i + 3]);
                out_val = (float4)(1.0) - native_exp(-out_val * (float4)(exposure));

                out_val = out_val * out_val;
                out_val /= gain;
                out_val = native_sqrt(out_val)*max(noise, 0.0f) + out_val;
                out_val *= gain;
                out_val = native_sqrt(out_val);
                out_val += noise*sigma + mu;

                out_val *= 255.0;

                blend[c][gl_id.y*TILE_SIZE_ROW + j][(gl_id.x*TILE_SIZE_COL + i)/4] = convert_uchar4(out_val);
            }
        }
    }

    return;
}
