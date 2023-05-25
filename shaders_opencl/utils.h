// Copyright:    Copyright (c) Imagination Technologies Ltd 2023
// License:      MIT (refer to the accompanying LICENSE file)
// Author:       AI Research, Imagination Technologies Ltd
// Paper:        Generative Adversarial Shaders for Real-Time Realism Enhancement


#include "CL/cl.h"
#include <stdio.h>
#include <stdlib.h>
#include <png.h>
#include <errno.h>
#include <string.h>
#include <stdarg.h>
typedef unsigned char byte;

int ReadBmpFile(char* pFileName, byte* pBmpFile)
{
  FILE* fp;
  int fSize;
  int count;

  fp=fopen(pFileName,"rb");

  if(fp==NULL) return 1;

  fseek(fp,2,SEEK_SET);
  fread(&fSize,4,1,fp);

  // back to the beginning of the file
  fseek(fp,0,SEEK_SET);
  count=fread((void*)pBmpFile,1,fSize,fp);

  fclose(fp);

  if(count!=fSize) return 2;

  return 0;
}

int GetColorComponentFromBmp(byte* pBmp, byte* pRed, byte* pGreen, byte* pBlue)
{
  int nWidth, nHeight;
  byte* pBmpMat= pBmp + 14 + 40;
  int bWidth;

  int i,j;

  nWidth = *((int*) (pBmp+14+4));
  nHeight= *((int*) (pBmp+14+4+4));
  bWidth=( (nWidth*24+31)/32 )*4; // byte width of one line

  for(i=0; i<nHeight; i++)
    for(j=0; j<nWidth; j++)
    {
      pBlue[i*nWidth+j]   = pBmpMat[i*bWidth+3*j+0]; //blue 
      pGreen[i*nWidth+j] = pBmpMat[i*bWidth+3*j+1];  //green
      pRed[i*nWidth+j]  = pBmpMat[i*bWidth+3*j+2];   //red
    }

  return 0;
}

float* read_point_cloud(char* file_name, int *len){
    FILE *ptr; 

    ptr = fopen(file_name, "rb");

    fseek(ptr, 0, SEEK_END);
    int length = ftell(ptr);
    *len = length;
    rewind(ptr);

    float *buffer;
    buffer = (float*)malloc((length/4)*sizeof(float));

    fread(buffer, sizeof(float), (length/4), ptr);

    fclose(ptr);

    return buffer;
}

static void fatal_error(const char * message, ...)
{
    va_list args;
    va_start (args, message);
    vfprintf (stderr, message, args);
    va_end (args);
    exit (EXIT_FAILURE);
}

float* read_png_image(const char* png_file, int *h, int *w){
    png_structp png_ptr;
    png_infop info_ptr;

    FILE *fp;

    png_uint_32 width;
    png_uint_32 height;

    int bit_depth;
    int color_type;
    int interlace_method;
    int compression_method;
    int filter_method;
    png_bytepp rows;

    fp = fopen (png_file, "rb");

    if (! fp) {
	fatal_error ("Cannot open '%s': %s\n", png_file, strerror (errno));
    }
    png_ptr = png_create_read_struct (PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (! png_ptr) {
	fatal_error ("Cannot create PNG read structure");
    }
    info_ptr = png_create_info_struct (png_ptr);
    if (! png_ptr) {
	fatal_error ("Cannot create PNG info structure");
    }

    png_init_io (png_ptr, fp);
    png_read_png (png_ptr, info_ptr, 0, 0);
    png_get_IHDR (png_ptr, info_ptr, & width, & height, & bit_depth,
		  & color_type, & interlace_method, & compression_method,
		  & filter_method);
    rows = png_get_rows (png_ptr, info_ptr);
    // printf ("Width is %d, height is %d\n", width, height);

    int rowbytes;
    rowbytes = png_get_rowbytes (png_ptr, info_ptr);
    // printf ("Row bytes = %d\n", rowbytes);

    float* img_array = (float*)malloc(height*rowbytes*sizeof(float));

    for (uint j=0; j < height; j++){
        png_bytep row;
        row = rows[j];
        for (int i=0; i < rowbytes; i++){
            img_array[j*rowbytes + i] = row[i] / 255.0;
        }
    }

    fclose(fp);

    *h = height;
    *w = width;

    return img_array;
}

int write_png_file(
    float *img_array, 
    const char *file_path,
    int height,
    int width
){
    
    FILE *fp;
    png_structp png_ptr = NULL; 
    png_infop info_ptr = NULL;
    // size_t x, y;
    png_byte ** row_pointers = NULL;
    int status = -1;
    // int pixel_size = 3;
    int depth = 8;

    fp = fopen(file_path, "wb");

    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    info_ptr = png_create_info_struct(png_ptr);

    png_set_IHDR(
        png_ptr,
        info_ptr,
        width,
        height,
        depth,
        PNG_COLOR_TYPE_RGB,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT
    );

    row_pointers = png_malloc(png_ptr, height * sizeof(png_byte*));

    for (int j=0; j < height; j++){
        png_byte *row = png_malloc(png_ptr, sizeof(uint8_t) * width * 3);
        row_pointers[j] = row;
        for (int i=0; i < width*3; i++){
            if (img_array[j*width*3 + i] >= 1){
                row[i] = 255;
            } else if (img_array[j*width*3 + i] < 0){
                row[i] = 0;
            } else {
                row[i] = img_array[j*width*3 + i] * 255;
            }
        }
    }

    png_init_io (png_ptr, fp);
    png_set_rows (png_ptr, info_ptr, row_pointers);
    png_write_png (png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);

    status = 0;
    
    for (int y = 0; y < height; y++) {
        png_free (png_ptr, row_pointers[y]);
    }
    png_free (png_ptr, row_pointers);

    fclose (fp);

    return(status);
}

// int write_png_file_uint(
//     unsigned char *img_array, 
//     const char *file_path,
//     int height,
//     int width
// ){
    
//     FILE *fp;
//     png_structp png_ptr = NULL; 
//     png_infop info_ptr = NULL;
//     // size_t x, y;
//     png_byte ** row_pointers = NULL;
//     int status = -1;
//     // int pixel_size = 3;
//     int depth = 8;

//     fp = fopen(file_path, "wb");

//     png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
//     info_ptr = png_create_info_struct(png_ptr);

//     png_set_IHDR(
//         png_ptr,
//         info_ptr,
//         width,
//         height,
//         depth,
//         PNG_COLOR_TYPE_RGB,
//         PNG_INTERLACE_NONE,
//         PNG_COMPRESSION_TYPE_DEFAULT,
//         PNG_FILTER_TYPE_DEFAULT
//     );

//     row_pointers = png_malloc(png_ptr, height * sizeof(png_byte*));

//     for (int j=0; j < height; j++){
//         png_byte *row = png_malloc(png_ptr, sizeof(uint8_t) * width * 3);
//         row_pointers[j] = row;
//         for (int i=0; i < width*3; i++){
//             // if (img_array[j*width*3 + i] >= 1){
//             //     row[i] = 255;
//             // } else if (img_array[j*width*3 + i] < 0){
//             //     row[i] = 0;
//             // } else {
//             //     row[i] = img_array[j*width*3 + i] * 255;
//             // }
//             row[i] = img_array[j*width*3 + i];
//         }
//     }

//     png_init_io (png_ptr, fp);
//     png_set_rows (png_ptr, info_ptr, row_pointers);
//     png_write_png (png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);

//     status = 0;
    
//     for (int y = 0; y < height; y++) {
//         png_free (png_ptr, row_pointers[y]);
//     }
//     png_free (png_ptr, row_pointers);

//     fclose (fp);

//     return(status);
// }

unsigned char* convert_to_uint(float* img_buffer, int width, int height, int channels){

    unsigned char* img_buffer_uint = (unsigned char*)malloc(sizeof(unsigned char)*width*height*3);

    for (int c=0; c < channels; c++){
        for (int j=0; j < height; j++){
            for (int i=0; i < width; i++){
                img_buffer_uint[c*height*width + j*width + i] = img_buffer[c*height*width + j*width + i]*255;
            }
        }
    }

    return img_buffer_uint;
}

float* convert_to_float(unsigned char* img_buffer_uint, int width, int height, int channels){

    float* img_buffer = (float*)malloc(sizeof(float)*width*height*3);

    for (int c=0; c < channels; c++){
        for (int j=0; j < height; j++){
            for (int i=0; i < width; i++){
                img_buffer[c*height*width + j*width + i] = img_buffer_uint[c*height*width + j*width + i] / 255.0;
            }
        }
    }

    return img_buffer;
}

void channel_first_reorder(float* img_buffer, int width, int height){
    float* R = (float*)malloc(sizeof(float)*width*height);
    float* G = (float*)malloc(sizeof(float)*width*height);
    float* B = (float*)malloc(sizeof(float)*width*height);

    for(int j=0; j < height; j++){
        for (int i=0; i < width; i++){
            R[j*width + i] = img_buffer[j*width*3 + i*3 + 0];
            G[j*width + i] = img_buffer[j*width*3 + i*3 + 1];
            B[j*width + i] = img_buffer[j*width*3 + i*3 + 2];
        }
    }

    for (int j=0; j < height; j++){
        for (int i=0; i < width; i++){
            img_buffer[j*width + i] = R[j*width + i];
            img_buffer[(j + height)*width + i] = G[j*width + i];
            img_buffer[(j + height*2)*width + i] = B[j*width + i];
        }
    }

    free(R);
    free(G);
    free(B);
}

void channel_last_reorder(float* img_buffer, int width, int height){
    float* R = (float*)malloc(sizeof(float)*width*height);
    float* G = (float*)malloc(sizeof(float)*width*height);
    float* B = (float*)malloc(sizeof(float)*width*height);

    for(int j=0; j < height; j++){
        for (int i=0; i < width; i++){
            R[j*width + i] = img_buffer[j*width + i];
            G[j*width + i] = img_buffer[(j + height)*width + i];
            B[j*width + i] = img_buffer[(j + height*2)*width + i];
        }
    }

    for (int j=0; j < height; j++){
        for (int i=0; i < width; i++){
            img_buffer[j*width*3 + i*3 + 0] = R[j*width + i];
            img_buffer[j*width*3 + i*3 + 1] = G[j*width + i];
            img_buffer[j*width*3 + i*3 + 2] = B[j*width + i];
        }
    }

    free(R);
    free(G);
    free(B);
}

float randn(){
    double u1, u2, w, mult;
    static double x1, x2;
    static int call = 0;
    
    if (call == 1){
        call = !call;
        return (float) x2;
    }

    do
    {
        u1 = -2 + ((double) rand () / (RAND_MAX)) * 4;
        u2 = -2 + ((double) rand () / (RAND_MAX)) * 4;
        w = pow(u1, 2) + pow(u2, 2);
    } 
    while (w >= 1 || w == 0);
    
    mult = sqrt ((-2 * log (w)) / w);
    x1 = u1 * mult;
    x2 = u2 * mult;

    call = !call;

    return (float) x1;
}


float* white_noise(float sigma, float mu, int w, int h){
    float* array = (float*)malloc(w*h*3*sizeof(float));

    for (int i=0; i < w*h*3; i++){
        array[i] = mu + sigma*randn();
    }

    return array;
}

float* poisson_noise(int w, int h){
    float* array = (float*)malloc(w*h*3*sizeof(float));

    for (int i=0; i < w*h*3; i++){
        float num = randn();
        if (num < 0){
            array[i] = 0;
        }
    }

    return array;
}

void write_output_float_bin(
    cl_command_queue queue,
    cl_mem ocl_buf,
    char* filename,
    size_t size
){
    float *buffer = (float*)malloc(size);
    clEnqueueReadBuffer(queue, ocl_buf, CL_TRUE, 0, size, buffer, 0, NULL, NULL);

    FILE *fptr;

    fptr = fopen(filename, "wb");
    if (!fptr){
        printf("%s\n", strerror(errno));
        exit(1);
    }

    fwrite(buffer, size, 1, fptr);
    fclose(fptr);

    free(buffer);

    return;
}
