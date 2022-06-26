#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"
#include"error_check.h"
#include"time_helper.h"

// Todo
// Implement the cuda kernel function ***rgb_to_sepia_gpu***

void rgb_to_sepia_cpu(unsigned char *input_image, unsigned char *output_image, int width, int height, int channels)
{
    for(int row=0; row<height; row++)
    {
        for(int col=0; col<width; col++)
        {
            int offset = (row*width + col)*channels;
            unsigned char c1 = input_image[offset];
            unsigned char c2 = input_image[offset+1];
            unsigned char c3 = input_image[offset+2];

            *(output_image + offset) = (unsigned char)fmin((c1 * 0.393 + c2 * 0.769 + c3 * 0.189), 255.0);
			*(output_image + offset + 1) = (unsigned char)fmin((c1 * 0.349 + c2 * 0.686 + c3 * 0.168), 255.0);
			*(output_image + offset + 2) = (unsigned char)fmin((c1 * 0.272 + c2 * 0.534 + c3 * 0.131), 255.0);

            if(channels==4)
            {
                *(output_image + offset + 3) = input_image[offset + 3];
            }
        }
    }
}
__global__ void blur_gpu(unsigned char *g_input_image,unsigned char *g_output_image, int width, int height,int channels){
     int col = threadIdx.x+blockIdx.x*blockDim.x;
     int row =threadIdx.y+blockIdx.y*blockDim.y;
     if (col<width && row < height){

    	 for(int i=0;i<channels;i++){   // 
            int pixVal =0;     
    	    int pixels =0;
    		 int rgboffset = (col+row*width)*channels+i;// dim of (r,g,b)
			 for(int blur_row=-BLUR_SIZE;blur_row<=BLUR_SIZE;blur_row++){
				 for(int blur_col =-BLUR_SIZE;blur_col<=BLUR_SIZE;blur_col++){
					 int cur_col =col+blur_col;
					 int cur_row = row + blur_row;
					 int offset= (cur_row*width+cur_col)*channels+i;
					 if(cur_row>-1&&cur_row<height&&cur_col>-1&&cur_col<width){
						 pixVal+=g_input_image[offset];
						 pixels++;
					 } 
				 }
			 }
			 g_output_image[rgboffset]=(unsigned char)(pixVal/pixels);
    	 }
     }
}


int main(int argc, char *argv[])
{
    if(argc<4)
    {
        printf("Usage: command    input-image-name    output-image-name option   option(cpu/gpu)");
        return -1;
    }
    char *input_image_name = argv[1];
    char *output_image_name = argv[2];
    char *option = argv[3];

    int width, height, original_no_channels;
    int desired_no_channels = 0; // Pass 0 to load the image as is
    unsigned char *stbi_img = stbi_load(input_image_name, &width, &height, &original_no_channels, desired_no_channels);
    if(stbi_img==NULL){ printf("Error in loading the image.\n"); exit(1);}
    printf("Loaded image with a width of %dpx, a height of %dpx. The original image had %d channels, the loaded image has %d channels.\n", width, height, original_no_channels, desired_no_channels);

    int channels = original_no_channels;
    int img_mem_size = width * height * channels * sizeof(char);
    double begin;
    if(strcmp(option, "cpu")==0)
    {
        printf("Processing with CPU!\n");
        unsigned char *sepia_img = (unsigned char *)malloc(img_mem_size);
        if(sepia_img==NULL){  printf("Unable to allocate memory for the sepia image. \n");  exit(1);  }

        
        // Time stamp
		begin = cpuSecond();

		// CPU computation (for reference)
		rgb_to_sepia_cpu(stbi_img, sepia_img, width, height, channels);

        // Time stamp
		printf("Time cost [CPU]:%f s\n", cpuSecond()-begin);

        // Save to an image file
        stbi_write_jpg(output_image_name, width, height, channels, sepia_img, 100);

        free(sepia_img);
    }
    else if(strcmp(option, "gpu")==0) 
    {
        printf("Processing with GPU!\n");

       //  Todo: 1. Allocate memory on GPU
        unsigned char * g_input_img = NULL;
        unsigned char * g_output_img = NULL;
        cudaMalloc((void**)&g_input_img,img_mem_size);
        CHECK(cudaGetLastError());
        cudaMalloc((void**)&g_output_img,img_mem_size);
        //  Todo: 2. Copy data from host memory to device memory
        cudaMemcpy(g_input_img,stbi_img,img_mem_size,cudaMemcpyHostToDevice);
        //  Todo: 3. Call kernel function
        //        3.1 Declare block and grid sizes
        dim3 block_size(16,16,1);
        dim3 grid_size((width-1)/16+1,(height-1)/16+1,1);
        //        3.2 Record the time cost of GPU computation
        //  begin = cpuSecond();

        blur_gpu<<<grid_size,block_size>>>(g_input_img,g_output_img,width,height,channels);
        CHECK(cudaGetLastError());
        cudaDeviceSynchronize();
        //   printf("Time cost [GPU]:%f s\n", cpuSecond()-begin);
        //  Todo:  4. Copy data from device to host
        unsigned char *sepia_img_from_gpu = (unsigned char *)malloc(img_mem_size);
        cudaMemcpy(sepia_img_from_gpu,g_output_img,img_mem_size,cudaMemcpyDeviceToHost);
        //  Todo:  5. Save results as an image
            stbi_write_jpg(output_image_name, width, height, channels, sepia_img_from_gpu, 100);
        //  Todo:  6. Release host memory and device memory
            cudaFree(g_input_img);
            cudaFree(g_output_img);
            free(sepia_img_from_gpu);
        } 
    else
    {
        printf("Unexpected option (please use cpu/gpu) !\n");
    }   

    stbi_image_free(stbi_img);

    return 0;
}