#include<math.h>
#include"error_check.h"
#define TILE_DIM 32
#define BLOCK_SIZE 32
void gpu_matrix_add(float *input, float *output, int num_rows, int num_cols, float alpha);
__global__ void kernel_matrix_transpose(float *input, float *output, int num_rows, int num_cols){
    __shared__ float data[BLOCK_SIZE][BLOCK_SIZE+1];
    int input_col_id = blockIdx.x*blockDim.x+threadIdx.x;
    int input_row_id = blockIdx.y*blockDim.y+threadIdx.y;

	int block_x = blockIdx.x*blockDim.x;
	int block_y = blockIdx.y*blockDim.y;

	if(input_col_id<num_cols&&input_row_id<num_rows&&threadIdx.x<BLOCK_SIZE)
       data[threadIdx.y][threadIdx.x]=input[input_row_id*num_cols+input_col_id];
    __syncthreads();

    int output_col_id = block_y+threadIdx.x;
    int output_row_id =block_x+threadIdx.y;

	if(output_col_id<num_rows&&output_row_id<num_cols)
     output[output_row_id*num_rows+output_col_id]=data[threadIdx.x][threadIdx.y];
}
void gpu_matrix_transpose(float *input, float *output, int num_rows, int num_cols){
    float* input_d=NULL;
    float* output_d=NULL;
    int numBytes =sizeof(float)*num_rows*num_cols;

    cudaMalloc((void**)&input_d,numBytes);
    cudaMalloc((void**)&output_d,numBytes);
    CHECK(cudaMemcpy(input_d,input,numBytes,cudaMemcpyHostToDevice));

    dim3 grid((num_cols-1)/BLOCK_SIZE+1,(num_rows-1)/BLOCK_SIZE+1);
    dim3 block(BLOCK_SIZE,BLOCK_SIZE);
    kernel_matrix_transpose<<<grid,block>>>(input_d,output_d,num_rows,num_cols);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    cudaMemcpy(output,output_d,numBytes,cudaMemcpyDeviceToHost);
    cudaFree(input_d);
    cudaFree(output_d);

}
__global__ void kernel_matrix_multiply(float *M, float *N, float *P, int M_rows, int M_cols, int N_rows, int N_cols){
    __shared__ float Mds[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Nds[BLOCK_SIZE][BLOCK_SIZE];
    int bx= blockIdx.x,by=blockIdx.y;
    int tx =threadIdx.x,ty=threadIdx.y;

    int row =by*BLOCK_SIZE+ty;
    int col=bx*BLOCK_SIZE+tx;

    float Pvalue=0;
    for(int ph=0;ph<(M_cols-1)/BLOCK_SIZE+1;ph++){
        if(row<M_rows &&ph*BLOCK_SIZE+tx<M_cols)
            Mds[ty][tx]=M[row*M_cols+ph*BLOCK_SIZE+tx];
        if((ph*BLOCK_SIZE+ty)<N_rows &&col<N_cols)
            Nds[ty][tx]=N[(ph*BLOCK_SIZE+ty)*N_cols+col];
        __syncthreads();

        for(int k=0;k<BLOCK_SIZE;k++){
            Pvalue+=Mds[ty][k]*Nds[k][tx];
        }
         __syncthreads();
    }
    if(row<M_rows&&col<N_cols)
    {
        P[row*N_cols+col]=Pvalue;
    }
}
void gpu_matrix_multiply(float *M, float *N, float *P, int M_rows, int M_cols, int N_rows, int N_cols){
    float* M_d=NULL;
    float* N_d=NULL;
    float* P_d=NULL;
    cudaMalloc((void**)&M_d,sizeof(float)*M_cols*M_rows);
    cudaMalloc((void**)&N_d,sizeof(float)*N_cols*N_rows);
    cudaMalloc((void**)&P_d,sizeof(float)*N_cols*M_rows);
    cudaMemcpy(M_d,M,sizeof(float)*M_cols*M_rows,cudaMemcpyHostToDevice);
    cudaMemcpy(N_d,N,sizeof(float)*N_cols*N_rows,cudaMemcpyHostToDevice);

    dim3 grid((N_cols-1)/BLOCK_SIZE+1,(M_rows-1)/BLOCK_SIZE+1);
    dim3 block(BLOCK_SIZE,BLOCK_SIZE);
    kernel_matrix_multiply<<<grid,block>>>(M_d,N_d,P_d,M_rows,M_cols,N_rows,N_cols);
    CHECK(cudaDeviceSynchronize());

    cudaMemcpy(P,P_d,sizeof(float)*M_rows*N_cols,cudaMemcpyDeviceToHost);
    cudaFree(P_d);
    cudaFree(N_d);
    cudaFree(M_d);
}
__global__ void kernel_softmax(float* activations, int rows, int cols){
    __shared__ float active_value[BLOCK_SIZE][BLOCK_SIZE];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int col = bid*BLOCK_SIZE+tid;
    float max_active=0;
    float sum=0;
    for(int i =0;i<rows;i++){
        active_value[tid][i]=activations[i*cols+col];
        if(active_value[tid][i]>max_active) max_active=active_value[tid][i];
    }
    __syncthreads();
    for(int i=0;i<rows;i++){
        sum+=exp(active_value[tid][i]-max_active);
    }
    for(int i=0;i<rows;i++){
        activations[i*cols+col]=exp(active_value[tid][i]-max_active)/sum;
    }
   
}
void gpu_softmax(float* activations, int rows, int cols){
    float* activations_d=NULL;
    cudaMalloc((void**)&activations_d,sizeof(float)*cols*rows);
    cudaMemcpy(activations_d,activations,sizeof(float)*cols*rows,cudaMemcpyHostToDevice);
    int grid = (cols-1)/BLOCK_SIZE+1;
    kernel_softmax<<<grid,BLOCK_SIZE>>>(activations_d,rows,cols);
    CHECK(cudaDeviceSynchronize());

    cudaMemcpy(activations,activations_d,sizeof(float)*cols*rows,cudaMemcpyDeviceToHost);
    cudaFree(activations_d);

}
void __global__ kernel_cross_entropy_loss(float* activations, int* labels, float*loss ,int rows, int cols){
    int tid=threadIdx.x;
    int base = blockIdx.x*blockDim.x;
    int col = base+tid;
    for(int i=0;i<rows;i++){
        if(labels[col]==i){
            atomicAdd(loss,-log(activations[i*cols+i]+1e-8));
        }
    }
}
float gpu_cross_entropy_loss(float* activations, int* labels, int rows, int cols){
    /*
    activations : 10*n
    labels: n*1
    */
    float* activations_d=NULL;
    int* labels_d =NULL;
    float* loss_d=NULL;
    cudaMalloc((void**)&activations_d,sizeof(float)*cols*rows);
    cudaMalloc((void**)&labels_d,sizeof(int)*cols);
    cudaMalloc((void**)&loss_d,sizeof(float));

    cudaMemcpy(activations_d,activations,sizeof(float)*cols*rows,cudaMemcpyHostToDevice);
    cudaMemcpy(labels_d,labels,sizeof(int)*cols,cudaMemcpyHostToDevice);
    int grid = (cols-1)/BLOCK_SIZE+1;
    kernel_cross_entropy_loss<<<grid,BLOCK_SIZE>>>(activations_d,labels_d,loss_d,rows,cols);
    CHECK(cudaDeviceSynchronize());

    float loss =0;
    cudaMemcpy(&loss,loss_d,sizeof(float),cudaMemcpyDeviceToHost);
    cudaFree(activations_d);
    cudaFree(labels_d);
    cudaFree(loss_d);
    return loss/cols;
}
void __global__ kernel_accuracy(float* activations, int* labels, float* acc,int rows, int cols){
    int tid=threadIdx.x;
    int base = blockIdx.x*blockDim.x;
    int col = base+tid;
    float max_pred=0;
    float arg_max =0;
    for(int i=0;i<rows;i++){
        if(activations[i*cols+col]>max_pred){
            max_pred=activations[i*cols+col];
            arg_max=i;
        }
    }
    if(labels[col]==arg_max) atomicAdd(acc,1);
}
float gpu_accuracy(float* activations, int* labels, int rows, int cols){
    float* activations_d=NULL;
    int* labels_d =NULL;
    float* acc_d=NULL;
    cudaMalloc((void**)&activations_d,sizeof(float)*cols*rows);
    cudaMalloc((void**)&labels_d,sizeof(int)*cols);
    cudaMalloc((void**)&acc_d,sizeof(float));
    cudaMemcpy(activations_d,activations,sizeof(float)*cols*rows,cudaMemcpyHostToDevice);
    cudaMemcpy(labels_d,labels,sizeof(int)*cols,cudaMemcpyHostToDevice);

    int grid = (cols-1)/BLOCK_SIZE+1;
    kernel_accuracy<<<grid,BLOCK_SIZE>>>(activations_d,labels_d,acc_d,rows,cols);
    CHECK(cudaDeviceSynchronize());

    float acc=0;
    cudaMemcpy(&acc,acc_d,sizeof(float),cudaMemcpyDeviceToHost);
    cudaFree(activations_d);
    cudaFree(labels_d);
    cudaFree(acc_d);
    return acc/cols;
}
void __global__ kernel_delta_compute(float* delta, float* activations, int *labels, int W_rows, int number_of_samples){
    int tid=threadIdx.x;
    int base = blockIdx.x*blockDim.x;
    int col = base+tid;
    for(int i=0;i<W_rows;i++){
        if(i==labels[col]){
                delta[i*number_of_samples+col]=activations[i*number_of_samples+col]-1;
        }
        else{
             delta[i*number_of_samples+col]=activations[i*number_of_samples+col];
         }
    }
}
void gpu_delta_compute(float* delta, float* activations, float *labels, float*data, int W_rows, int number_of_samples){
    /*  delta 10*784
        labels 10*n 
       activations 10*n
       data : n*784
       delta = (activation-laebls)* data
       row :10
       col : n
     */
     
    int cols=number_of_samples,rows =W_rows;
    gpu_matrix_add(labels,activations,W_rows,number_of_samples,1);
    gpu_matrix_multiply(activations,data,delta,rows,cols,cols,784);


}
__global__ void kernel_matrix_add(float *input, float *output, int num_rows, int num_cols, float alpha){
    int tid=threadIdx.x;
    int base = blockIdx.x*blockDim.x;
    int col = base+tid;
    for(int i=0;i<num_rows;i++){
        output[i*num_cols+col]-=alpha*input[i*num_cols+col];
    }
}
void gpu_matrix_add(float *input, float *output, int num_rows, int num_cols, float alpha){
    float* input_d=NULL;
    float* output_d=NULL;
    cudaMalloc((void**)&input_d,sizeof(float)*num_cols*num_rows);
    cudaMalloc((void**)&output_d,sizeof(float)*num_cols*num_rows);
    cudaMemcpy(input_d,input,sizeof(float)*num_cols*num_rows,cudaMemcpyHostToDevice);
    cudaMemcpy(output_d,output,sizeof(float)*num_cols*num_rows,cudaMemcpyHostToDevice);
    int grid = (num_cols-1)/BLOCK_SIZE+1;
    kernel_matrix_add<<<grid,BLOCK_SIZE>>>(input_d,output_d,num_rows,num_cols,alpha);
    CHECK(cudaDeviceSynchronize());

    cudaMemcpy(output,output_d,sizeof(float)*num_cols*num_rows,cudaMemcpyDeviceToHost);
    cudaFree(input_d);
    cudaFree(output_d);
}
