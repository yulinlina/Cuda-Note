#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include"mnist_helper.h"
#include"time_helper.h"
#include"utilcu.h"
#include"utils.h"

/* 
* Compile command *
* gcc -o softmax softmax.cu -lm *
* OR *
* nvcc -o softmax_gpu softmax.cu --run *
*/

int main(int argc, char *argv[]) {

	const char * train_images_file = "train-images-idx3-ubyte";
	const char * train_labels_file = "train-labels-idx1-ubyte";
	const char * test_images_file = "t10k-images-idx3-ubyte";
	const char * test_labels_file = "t10k-labels-idx1-ubyte";

	float *data_train, *data_test;
	int *labels_train, *labels_test;
	int number_of_samples_train, number_of_samples_test, rows, columns;

    /*
    * * * * Load training data  * * * *
    * data_train: float, 60000x784, each row represents a data sample *
    * labels_train: int, 60000, data labels, [1,2,3,4,5,...] *
    * number_of_samples_train: 60000 * 
    * rows: 28, number of pixel rows in an image; columns: 28, number of pixel columns in an image * 
    */
	get_dataset(train_images_file, train_labels_file, &data_train, &labels_train, &number_of_samples_train, &rows, &columns);
    scale_pixels(data_train, number_of_samples_train * rows * columns);
    printf("Training dataset: [%d %d %d] \n\n", number_of_samples_train, rows, columns);

	/*
    * * * * Load test data  * * * *
    * data_test: float, 10000x784, each row represents a data sample *
    * labels_test: int, 10000, data labels, [1,2,3,4,5,...] *
    * number_of_samples_test: 10000 * 
    * rows: 28, number of pixel rows in an image; columns: 28, number of pixel columns in an image * 
    */
	get_dataset(test_images_file, test_labels_file, &data_test, &labels_test, &number_of_samples_test, &rows, &columns);
	scale_pixels(data_test, number_of_samples_test * rows * columns);
    printf("\n Test dataset: [%d %d %d] \n", number_of_samples_test, rows, columns);
    
	/* 
    * Model initialization *
    * output = softmax(W*input) * 
    * W:10x784, input:784xn, output:10xn*
    */
    int W_rows = 10;
    int W_columns = 784;
    float* W = (float *)malloc(W_rows*W_columns*sizeof(float));
    weight_initialization(W, W_rows, W_columns);
    
    /* 
    * Training data, activation and gradient buffers *
    */
    float* activations_train = (float *)malloc(W_rows * number_of_samples_train * sizeof(float));
    float* data_transposed_train = (float *)malloc(rows * columns * number_of_samples_train * sizeof(float));
    float *delta = (float *)malloc(W_rows*number_of_samples_train*sizeof(float));
    float *W_grad = (float *)malloc(W_rows*W_columns*sizeof(float));   

    /* 
    * Test data and activation buffers *
    */
    float* activations_test = (float *)malloc(W_rows * number_of_samples_test * sizeof(float));
    float* data_transposed_test = (float *)malloc(rows * columns * number_of_samples_test * sizeof(float));
    
    /* 
    * Data sample visualization *
    */
    printf("label: %d\n", labels_train[0]);
    for(int i=0; i<rows; i++)
    {
        for(int j=0; j<columns; j++)
        {
            printf("%s ", *(data_train+i*columns+j + 0*rows*columns)>0?"#":" ");
        }
        printf("\n");
    }
    
    
    /* 
    * data: [n,784], one image per row *
    * data_transposed_train | data_transposed_test: [784,n], one image per column* 
    */
    float* y  = (float*)malloc(sizeof(float)*W_rows*number_of_samples_train);
    for(int j=0;j<number_of_samples_train;j++){
           int ground = labels_train[j];
        //   printf("ground: %d\n",ground);
           y[ground*number_of_samples_train+j]=1;
    }
    printf("example 1 of y: %f\n", y[5*number_of_samples_train+0]);
    printf("y :\n");
    print_matrix(y,10,number_of_samples_train);
    gpu_matrix_transpose(data_train, data_transposed_train, number_of_samples_train, rows*columns);
    gpu_matrix_transpose(data_test, data_transposed_test, number_of_samples_test, rows*columns);  
    /*
    printf("matrix transpose: \n"); 
    print_matrix(data_transposed_train,number_of_samples_train, rows*columns);
    */
    /* 
    * Training loop *
    */
    int epoch_num = 100; 
    float learning_rate = 0.01;
    float loss_train, acc_train, loss_test, acc_test;
    double time_begin;
    printf("GPU begin:\n");
    for(int epoch=0; epoch<epoch_num; epoch++)
    {
        time_begin = cpuSecond();
        /* 
        * Forward on training set *
        * data: [n,784], one image per row *
        * W:[10,784], data_transposed_train:[784,n], activations_train: [10,n] * 
        */
    
    // activations_train = W * data_transposed_train
    
        gpu_matrix_multiply(W,data_transposed_train,activations_train,W_rows,W_columns,W_columns,number_of_samples_train);
     
        gpu_softmax(activations_train,W_rows,number_of_samples_train);
      
        loss_train= gpu_cross_entropy_loss(activations_train, labels_train,W_rows,number_of_samples_train);
    
        acc_train=gpu_accuracy(activations_train,labels_train,W_rows,number_of_samples_train);

       gpu_matrix_multiply(W,data_transposed_test,activations_test,W_rows,W_columns,W_columns,number_of_samples_test);
 
       gpu_softmax(activations_test,W_rows,number_of_samples_test);

       loss_test= gpu_cross_entropy_loss(activations_test, labels_test,W_rows,number_of_samples_test);

        acc_test=gpu_accuracy(activations_test,labels_test,W_rows,number_of_samples_test);
    

        memset(delta, 0, W_rows*number_of_samples_train*sizeof(float));
        memset(W_grad, 0, W_rows*W_columns*sizeof(float));
        
        gpu_delta_compute(delta,activations_train,y, data_train, W_rows,  number_of_samples_train);
   
        gpu_matrix_add(delta,W,W_rows,W_columns,learning_rate);
      
        /* 
* Compile command *
* gcc -o softmax softmax.cu -lm *
* OR *
* nvcc -o softmax_gpu softmax.cu --run *
*/
        printf("[GPU][Epoch %d]: Train loss:%f, Train accuracy:%f;   Test loss:%f, Test accuracy:%f ; time cost: %lf s\n", epoch, loss_train, acc_train, loss_test, acc_test, cpuSecond()-time_begin);
  
    }
    
    return 0;
}
