#include "device_launch_parameters.h" 
#include <cuda_runtime.h> 
#include <stdlib.h> 
#include <stdio.h> 
#define N (2048 * 2048)
#define THREADS_PER_BLOCK 512



__global__ void dot(int *a, int *b, int *c){
    __shared__ int temp[THREADS_PER_BLOCK];
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    temp[threadIdx.x] = a[index] * b[index];
    
    __syncthreads();
   
    if(threadIdx.x == 0){
        int sum = 0;
        for(int i = 0 ; i < THREADS_PER_BLOCK ; i++){
        	sum += temp[i];
        }
        atomicAdd(c, sum);
    }
}

int main(void){
    int *a, *b, *c;    
    int *dev_a, *dev_b, *dev_c;
    int size = N * sizeof(int);
    
    // allocate host memories
    a = (int *)malloc(size);
    b = (int *)malloc(size);    
    c = (int *)malloc(sizeof(int));
    
    // allocate device memories
    cudaMalloc(&dev_a, size);
    cudaMalloc(&dev_b, size);
    cudaMalloc(&dev_c, sizeof(int));    
    
    // initialize variable
    for (int i = 0; i < size; ++i) { 
        a[i] = i; 
        b[i] = i; 
    } 
    
    // copy host memories to device memories
    cudaMemCpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemCpy(dev_b, b, size, cudaMemcpyHostToDevice);
    
    // run dot with N threads
    dot <<< N / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>> (dev_a, dev_b, dev_c);
    
    // copy device memories sum result(dev_c) to host memories(c) 
    cudaMemCpy(c, dev_c, sizeof(int), cudaMemCpyDeviceToHost);
    
    // print result of final sum
    printf("Final Sum: %d \n", *c);
    
    free(a); free(b); free(c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    
    return 0;
}
