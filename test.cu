#include <stdio.h>

__global__ void add(int *a, int *b, int *c) {
    *c = *a + *b;
}

int main(void) {
	// Allocate & initialize host data - run on the host
    int a, b, c;         // host copies of a, b, c
    a = 2;
    b = 7;

    int *d_a, *d_b, *d_c; // device copies of a, b, c
    // Allocate space for device copies of a, b, c
    int size = sizeof(int);
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);
	
    // Copy a & b from the host to the device
    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);
	
    // Launch add() kernel on GPU
    add<<<1,1>>>(d_a, d_b, d_c);
	
    // Copy result back to the host
    cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);
	
    // Cleanup
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}

