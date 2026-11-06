#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloKernel()
{
    printf("Hello from the GPU!\n");
}

int main()
{
    printf("Hello from the CPU!\n");

    helloKernel<<<1, 1>>>();

    cudaDeviceSynchronize();

    return 0;
}