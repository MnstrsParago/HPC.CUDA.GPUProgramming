#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloKernel()
{
    printf("Block %d, Thread %d\n",
           blockIdx.x,
           threadIdx.x);
}

int main()
{
    printf("Hello from the CPU!\n");

    helloKernel<<<2, 4>>>();

    cudaDeviceSynchronize();

    return 0;
}