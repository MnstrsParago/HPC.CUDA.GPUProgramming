#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float *A,
                          const float *B,
                          float *C,
                          int N)
{
    int i =
        threadIdx.x +
        blockIdx.x * blockDim.x;

    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char **argv)
{
    const int N = 1000000;

    size_t bytes = N * sizeof(float);

    float *A = (float *)malloc(bytes);
    float *B = (float *)malloc(bytes);
    float *C = (float *)malloc(bytes);

    for (int i = 0; i < N; ++i)
    {
        A[i] = (float)i;
        B[i] = 2.0f * i;
    }

    float *dev_A;
    float *dev_B;
    float *dev_C;

    cudaMalloc((void **)&dev_A, bytes);
    cudaMalloc((void **)&dev_B, bytes);
    cudaMalloc((void **)&dev_C, bytes);

    cudaMemcpy(dev_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, bytes, cudaMemcpyHostToDevice);

    // threads per block comes from the command line, default is 256
    int threadsPerBlock = 256;
    if (argc > 1)
    {
        threadsPerBlock = atoi(argv[1]);
    }

    int blocks =
        (N + threadsPerBlock - 1)
        / threadsPerBlock;

    vectorAdd<<<blocks, threadsPerBlock>>>(dev_A, dev_B, dev_C, N);

    cudaDeviceSynchronize();

    cudaMemcpy(C, dev_C, bytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N; ++i)
    {
        if (C[i] != 3.0f * i)
        {
            errors++;
        }
    }

    printf("Threads per block: %d, Blocks: %d\n", threadsPerBlock, blocks);
    printf("C[0] = %f, C[10] = %f, C[N-1] = %f, Errors: %d\n\n",
           C[0], C[10], C[N - 1], errors);

    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);

    free(A);
    free(B);
    free(C);

    return 0;
}