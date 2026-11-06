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

int main()
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

    // allocate GPU memory
    cudaMalloc((void **)&dev_A, bytes);
    cudaMalloc((void **)&dev_B, bytes);
    cudaMalloc((void **)&dev_C, bytes);

    // copy A and B to GPU memory
    cudaMemcpy(dev_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;

    int blocks =
        (N + threadsPerBlock - 1)
        / threadsPerBlock;

    // launch vectorAdd kernel
    vectorAdd<<<blocks, threadsPerBlock>>>(dev_A, dev_B, dev_C, N);

    // Wait until the GPU finishes
    cudaDeviceSynchronize();

    // copy C back to the CPU
    cudaMemcpy(C, dev_C, bytes, cudaMemcpyDeviceToHost);

    printf("C[0] = %f\n", C[0]);
    printf("C[10] = %f\n", C[10]);
    printf("C[N-1] = %f\n", C[N - 1]);

    // check every element: C[i] must be 3i
    int errors = 0;
    for (int i = 0; i < N; ++i)
    {
        if (C[i] != 3.0f * i)
        {
            errors++;
        }
    }
    printf("Errors: %d\n", errors);

    // free device memory
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);

    free(A);
    free(B);
    free(C);

    return 0;
}