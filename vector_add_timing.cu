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

    int threadsPerBlock = 256;
    if (argc > 1)
    {
        threadsPerBlock = atoi(argv[1]);
    }

    int blocks =
        (N + threadsPerBlock - 1)
        / threadsPerBlock;

    // warm up launch, not measured (the first launch is always slower)
    vectorAdd<<<blocks, threadsPerBlock>>>(dev_A, dev_B, dev_C, N);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int RUNS = 10;
    float total = 0.0f;

    printf("Threads per block: %d\n", threadsPerBlock);
    printf("Times (ms):");

    for (int r = 0; r < RUNS; ++r)
    {
        cudaEventRecord(start);

        vectorAdd<<<blocks, threadsPerBlock>>>(
            dev_A, dev_B, dev_C, N);

        cudaEventRecord(stop);

        cudaEventSynchronize(stop);

        float milliseconds = 0.0f;

        cudaEventElapsedTime(
            &milliseconds, start, stop);

        printf(" %.4f", milliseconds);
        total += milliseconds;
    }

    printf("\nAverage kernel time: %f ms\n", total / RUNS);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(C, dev_C, bytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N; ++i)
    {
        if (C[i] != 3.0f * i)
        {
            errors++;
        }
    }

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