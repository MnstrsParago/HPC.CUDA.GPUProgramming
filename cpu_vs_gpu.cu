#include <stdio.h>
#include <stdlib.h>
#include <chrono>
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

// the same operation on the CPU
void cpuAdd(const float *A, const float *B, float *C, int N)
{
    for (int i = 0; i < N; ++i)
    {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char **argv)
{
    // N comes from the command line, default is 1000000
    int N = 1000000;
    if (argc > 1)
    {
        N = atoi(argv[1]);
    }

    size_t bytes = (size_t)N * sizeof(float);

    float *A = (float *)malloc(bytes);
    float *B = (float *)malloc(bytes);
    float *C_cpu = (float *)malloc(bytes);
    float *C_gpu = (float *)malloc(bytes);

    for (int i = 0; i < N; ++i)
    {
        A[i] = (float)i;
        B[i] = 2.0f * i;
    }

    const int RUNS = 10;

    // CPU: one warm up run, then the average of 10 runs
    cpuAdd(A, B, C_cpu, N);

    double cpuTotal = 0.0;
    for (int r = 0; r < RUNS; ++r)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        cpuAdd(A, B, C_cpu, N);
        auto t1 = std::chrono::high_resolution_clock::now();
        cpuTotal += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    double cpuTime = cpuTotal / RUNS;

    // GPU
    float *dev_A;
    float *dev_B;
    float *dev_C;

    cudaMalloc((void **)&dev_A, bytes);
    cudaMalloc((void **)&dev_B, bytes);
    cudaMalloc((void **)&dev_C, bytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // time of copying A and B from host to device
    cudaEventRecord(start);
    cudaMemcpy(dev_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float h2dTime = 0.0f;
    cudaEventElapsedTime(&h2dTime, start, stop);

    int threadsPerBlock = 256;
    int blocks =
        (N + threadsPerBlock - 1)
        / threadsPerBlock;

    // kernel: one warm up launch, then the average of 10 launches
    vectorAdd<<<blocks, threadsPerBlock>>>(dev_A, dev_B, dev_C, N);
    cudaDeviceSynchronize();

    float kernelTotal = 0.0f;
    for (int r = 0; r < RUNS; ++r)
    {
        cudaEventRecord(start);
        vectorAdd<<<blocks, threadsPerBlock>>>(dev_A, dev_B, dev_C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start, stop);
        kernelTotal += milliseconds;
    }
    float kernelTime = kernelTotal / RUNS;

    // time of copying C from device to host
    cudaEventRecord(start);
    cudaMemcpy(C_gpu, dev_C, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float d2hTime = 0.0f;
    cudaEventElapsedTime(&d2hTime, start, stop);

    // the GPU result must be the same as the CPU result
    int errors = 0;
    for (int i = 0; i < N; ++i)
    {
        if (C_gpu[i] != C_cpu[i])
        {
            errors++;
        }
    }

    printf("N = %d\n", N);
    printf("CPU time: %f ms\n", cpuTime);
    printf("GPU kernel time: %f ms\n", kernelTime);
    printf("Copy host to device: %f ms\n", h2dTime);
    printf("Copy device to host: %f ms\n", d2hTime);
    printf("GPU total (copies + kernel): %f ms\n",
           h2dTime + kernelTime + d2hTime);
    printf("Errors: %d\n\n", errors);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);

    free(A);
    free(B);
    free(C_cpu);
    free(C_gpu);

    return 0;
}