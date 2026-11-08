#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cuda_runtime.h>

// each thread calculates one element C[row][col]
__global__ void matrixMul(const float *A,
                          const float *B,
                          float *C,
                          int N)
{
    int col =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    int row =
        blockIdx.y * blockDim.y +
        threadIdx.y;

    if (row < N && col < N)
    {
        float sum = 0.0f;

        for (int k = 0; k < N; ++k)
        {
            sum += A[row * N + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

// the same calculation on the CPU
void cpuMatrixMul(const float *A, const float *B, float *C, int N)
{
    for (int row = 0; row < N; ++row)
    {
        for (int col = 0; col < N; ++col)
        {
            float sum = 0.0f;

            for (int k = 0; k < N; ++k)
            {
                sum += A[row * N + k] * B[k * N + col];
            }

            C[row * N + col] = sum;
        }
    }
}

void printMatrix(const char *name, const float *M, int N)
{
    printf("%s:\n", name);
    for (int row = 0; row < N; ++row)
    {
        for (int col = 0; col < N; ++col)
        {
            printf("%6.0f", M[row * N + col]);
        }
        printf("\n");
    }
}

int main(int argc, char **argv)
{
    // N comes from the command line, default is 4
    int N = 4;
    if (argc > 1)
    {
        N = atoi(argv[1]);
    }

    size_t bytes = (size_t)N * N * sizeof(float);

    float *A = (float *)malloc(bytes);
    float *B = (float *)malloc(bytes);
    float *C_cpu = (float *)malloc(bytes);
    float *C_gpu = (float *)malloc(bytes);

    // small whole numbers, so CPU and GPU results must be exactly equal
    for (int i = 0; i < N * N; ++i)
    {
        A[i] = (float)(i % 10);
        B[i] = (float)(i % 7);
    }

    float *dev_A;
    float *dev_B;
    float *dev_C;

    cudaMalloc((void **)&dev_A, bytes);
    cudaMalloc((void **)&dev_B, bytes);
    cudaMalloc((void **)&dev_C, bytes);

    cudaMemcpy(dev_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, bytes, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);

    dim3 blocks(
        (N + threads.x - 1) / threads.x,
        (N + threads.y - 1) / threads.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    matrixMul<<<blocks, threads>>>(dev_A, dev_B, dev_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpuTime = 0.0f;
    cudaEventElapsedTime(&gpuTime, start, stop);

    cudaMemcpy(C_gpu, dev_C, bytes, cudaMemcpyDeviceToHost);

    auto t0 = std::chrono::high_resolution_clock::now();
    cpuMatrixMul(A, B, C_cpu, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpuTime =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    // compare GPU and CPU results element by element
    int errors = 0;
    for (int i = 0; i < N * N; ++i)
    {
        if (C_gpu[i] != C_cpu[i])
        {
            errors++;
        }
    }

    printf("N = %d, grid: %d x %d blocks\n", N, blocks.x, blocks.y);

    if (N <= 8)
    {
        printMatrix("A", A, N);
        printMatrix("B", B, N);
        printMatrix("C (GPU)", C_gpu, N);
    }

    printf("CPU time: %f ms\n", cpuTime);
    printf("GPU kernel time: %f ms\n", gpuTime);
    printf("Errors (GPU vs CPU): %d\n\n", errors);

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