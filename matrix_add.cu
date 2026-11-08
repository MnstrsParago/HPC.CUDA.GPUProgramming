#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void matrixAdd(
    const float *A,
    const float *B,
    float *C,
    int width,
    int height)
{
    int col =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    int row =
        blockIdx.y * blockDim.y +
        threadIdx.y;

    if (row < height && col < width)
    {
        int index = row * width + col;

        C[index] = A[index] + B[index];
    }
}

int main()
{
    const int width = 512;
    const int height = 512;
    const int total = width * height;

    size_t bytes = total * sizeof(float);

    float *A = (float *)malloc(bytes);
    float *B = (float *)malloc(bytes);
    float *C = (float *)malloc(bytes);

    for (int i = 0; i < total; ++i)
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

    dim3 threads(16, 16);

    dim3 blocks(
        (width + threads.x - 1) / threads.x,
        (height + threads.y - 1) / threads.y);

    matrixAdd<<<blocks, threads>>>(
        dev_A,
        dev_B,
        dev_C,
        width,
        height);

    cudaDeviceSynchronize();

    cudaMemcpy(C, dev_C, bytes, cudaMemcpyDeviceToHost);

    printf("Grid: %d x %d blocks, block: %d x %d threads\n",
           blocks.x, blocks.y, threads.x, threads.y);

    // print several elements: (row, col)
    int rows[] = {0, 0, 1, 100, 511};
    int cols[] = {0, 1, 0, 200, 511};

    for (int k = 0; k < 5; ++k)
    {
        int index = rows[k] * width + cols[k];
        printf("C[%d][%d] = %f (expected %f)\n",
               rows[k], cols[k], C[index], 3.0f * index);
    }

    // check every element: C[index] must be 3 * index
    int errors = 0;
    for (int i = 0; i < total; ++i)
    {
        if (C[i] != 3.0f * i)
        {
            errors++;
        }
    }
    printf("Errors: %d\n", errors);

    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);

    free(A);
    free(B);
    free(C);

    return 0;
}