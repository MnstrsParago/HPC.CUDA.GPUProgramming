# HPC Lab 8: GPU Programming with CUDA

Ayazbek Abdanur, BDA2406, Astana IT University

CUDA programs written and run in Google Colab on a Tesla T4 GPU (15 GB, CUDA 13.0).

## Files

| File | Part | What it does |
|---|---|---|
| `hello.cu` | 1 | First kernel, prints from the CPU and the GPU |
| `blocks_threads.cu` | 2 | Prints block and thread numbers for `<<<2, 4>>>` |
| `global_index.cu` | 3 | Calculates the global thread index for `<<<3, 4>>>` |
| `vector_add.cu` | 5 | Vector addition C = A + B for N = 1000000 |
| `vector_add_blocks.cu` | 6 | Same addition with 32, 64, 128, 256, 512 threads per block |
| `vector_add_timing.cu` | 7 | Kernel time for each block size with CUDA events |
| `cpu_vs_gpu.cu` | 8 | CPU time vs GPU kernel time and copy time |
| `matrix_add.cu` | 10 | 512 x 512 matrix addition with a 2D grid |
| `matrix_mul.cu` | Optional | Matrix multiplication, checked against the CPU |
| `Lab8_AbdanurAyazbek_BDA2406.ipynb` | All | Colab notebook with all code and outputs |
| `Lab8_AbdanurAyazbek_BDA2406.docx` | All | Report with tables and answers |

## Main results

| Experiment | Result |
|---|---|
| Vector addition, N = 10^7 | CPU 31.19 ms, GPU kernel 0.47 ms, copies 44.05 ms |
| Block size 32 vs 512 | 0.0688 ms vs 0.0517 ms, same result |
| Matrix multiplication, N = 512 | CPU 793.51 ms, GPU kernel 0.61 ms |

All programs gave 0 errors.

## How to run

In Google Colab: Runtime → Change runtime type → T4 GPU, then for each file:

```
!nvcc vector_add.cu -o vector_add
!./vector_add
```