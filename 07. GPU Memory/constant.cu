
#include <stdio.h>
#include <stdlib.h>
#include <math_constants.h>

#define BLOCK_DIM 16
#define PI_STEP 256

__constant__ float c_sin_table[PI_STEP * 2];
__constant__ float c_cos_table[PI_STEP * 2];

__global__ void foo(float* out, const int width, const int height) {
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    int idx = idx_y * width + idx_x;
    
    if (idx_x < width && idx_y < height) {
        
        out[idx] = sin(idx * CUDART_PI_F / PI_STEP) * sin(idx * CUDART_PI_F / PI_STEP)
                 - cos(idx * CUDART_PI_F / PI_STEP) * cos(idx * CUDART_PI_F / PI_STEP);
    }
}

__global__ void fooConstant(float* out, const int width, const int height) {
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    int idx = idx_y * width + idx_x;
    
    if (idx_x < width && idx_y < height) {
        out[idx] = c_sin_table[idx % (PI_STEP * 2)] * c_sin_table[idx % (PI_STEP * 2)]
                 - c_cos_table[idx % (PI_STEP * 2)] * c_cos_table[idx % (PI_STEP * 2)];
    }
}

int main() {
    int width = 4096;
    int height = 4096;
    float sin_table[PI_STEP * 2];
    float cos_table[PI_STEP * 2];
    float *first, *second;
    float *d_first, *d_second;
    
    // CUDA Event Create to estimate elased time
    cudaEvent_t start, stop;
    float elapsed_gpu;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    for (int i = 0; i < PI_STEP * 2; i++) {
        sin_table[i] = sin(i * CUDART_PI_F / PI_STEP);
        cos_table[i] = cos(i * CUDART_PI_F / PI_STEP);
    }
    
    first = (float*)malloc(width * height * sizeof(float));
    second = (float*)malloc(width * height * sizeof(float));
    cudaMalloc((void**)&d_first, width * height * sizeof(float));
    cudaMalloc((void**)&d_second, width * height * sizeof(float));
    
    // Copy to Constant Memory
    cudaMemcpyToSymbol(c_sin_table, sin_table, PI_STEP * 2 * sizeof(float));
    cudaMemcpyToSymbol(c_cos_table, cos_table, PI_STEP * 2 * sizeof(float));
    
    dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
    dim3 gridDim((width + BLOCK_DIM - 1) / BLOCK_DIM, (height + BLOCK_DIM - 1) / BLOCK_DIM);
    
    cudaEventRecord(start, 0);
    foo<<<gridDim, blockDim>>>(d_first, width, height);
    
    // Estimate CUDA operation time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&elapsed_gpu, start, stop);
    printf("SGEMM CUDA Elapsed time (operation): %f ms\n", elapsed_gpu);
        
    cudaEventRecord(start, 0);
    fooConstant<<<gridDim, blockDim>>>(d_second, width, height);
    
    // Estimate CUDA operation time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&elapsed_gpu, start, stop);
    printf("SGEMM CUDA Elapsed time (constant): %f ms\n", elapsed_gpu);
    
    cudaMemcpy(first, d_first, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(second, d_second, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    
    float diff_sum = 0.f;
    for (int i = 0; i < width * height; i++) {
        if (first[i] - second[i] != 0.f) {
            diff_sum += first[i] - second[i];
        }
    }
    printf("diff_sum: %f\n", diff_sum);
    
    cudaFree(d_first);
    cudaFree(d_second);
    free(first);
    free(second);
    
    // finalize CUDA event
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}