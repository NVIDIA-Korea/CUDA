
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef enum TARGET {HOST, DEVICE} TARGET;

typedef struct {
    int width;
    int height;
    float *elements;
} Matrix;

__global__ void sgemm(Matrix A, Matrix B, Matrix C, 
                      const float alpha, const float beta, 
                      const int width, const int height) {
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    int idx = idx_y * width + idx_x;
    
    if (idx_x >= width || idx_y >= height)
        return;
    
    // TODO: Copy sgemm code from above you write
}

void InitMatrix(Matrix &mat, const int width, const int height, TARGET target = HOST);

int main(int argv, char* argc[]) {
    Matrix A, B, C_host, C_device;
    Matrix dA, dB, dC;
    const float alpha = 2.f;
    const float beta = .5f;
    const int width = 2048;
    const int height = 2048;
    float elapsed_gpu;
    double elapsed_cpu;
    
    // CUDA Event Create to estimate elased time
    cudaEvent_t start, stop;
    struct timespec begin, finish;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Initialize host matrix
    InitMatrix(A, width, height);
    InitMatrix(B, width, height);
    InitMatrix(C_device, width, height);

    // CUDA Memory Initialize
    InitMatrix(dA, width, height, DEVICE);
    InitMatrix(dB, width, height, DEVICE);
    InitMatrix(dC, width, height, DEVICE);
    
    // CUDA Operation
    cudaEventRecord(start, 0);
    clock_gettime(CLOCK_MONOTONIC, &begin);
    
    // Copy host data to the device (CUDA global memory)
    cudaMemcpy(dA.elements, A.elements, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB.elements, B.elements, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dC.elements, C_device.elements, width * height * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch GPU Kernel
    // TODO: Define your block size and check the performance
    dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    sgemm<<<gridDim, blockDim>>>(dA, dB, dC, alpha, beta, width, height);
    
    // Copy computation result from the Device the host memory
    cudaMemcpy(C_device.elements, dC.elements, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    clock_gettime(CLOCK_MONOTONIC, &finish);
    cudaEventRecord(stop, 0);
    
    // Estimate CUDA operation time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&elapsed_gpu, start, stop);
    printf("SGEMM CUDA Elapsed time: %f ms\n", elapsed_gpu);
    elapsed_cpu = (finish.tv_sec - begin.tv_sec);
    elapsed_cpu += (finish.tv_nsec - begin.tv_nsec) / 1000000000.0;
    printf("Host time: %f ms\n", elapsed_cpu * 1000);
    
    // finalize CUDA event
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Finalize
    cudaFree(dA.elements);
    cudaFree(dB.elements);
    cudaFree(dC.elements);
    
    free(A.elements);
    free(B.elements);
    //free(C_host.elements);
    free(C_device.elements);
    
    return 0;
}

void InitMatrix(Matrix &mat, const int width, const int height, TARGET target) {
    mat.width = width;
    mat.height = height;
    
    if (target == DEVICE) {
        cudaMalloc((void**)&mat.elements, width * height * sizeof(float));
    }
    else {
        mat.elements = (float*)malloc(width * height * sizeof(float));
    
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                mat.elements[row * width + col] = row * width + col * 0.001;
            }
        }
    }
}