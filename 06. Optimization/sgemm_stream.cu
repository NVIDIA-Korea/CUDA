
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef enum TARGET {HOST, DEVICE} TARGET;
typedef enum MEMTYPE {NORMAL, PINNED} MEMTYPE;

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
    
    float value = 0.f;
    for (int e = 0; e < width; e++)
        value = alpha * A.elements[idx_y * width + e] * B.elements[e * width + idx_x];
    C.elements[idx] = value + beta * C.elements[idx];
}

void InitMatrix(Matrix &mat, const int width, const int height, TARGET target = HOST, MEMTYPE memtype = NORMAL);

int main(int argv, char* argc[]) {
    Matrix A, B, C, C_dst;
    Matrix dA, dB, dC[2];
    const float alpha = 2.f;
    const float beta = .5f;
    const int width = 2048;
    const int height = 2048;
    float elapsed_gpu;
    double elapsed_cpu;
    
    // Select Host memory type (NORMAL, PINNED)
    MEMTYPE memtype = PINNED;
    
    // CUDA Event Create to estimate elased time
    cudaEvent_t start, stop;
    struct timespec begin, finish;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Create CUDA Stream
    cudaStream_t stream[2];
    // TODO: Write CUDA Stream creation code

    /////////////
    
    // Initialize host matrix
    InitMatrix(A, width, height, HOST, memtype);
    InitMatrix(B, width, height, HOST, memtype);
    InitMatrix(C, width, height, HOST, memtype);
    InitMatrix(C_dst, width, height, HOST, memtype);

    // CUDA Memory Initialize
    InitMatrix(dA, width, height, DEVICE);
    InitMatrix(dB, width, height, DEVICE);
    InitMatrix(dC[0], width, height, DEVICE);
    InitMatrix(dC[1], width, height, DEVICE);
    
    // CUDA Operation
    cudaEventRecord(start, 0);
    clock_gettime(CLOCK_MONOTONIC, &begin);
    
    // Copy host data to the device (CUDA global memory)
    cudaMemcpyAsync(dA.elements, A.elements, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(dB.elements, B.elements, width * height * sizeof(float), cudaMemcpyHostToDevice);
    
    
    // Launch GPU Kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    for (int i = 0; i < 10; i++) {
        // TODO: Write CUDA Stream enabled operation
        cudaMemcpyAsync(dC[0].elements, C.elements, width * height * sizeof(float), cudaMemcpyHostToDevice);
        
        sgemm<<<gridDim, blockDim>>>(dA, dB, dC[0], alpha, beta, width, height);
        
        cudaMemcpyAsync(C_dst.elements, dC[0].elements, width * height * sizeof(float), cudaMemcpyDeviceToHost);
        //////////////
    }
        
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
    
    // finalize CUDA stream
    // TODO: Write CUDA Stream destory code
    
    /////////////////
    
    // finalize CUDA event
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Finalize
    cudaFree(dA.elements);
    cudaFree(dB.elements);
    cudaFree(dC[0].elements);
    cudaFree(dC[1].elements);
    
    if (memtype == NORMAL) {
        free(A.elements);
        free(B.elements);
        free(C.elements);
    }
    else {
        cudaFreeHost(A.elements);
        cudaFreeHost(B.elements);
        cudaFreeHost(C.elements);
    }
    
    return 0;
}

void InitMatrix(Matrix &mat, const int width, const int height, TARGET target, MEMTYPE memtype) {
    mat.width = width;
    mat.height = height;
    
    if (target == DEVICE) {
        cudaMalloc((void**)&mat.elements, width * height * sizeof(float));
    }
    else {
        if (memtype == NORMAL)
            mat.elements = (float*)malloc(width * height * sizeof(float));
        else
            cudaHostAlloc(&mat.elements, width * height * sizeof(float), cudaHostAllocDefault);
    
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                mat.elements[row * width + col] = row * width + col * 0.001;
            }
        }
    }
}