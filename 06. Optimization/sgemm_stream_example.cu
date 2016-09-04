
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
    Matrix A[2], B[2], C[2], C_src;
    Matrix dA[2], dB[2], dC[2];
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
    
    // Initialize host matrix
    for (int i = 0; i < 2; i++) {
        InitMatrix(A[i], width, height, HOST, memtype);
        InitMatrix(B[i], width, height, HOST, memtype);
        InitMatrix(C[i], width, height, HOST, memtype);
    }
    InitMatrix(C_src, width, height, HOST, memtype);
    
    // CUDA Memory Initialize
    for (int i = 0; i < 2; i++) {
        InitMatrix(dA[i], width, height, DEVICE);
        InitMatrix(dB[i], width, height, DEVICE);
        InitMatrix(dC[i], width, height, DEVICE);
    }
    
    // Create CUDA Stream
    cudaStream_t stream[2];
    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);
    
    // CUDA Operation
    cudaEventRecord(start, 0);
    clock_gettime(CLOCK_MONOTONIC, &begin);
    
    int idx_stream = 0;
    for (int i = 0; i < 10; i++) {
        // Copy host data to the device (CUDA global memory)
        cudaMemcpyAsync(dA[idx_stream].elements, A[idx_stream].elements, width * height * sizeof(float), cudaMemcpyHostToDevice, stream[idx_stream]);
        cudaMemcpyAsync(dB[idx_stream].elements, B[idx_stream].elements, width * height * sizeof(float), cudaMemcpyHostToDevice, stream[idx_stream]);
        cudaMemcpyAsync(dC[idx_stream].elements, C_src.elements, width * height * sizeof(float), cudaMemcpyHostToDevice, stream[idx_stream]);
    
        // Launch GPU Kernel
        dim3 blockDim(16, 16);
        dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
        sgemm<<<gridDim, blockDim, 0, stream[idx_stream]>>>(dA[idx_stream], dB[idx_stream], dC[idx_stream], alpha, beta, width, height);
    
        // Copy computation result from the Device the host memory
        cudaMemcpyAsync(C[idx_stream].elements, dC[idx_stream].elements, width * height * sizeof(float), cudaMemcpyDeviceToHost, stream[idx_stream]);
        
        idx_stream = (idx_stream == 0) ? 1 : 0;
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
    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);
    
    // finalize CUDA event
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Finalize
    for (int i = 0; i < 2; i++) {
        cudaFree(dA[i].elements);
        cudaFree(dB[i].elements);
        cudaFree(dC[i].elements);
    
    
        if (memtype == NORMAL) {
            free(A[i].elements);
            free(B[i].elements);
            free(C[i].elements);
        }
        else {
            cudaFreeHost(A[i].elements);
            cudaFreeHost(B[i].elements);
            cudaFreeHost(C[i].elements);
        }
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