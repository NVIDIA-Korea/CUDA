
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

float* get_vector(int n_size, float seed = 0.0) {
    // buffer create
    float* p_vector = (float*)malloc(n_size * sizeof(float));
    
    // initialize vector
    if (seed != 0.0) {
        for (int i = 0; i < n_size; i++) {
            p_vector[i] = seed * i;
        }
    }
    
    return p_vector;
}

void check_result(float* py, float* py_cuda, int n_size) {
    float compare = 0.0;
    for (int i = 0; i < n_size; i++) {
        compare += py[i] - py_cuda[i];
    }
    printf("Result: %f\n", compare);
}

// CPU 연산
void saxpy(float* py, float* px, float alpha, int n_size) {
    for (int i = 0; i < n_size; i++) {
        py[i] = alpha * px[i] + py[i];
    }
}

// CUDA Kernel function
__global__ 
void d_saxpy(float* d_y, float* d_x, float alpha, int n_size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    d_y[idx] = alpha * d_x[idx] + d_y[idx];
}

int main() {
    float *px, *py, *py_cuda;
    int n_size = 65536;
    
    px = get_vector(n_size, 0.01);
    py = get_vector(n_size, 0.05);
    py_cuda = get_vector(n_size);
    
    // Step 1. Create GPU memory
    float *d_x, *d_y;
    cudaMalloc((void**)&d_x, n_size * sizeof(float));
    cudaMalloc((void**)&d_y, n_size * sizeof(float));
    
    // Step 2. Copy to GPU memory
    cudaMemcpy(d_x, px, n_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, py, n_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Step 3. Kernel Call
    saxpy(py, px, 2.0, n_size);
    
    dim3 blockDim(16);
    dim3 gridDim((n_size + blockDim.x - 1) / blockDim.x);
    d_saxpy<<< gridDim, blockDim >>>(d_y, d_x, 2.0, n_size);

    // Step 4. Copy from GPU
    cudaMemcpy(py_cuda, d_y, n_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Step 5. Check Result
    check_result(py, py_cuda, n_size);
    
    // Step 6. Finalize GPU memory
    cudaFree(d_x);
    cudaFree(d_y);
    
    free(px);
    free(py);
    free(py_cuda);
    
    return 0;
}