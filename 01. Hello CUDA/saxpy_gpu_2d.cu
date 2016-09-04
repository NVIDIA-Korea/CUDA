
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

float* get_buffer(int n_size, float seed = 0.0) {
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

void print_buffer(float* p_buffer, int n_size) {
    for (int j = 0; j < n_size / 10; j++) {
        for (int i = 0; i < 10; i++) {
            printf("%3.2f ", p_buffer[10*j + i]);
        }
        printf("\n");
    }
}

void check_result(float* py, float* py_cuda, int n_width, int n_height) {
    float compare = 0.0;
    for (int j = 0; j < n_width; j++) {
        for (int i = 0; i < n_height; i++) {
            compare += py[j * n_width + i] - py_cuda[j * n_width + i];
        }
    }
    printf("Result: %f\n", compare);
}

/* CPU function */
void saxpy(float* py, float* px, float alpha, int n_width, int n_height) {
    for (int j = 0; j < n_height; j++) {
        for (int i = 0; i < n_width; i++) {
            py[n_height * j + i] = alpha * px[n_height * j + i] + py[n_height * j + i];
        }
    }
}

/* CUDA Kernel function */
__global__ 
void d_saxpy(float* d_y, float* d_x, float alpha, int n_width, int n_height) {
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x; 
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    
    int idx = n_width * idx_y + idx_x;

    d_y[idx] = alpha * d_x[idx] + d_y[idx];
}

int main() {
    float *px, *py, *py_cuda;
    int n_width = 256;
    int n_height = 256;
    
    px = get_buffer(n_width * n_height, 0.01);
    py = get_buffer(n_width * n_height, 0.05);
    py_cuda = get_buffer(n_width * n_height);
        
    // Step 1. Create GPU memory
    float *d_x, *d_y;
    cudaMalloc((void**)&d_x, n_width * n_height * sizeof(float));
    cudaMalloc((void**)&d_y, n_width * n_height * sizeof(float));
    
    // Step 2. Copy to GPU memory
    cudaMemcpy(d_x, px, n_width * n_height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, py, n_width * n_height * sizeof(float), cudaMemcpyHostToDevice);
    
    // Step 3. Kernel Call
    saxpy(py, px, 2.0, n_width, n_height);
    
    dim3 blockDim(256);
    dim3 gridDim((n_width * n_height + blockDim.x - 1) / blockDim.x);
    d_saxpy<<< gridDim, blockDim >>>(d_y, d_x, 2.0, n_width, n_height);

    // Step 4. Copy from GPU
    cudaMemcpy(py_cuda, d_y, n_width * n_height * sizeof(float), cudaMemcpyDeviceToHost);

    // Step 5. Compare CPU & GPU result
    check_result(py, py_cuda, n_width, n_height);
    
    // Step 6. Finalize GPU memory
    cudaFree(d_x);
    cudaFree(d_y);
    
    free(px);
    free(py);
    free(py_cuda);
    
    return 0;
}