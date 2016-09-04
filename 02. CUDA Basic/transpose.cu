#include <stdio.h>
#include "util.c"

void transpose(float *p_out, float *p_in, int n_width, int n_height) {
    for (int j = 0; j < n_height; j++) {
        for (int i = 0; i < n_width; i++) {
            p_out[i * n_height + j] = p_in[j * n_width + i];
        }
    }
}

__global__ 
void d_transpose(float *d_out, float *d_in, int n_width, int n_height) {
    // TODO: Write transpose code
}

int main() {
    float *p_in, *p_out, *p_out_cuda;
    float *d_in, *d_out;
    
    int n_width = 1920;
    int n_height = 1080;
    
    p_in = get_buffer(n_width * n_height);
    p_out = get_buffer(n_width * n_height);
    p_out_cuda = get_buffer(n_width * n_height);
    
    // Step 1. Allocate to GPU memory
    cudaMalloc((void**)&d_in, n_width * n_height * sizeof(float));
    cudaMalloc((void**)&d_out, n_width * n_height * sizeof(float));
    
    // Initialize input data
    for (int j = 0; j < n_height; j++) {
        for (int i = 0; i < n_width; i++) {
            p_in[j * n_width + i] = float(j * n_width + i);
        }
    }
    
    // Step 2. Copy to GPU memory
    cudaMemcpy(d_in, p_in, n_width * n_height * sizeof(float), cudaMemcpyHostToDevice);
    
    transpose(p_out, p_in, n_width, n_height);
    
    // Step 3. Kernel leaunch
    dim3 blockDim(16, 16);
    dim3 gridDim((n_width + blockDim.x - 1) / blockDim.x, (n_height + blockDim.y - 1) / blockDim.y);
    d_transpose<<<gridDim, blockDim>>>(d_out, d_in, n_width, n_height);
    
    // Step 4. Copy from GPU
    cudaMemcpy(p_out_cuda, d_out, n_width * n_height * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Step 5. check result
    check_result(p_out, p_out_cuda, n_width * n_height);
    
    // Step 6. free GPU memory
    cudaFree(d_in);
    cudaFree(d_out);
    
    free(p_in);
    free(p_out);
    free(p_out_cuda);
}