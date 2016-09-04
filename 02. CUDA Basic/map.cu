
#include <stdio.h>
#include "util.c"

__global__
void d_map(float* d_out, float* d_in, int n_size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    d_out[idx] = d_in[idx];
}

__global__
void d_map_shift(float* d_out, float* d_in, int n_size, int n_shift) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    d_out[(n_size + idx + n_shift) % n_size] = d_in[idx];
}

int main() {
    float *p_in, *p_out;
    float *d_in, *d_out;
    
    int n_size = 65536;
    
    p_in = get_buffer(n_size);
    p_out = get_buffer(n_size);
    
    cudaMalloc((void**)&d_in, n_size * sizeof(float));
    cudaMalloc((void**)&d_out, n_size * sizeof(float));
    
    cudaMemcpy(d_in, p_in, n_size * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 blockDim(256);
    dim3 gridDim((n_size + blockDim.x - 1) / blockDim.x);
    d_map<<<gridDim, blockDim>>>(d_out, d_in, n_size);
    
    cudaMemcpy(p_out, d_out, n_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    check_result(p_in, p_out, n_size);
    
    d_map_shift<<<gridDim, blockDim>>>(d_out, d_in, n_size, 2);
    cudaMemcpy(p_out, d_out, n_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    check_result(p_in, p_out, n_size);
    
    cudaFree(d_in);
    cudaFree(d_out);
    
    free(p_in);
    free(p_out);
}