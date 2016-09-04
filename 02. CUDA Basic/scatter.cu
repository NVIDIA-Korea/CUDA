#include <stdio.h>
#include "util.c"

const int n_width = 1024;
const int n_height = 1024;

void scatter(float* p_out, float* p_in, int n_width, int n_height) {
    for (int j = 0; j < n_width; j++) {
        for (int i = 0; i < n_height; i++) {
            p_out[i * n_width + j] = j;
        }
    }
}

__global__
void d_scatter_1D(float* d_out, float* d_in, int n_width, int n_height) {
    // TODO: Write scatter code
}

void CallScatter1D(float* d_out, float* d_in, int n_width, int n_height) {
    // TODO: Write Kernal Call using 1D block size
    dim3 blockDim(256);
}

__global__
void d_scatter_2D(float* d_out, float* d_in, int n_width, int n_height) {
    // TODO: Write scatter code
}

void CallScatter2D(float* d_out, float* d_in, int n_width, int n_height) {
    // TODO: Write Kernel Call using 2D block size
    dim3 blockDim(16, 16);
}

int main() {
    float *p_in, *p_out, *p_out_host;
    float *d_in, *d_out;
    
    p_in = get_buffer(n_width);
    p_out = get_buffer(n_width * n_height);
    p_out_host = get_buffer(n_width * n_height);
    
    cudaMalloc((void**)&d_in, n_width * sizeof(float));
    cudaMalloc((void**)&d_out, n_width * n_height * sizeof(float));
    
    cudaMemcpy(d_in, p_in, n_width * sizeof(float), cudaMemcpyHostToDevice);
    
    scatter(p_out_host, p_in, n_width, n_height);
    
    CallScatter1D(d_out, d_in, n_width, n_height);
    cudaMemcpy(p_out, d_out, n_width * n_height * sizeof(float), cudaMemcpyDeviceToHost);
    check_result(p_out_host, p_out, n_width * n_height);
    
    CallScatter2D(d_out, d_in, n_width, n_height);
    cudaMemcpy(p_out, d_out, n_width * n_height * sizeof(float), cudaMemcpyDeviceToHost);
    check_result(p_out_host, p_out, n_width * n_height);
    
    cudaFree(d_in);
    cudaFree(d_out);
    
    free(p_in);
    free(p_out);
    free(p_out_host);
}