#include <stdio.h>
#include "util.c"

const int n_width = 1024;
const int n_height = 1024;

void gather_sum(float* p_out, float* p_in, int n_filter_size, int n_width, int n_height) {
    for (int row = 0; row < n_height; row++) {
        for (int col = 0; col < n_width; col++) {
            float sum = 0.f;
            
            for (int row_filter = 0; row_filter < n_filter_size; row_filter++) {
                for (int col_filter = 0; col_filter < n_filter_size; col_filter++) {
                    int input_idx = n_width * (row + row_filter) + col + col_filter;
                    
                    if ((row + row_filter >= 0 && row + row_filter < n_height) && 
                        (col + col_filter >= 0 && col + col_filter < n_width)) {
                        sum += p_in[input_idx];
                    }
                }
            }
            p_out[row * n_width + col] = sum;
        }
    }
}

__global__
void d_gather_sum(float* p_out, float* p_in, int n_filter_size, int n_width, int n_height) {
    // TODO: Write gather code
}

int main() {
    float *p_in, *p_out, *p_out_host;
    float *d_in, *d_out;
    int n_filter_size = 3;
    int n_size = n_width * n_height;
    
    p_in = get_buffer(n_size);
    p_out = get_buffer(n_size);
    p_out_host = get_buffer(n_size);
    
    cudaMalloc((void**)&d_in, n_size * sizeof(float));
    cudaMalloc((void**)&d_out, n_size * sizeof(float));
  
    cudaMemcpy(d_in, p_in, n_size * sizeof(float), cudaMemcpyHostToDevice);
    
    gather_sum(p_out_host, p_in, n_filter_size, n_width, n_height);
    
    dim3 blockDim(16, 16);
    dim3 gridDim((n_width + blockDim.x - 1) / blockDim.x, (n_height + blockDim.y - 1) / blockDim.y);
    d_gather_sum<<<gridDim, blockDim>>>(d_out, d_in, n_filter_size, n_width, n_height);
    
    cudaMemcpy(p_out, d_out, n_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    check_result(p_out, p_out_host, n_size);
    printf("%d\n", n_size);
    
    cudaFree(d_in);
    cudaFree(d_out);
    
    free(p_in);
    free(p_out);
    free(p_out_host);
}