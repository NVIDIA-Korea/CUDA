#include <stdio.h>
#include "util.c"

const int n_width = 1024;
const int n_height = 1024;

void stencil_sum(float* p_out, float* p_in, int* p_filter, int n_filter_size, int n_width, int n_height) {
    for (int row = 0; row < n_height; row++) {
        for (int col = 0; col < n_width; col++) {
            float sum = 0.f;
            for (int i = 0; i < n_filter_size; i++) {
                int col_filter = p_filter[i*2 + 0];
                int row_filter = p_filter[i*2 + 1];
                
                int input_idx = n_width * (row + row_filter) + col + col_filter;
                    
                if ((row + row_filter >= 0 && row + row_filter < n_height) && 
                    (col + col_filter >= 0 && col + col_filter < n_width)) {
                    sum += p_in[input_idx];
                }
            }
            p_out[row * n_width + col] = sum;
        }
    }
}

__global__
void d_stencil_sum(float* p_out, float* p_in, int* p_filter, int n_filter_size, int n_width, int n_height) {
    // TODO: Write gather code
}

int main() {
    float *p_in, *p_out, *p_out_host;
    float *d_in, *d_out;
    int *p_filter, *d_filter;
    int n_filter_size = 5;
    int n_size = n_width * n_height;
    int stencil_filter[5][2] = {{0, -1}, {-1, 0}, {0, 0}, {1, 0}, {0, 1}};
    
    p_in = get_buffer(n_size);
    p_out = get_buffer(n_size);
    p_out_host = get_buffer(n_size);
    p_filter = (int*)get_buffer(n_filter_size * 2);
    
    // Build stencil filter
    memcpy(p_filter, stencil_filter, n_filter_size * 2 * sizeof(int));
        
    cudaMalloc((void**)&d_in, n_size * sizeof(float));
    cudaMalloc((void**)&d_out, n_size * sizeof(float));
    cudaMalloc((void**)&d_filter, n_filter_size * 2 * sizeof(int));
  
    cudaMemcpy(d_in, p_in, n_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, p_filter, n_filter_size * 2 * sizeof(float), cudaMemcpyHostToDevice);
    
    stencil_sum(p_out_host, p_in, p_filter, n_filter_size, n_width, n_height);
    
    // TODO: Write Kernel Call line
    
    cudaMemcpy(p_out, d_out, n_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    check_result(p_out_host, p_out, n_size);
        
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_filter);
    
    free(p_in);
    free(p_out);
    free(p_out_host);
    free(p_filter);
}
