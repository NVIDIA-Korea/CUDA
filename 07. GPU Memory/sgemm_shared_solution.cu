
#include "sgemm.cuh"
template <typename T>
__global__ void sgemm_shared(Matrix<T> A, Matrix<T> B, Matrix<T> C, 
                      const T alpha, const T beta, 
                      const int width, const int height) {
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    
    // TODO: Write shared memory declaration code
    __shared__ T s_A[BLOCK_DIM][BLOCK_DIM];
    __shared__ T s_B[BLOCK_DIM][BLOCK_DIM];
    
    // outer box move
    T value = 0;
    for (int step = 0; step < width; step += BLOCK_DIM) {
        // TODO: Write block obtaining Code
        s_A[threadIdx.y][threadIdx.x] = 
                    step + threadIdx.x < width && idx_y < height ? A.elements[idx_y * width + step + threadIdx.x] : 0;
        s_B[threadIdx.y][threadIdx.x] = 
            step + threadIdx.y < height && idx_x < width ? B.elements[(step + threadIdx.y) * width + idx_x] : 0;
        
        __syncthreads();
        
        // inner operation
        for (int e = 0; e < BLOCK_DIM; e++)
            value += s_A[threadIdx.y][e] * s_B[e][threadIdx.x];
    
        __syncthreads();
    }

    if (idx_x >= width || idx_y >= height)
        return;
    C.elements[idx_y * width + idx_x] = alpha * value + beta * C.elements[idx_y * width + idx_x];
    /////////////////
}

template <typename T>
void launch_sgemm_shared(Matrix<T> &dA, Matrix<T> &dB, Matrix<T> &dC,
                      const T alpha, const T beta, 
                      const int width, const int height) {
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    sgemm_shared<<<gridDim, blockDim>>>(dA, dB, dC, alpha, beta, width, height);
}