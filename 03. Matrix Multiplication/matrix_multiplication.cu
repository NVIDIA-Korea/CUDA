
#include <stdio.h>
#include "matrix_multiplication.h"
#include "util.h"

__global__
void d_matrix_multiplication(Matrix C, Matrix A, Matrix B) {
    // TODO
    // Matrix Multiplication 연산 GPU 코드를 작성하세요.
    // 필요하다면 matrix_multiplication_host.cc 파일을 참고하세요.
    
}

void matrix_multiplication(Matrix &C, Matrix A, Matrix B) {
    int size;
    
    ////////////////////////////////
    // CUDA Event Create to estimate elased time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // CUDA Operation
    cudaEventRecord(start, 0);
    /////////////////////////////////
    
    // Create GPU memory
    Matrix d_A;
    d_A.width = A.width; d_A.height = A.height;
    size = A.width * A.height * sizeof(int);
    cudaMalloc((void**)&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
    
    Matrix d_B;
    d_B.width = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(int);
    cudaMalloc((void**)&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
    
    Matrix d_C;
    d_C.height = C.height; d_C.width = C.width;
    size = C.width * C.height * sizeof(int);
    cudaMalloc((void**)&d_C.elements, size);
    
    // Kernel launch
    // TODO: call CUDA Kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((d_C.width + blockDim.x - 1) / blockDim.x, (d_C.height + blockDim.y - 1) / blockDim.y);
    ...

    cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
    
    /////////////////////////////////
    // Estimate CUDA operation time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("CUDA Elapsed time: %f ms\n", elapsedTime);
    
    // finalize CUDA event
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    /////////////////////////////////
}

int main() {
    int width_a = 256;
    int height_a = 128;
    int height_b = 256;
    int width_b = height_a;
    
    Matrix A, B, C, C_cuda;
    
    init_matrix(&A, width_a, height_a, 1);
    init_matrix(&B, width_b, height_b, 2);
    init_matrix(&C, A.height, B.width, 0);
    init_matrix(&C_cuda, A.height, B.width, 0);
    
    // Matrix Multiplication
    matrix_multiplication(C_cuda, A, B);
    matrix_multiplication_host(C, A, B);
    
    // Check results
    check_result(C, C_cuda);
    
    free(A.elements);
    free(B.elements);
    free(C.elements);
    free(C_cuda.elements);
     
    return 0;
}