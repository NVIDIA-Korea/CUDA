#ifndef _MATRIX_MULTIPLICATION_H_
#define _MATRIX_MULTIPLICATION_H_
    
// Matrices are stored in row-major order: 
// M(row, col) = *(M.elements + row * M.stride + col) 
typedef struct { 
    int width; 
    int height; 
    int stride; 
    int* elements; 
} Matrix; 

void matrix_multiplication(Matrix &C, Matrix A, Matrix B);
void matrix_multiplication_host(Matrix &C, Matrix A, Matrix B);

#endif /* _MATRIX_MULTIPLICATION_H_ */