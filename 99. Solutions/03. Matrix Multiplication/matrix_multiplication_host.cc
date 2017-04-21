
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix_multiplication.h"

void matrix_multiplication_host(Matrix &C, Matrix A, Matrix B) {
    struct timespec start, finish;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int y = 0; y < C.height; y++) {
        for (int x = 0; x < C.width; x++) {
            int value = 0.f;
            for (int e = 0; e < A.width; e++) {
                value += A.elements[y * A.width + e] * B.elements[e * B.width + x];
            }
            C.elements[y * C.width + x] = value;
            
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC, &finish);
    
    double elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000.0;
    printf("CPU elapsed time: %f ms\n", elapsed);
}
