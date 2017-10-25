#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

void init_rand() {
    srand (time(0));
}

float* init_vector(int n_size) {
    // buffer create
    float* p_vector = (float*)malloc(n_size * sizeof(float));
    
    // initialize vector
    for (int i = 0; i < n_size; i++) {
        p_vector[i] = (float)rand() / (float)(RAND_MAX);
    }
    
    return p_vector;
}

void print_vector(float* p_vector, int n_size) {
    for (int j = 0; j < n_size / 10; j++) {
        for (int i = 0; i < 10; i++) {
//            cout << setw(5) << fixed << setprecision( 2 ) << p_vector[10*j + i];
            printf("%3.2f ", (float)p_vector[10*j + i]);
        }
        printf("\n");
    }
}

void check_result(float* a, float* b, int n_size) {
    int diff_count = 0;
    for (int i = 0; i < n_size; i++) {
        diff_count += a[i] != b[i] ? 1 : 0;
    }
    
    if (diff_count == 0)
        printf("Complete!!");
    else
        printf("Values are mis-matching from CPU & GPU");
}