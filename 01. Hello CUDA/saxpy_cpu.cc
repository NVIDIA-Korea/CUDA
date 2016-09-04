
#include <stdio.h>
#include <stdlib.h>

float* get_vector(int n_size, float seed) {
    // buffer create
    float* p_vector = (float*)malloc(n_size * sizeof(float));
    
    // initialize vector
    for (int i = 0; i < n_size; i++) {
        p_vector[i] = seed * i;
    }
    
    return p_vector;
}

void print_vector(float* p_vector, int n_size) {
    for (int j = 0; j < n_size / 10; j++) {
        for (int i = 0; i < 10; i++) {
            printf("%3.2f ", p_vector[10*j + i]);
        }
        printf("\n");
    }
}

// y = ax + y 연산
void saxpy(float* py, float* px, float alpha, int n_size) {
    for (int i = 0; i < n_size; i++) {
        py[i] = alpha * px[i] + py[i];
    }
}

int main() {
    float *px, *py;
    int n_size = 65536;
    
    px = get_vector(n_size, 0.01);
    py = get_vector(n_size, 0.05);
    
    printf("X\n");
    print_vector(px, 100);
    printf("Y\n");
    print_vector(py, 100);
    
    saxpy(py, px, 2.0, n_size);
    
    printf("saxpy:: y = ax + y\n");
    print_vector(py, 100);
    
    free(px);
    free(py);
    
    return 0;
}