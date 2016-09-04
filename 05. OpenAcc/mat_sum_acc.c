
#include <stdio.h>

/* matrix-sum-acc.c */
#define SIZE 1000
float a[SIZE][SIZE];
float b[SIZE][SIZE];
float c[SIZE][SIZE];

int main() {
    int i,j,k;

    // Initialize matrices.
    for (i = 0; i < SIZE; ++i) {
      for (j = 0; j < SIZE; ++j) {
          a[i][j] = (float)i + j;
          b[i][j] = (float)i - j;
          c[i][j] = 0.0f;
      }
    }

    // Compute matrix multiply
    #pragma acc kernel
    for (i = 0; i < SIZE; ++i) {
      for (j = 0; j < SIZE; ++j) {
        //for (k = 0; k < SIZE; ++k) {
        //  c[i][j] = a[i][k] * b[k][j];
        //}
        c[i][j] = a[i][j] + b[i][j];
      }
    }

    // Print the result matrix.
    /*
    for (i = 0; i < SIZE; ++i) {
      for (j = 0; j < SIZE; ++j)
        printf("%f ", c[i][j]);
      printf("\n");
    }
    */
    printf("OpenACC matrix sum test was successful!\n");

    return 0;
}