
#include <stdio.h>
#include <stdlib.h>
#include "util.h"

void init_matrix(Matrix* M, int width, int height, int seed) {
    M->width = width;
    M->height = height;
    M->elements = (int*)malloc(width * height * sizeof(int));
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            M->elements[j * width + i] = (j * width + i) * seed;
        }
    }
}

void check_result(Matrix &A, Matrix &B) {
    if ((A.width != B.width) || (A.height != B.height)) {
        printf("Wrong output size..\n");
        return;
    }

    int count = 0;
    for (int y = 0; y < A.width; y++) {
        for (int x = 0; x < A.height; x++) {
            count += (A.elements[y * A.width + x] != B.elements[y * B.width + x]) ? 1 : 0;
        }
    }
    if (count != 0) {
        printf("Wrong result...Check the code again..(%d)\n", count);
    }
    else {
        printf("Success!!");
    }
}