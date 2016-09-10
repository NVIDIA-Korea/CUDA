
#include <stdio.h>
#include "utils.h"

int checkCudaErrors(cudaError_t error) {
    if (cudaSuccess != error) {
        printf("%s[%d]\n", __FILE__, __LINE__);
        printf("%s\n", cudaGetErrorName(error));
        printf("%s\n", cudaGetErrorString(error));
        return 1; 
    }
    else {
        return 0;
    }
}