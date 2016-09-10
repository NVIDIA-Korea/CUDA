
#include <iostream>
#include "utils.h"
#include <string>
#include <stdio.h>

size_t numRows();  //return # of rows in the image
size_t numCols();  //return # of cols in the image

//include the definitions of the above functions for this homework
#include "blur_filter.cuh"
#include "gpu_memory_proc.cc"

int main(int argc, char **argv) {
    uchar4 *h_inputImageRGBA,  *d_inputImageRGBA;
    uchar4 *h_outputImageRGBA, *d_outputImageRGBA;
    unsigned char *d_redBlurred, *d_greenBlurred, *d_blueBlurred;

    float *h_filter;
    int    filterWidth;

    std::string input_file;
    std::string output_file;
    if (argc == 3) {
        input_file  = std::string(argv[1]);
        output_file = std::string(argv[2]);
    }
    else {
        std::cerr << "Usage: ./blur input_file output_file" << std::endl;
        exit(1);
    }
    //load the image and give us our input and output pointers
    preProcess(&h_inputImageRGBA, &h_outputImageRGBA, &d_inputImageRGBA, &d_outputImageRGBA,
             &d_redBlurred, &d_greenBlurred, &d_blueBlurred,
             &h_filter, &filterWidth, input_file);

    allocateMemoryAndCopyToGPU(numRows(), numCols(), h_filter, filterWidth);
    //call the students' code
    your_gaussian_blur(h_inputImageRGBA, d_inputImageRGBA, d_outputImageRGBA, numRows(), numCols(),
                     d_redBlurred, d_greenBlurred, d_blueBlurred, filterWidth);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    

    cleanup();
    //check results and output the blurred image
    postProcess(output_file);

    checkCudaErrors(cudaFree(d_redBlurred));
    checkCudaErrors(cudaFree(d_greenBlurred));
    checkCudaErrors(cudaFree(d_blueBlurred));
    
    std::cout << "Finished Blurring.." << std::endl; 

    return 0;
}