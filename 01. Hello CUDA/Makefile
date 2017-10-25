
CC=g++
NVCC=nvcc
CCFLAG=-Wall

all: cpu gpu gpu_2d

vector_copy: vector_copy.cc
	$(CC) $(CCFLAG) vector_copy.cc -o vector_copy
    
buffer_copy_iter: buffer_copy_iter.cc
	$(CC) $(CCFLAG) buffer_copy_iter.cc -o buffer_copy_iter
    
buffer_alloc: buffer_alloc.cu
	$(NVCC) buffer_alloc.cu -o buffer_alloc
    
cuda_memcpy: cuda_memcpy.cu
	$(NVCC) cuda_memcpy.cu -o cuda_memcpy
    
kernel_call: kernel_call.cu
	$(NVCC) kernel_call.cu -o kernel_call
