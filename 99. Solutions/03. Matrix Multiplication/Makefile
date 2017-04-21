
CC=gcc
NVCC=nvcc

all: cuda cuda_opt

clean:
	rm *.o matrix_multiplication matrix_multiplication_opt
    
cuda: matrix_multiplication.o matrix_multiplication_host.o util.o
	$(NVCC) -o matrix_multiplication matrix_multiplication.o matrix_multiplication_host.o util.o

matrix_multiplication.o: matrix_multiplication.cu matrix_multiplication.h
	$(NVCC) -c matrix_multiplication.cu
    
matrix_multiplication_host.o: matrix_multiplication_host.cc matrix_multiplication.h util.h
	$(NVCC) -c matrix_multiplication_host.cc util.o

util.o: util.cc util.h
	$(NVCC) -c util.cc
    
cuda_opt: matrix_multiplication_opt.o
	$(NVCC) -o matrix_multiplication_opt matrix_multiplication_opt.o matrix_multiplication_host.o util.o

matrix_multiplication_opt.o: matrix_multiplication_opt.cu
	$(NVCC) -c matrix_multiplication_opt.cu