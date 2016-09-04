
CC=gcc
NVCC=nvcc
CCFLAG=-Wall

all: cpu gpu gpu_2d

cpu: saxpy_cpu.o
	$(CC) saxpy_cpu.o -o saxpy_cpu
    
gpu: saxpy_gpu.o
	$(NVCC) saxpy_gpu.o -o saxpy_gpu
    
gpu_2d: saxpy_gpu_2d.o
	$(NVCC) saxpy_gpu_2d.o -o saxpy_gpu_2d

saxpy_cpu.o: saxpy_cpu.cc
	$(CC) $(CCFLAG) -c saxpy_cpu.cc
    
saxpy_gpu.o: saxpy_gpu.cu
	$(NVCC) -c saxpy_gpu.cu

saxpy_gpu_2d.o: saxpy_gpu_2d.cu
	$(NVCC) -c saxpy_gpu_2d.cu