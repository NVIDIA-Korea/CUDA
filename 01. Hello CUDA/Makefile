
CC=g++
NVCC=nvcc
CCFLAG=-Wall

all: cpu gpu gpu_2d

cpu: saxpy_cpu.cc
	$(CC) $(CCFLAG) saxpy_cpu.cc -o saxpy_cpu
    
gpu: saxpy_gpu.cu
	$(NVCC) saxpy_gpu.cu -o saxpy_gpu
    
gpu_2d: saxpy_gpu_2d.cu
	$(NVCC) saxpy_gpu_2d.cu -o saxpy_gpu_2d
