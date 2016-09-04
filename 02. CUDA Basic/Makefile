
CC=gcc
CCFLAGS=-Wall
NVCC=nvcc

all: map gather scatter transpose

map: map.cu
	$(NVCC) map.cu -o map

gather: gather.cu
	$(NVCC) gather.cu -o gather
    
stencil: stencil.cu
	$(NVCC) stencil.cu -o stencil
    
scatter: scatter.cu
	$(NVCC) scatter.cu -o scatter

transpose: transpose.cu
	$(NVCC) transpose.cu -o transpose    
