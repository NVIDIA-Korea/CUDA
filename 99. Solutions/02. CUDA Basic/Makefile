
CC=gcc
CCFLAGS=-Wall
NVCC=nvcc

all: map gather scatter transpose

map: map.o
	$(NVCC) map.o -o map

gather: gather.o
	$(NVCC) gather.o -o gather
    
scatter: scatter.o
	$(NVCC) scatter.o -o scatter

transpose: transpose.o
	$(NVCC) transpose.o -o transpose    
    
map.o: map.cu
	$(NVCC) -c map.cu
    
gather.o: gather.cu
	$(NVCC) -c gather.cu

scatter.o: scatter.cu
	$(NVCC) -c scatter.cu

transpose.o: transpose.cu
	$(NVCC) -c transpose.cu