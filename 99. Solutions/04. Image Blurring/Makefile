
# Common binaries
NVCC            ?= nvcc
GCC             ?= g++

NVCCFLAGS=-I/usr/local/cuda/include -I. -L/usr/local/cuda/lib -lcuda -lcudart
CVFLAGS=$(shell pkg-config --cflags --libs opencv)

gaussian_blur: gaussian_blur.o blur_filter.o utils.o
	$(NVCC) $(NVCCFLAGS) $(CVFLAGS) -o gaussian_blur utils.o gaussian_blur.o blur_filter.o 
    
gaussian_blur_gold: reference_calc.o
	$(NVCC) $(NVCCFLAGS) $(CVFLAGS) -o gaussian_blur_gold reference_calc.o

gaussian_blur.o: gaussian_blur.cpp blur_filter.cuh utils.h reference_calc.h gpu_memory_proc.h
	$(NVCC) $(NVCCFLAGS) -c gaussian_blur.cpp
    
reference_calc.o: reference_calc.cc utils.h reference_calc.h
	$(NVCC) $(NVCCFLAGS) -c reference_calc.cc
    
blur_filter.o: blur_filter.cu blur_filter.cuh reference_calc.cc utils.h
	$(NVCC) $(NVCCFLAGS) -c blur_filter.cu

gpu_memory_proc.o: gpu_memory_proc.cc gpu_memory_proc.h utils.h
	$(NVCC) $(NVCCFLAGS) $(CVFLAGS) -c gpu_memory_proc.cc
    
utils.o: utils.cc utils.h
	$(NVCC) $(NVCCFLAGS) -c utils.cc
    
clean:
	rm *.o