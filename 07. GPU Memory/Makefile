
CC = gcc
CC_OPTS = -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart

NVCC = nvcc
NVCC_OPTS = --ptxas-options=--verbose \
			-gencode arch=compute_30,code=sm_30 \
			-I/usr/local/cuda/samples/common/inc

all: test_shared test_texture

test_shared: test_shared.o 
	$(NVCC) $(NVCC_OPTS) test_shared.o -o test_shared

test_texture: test_texture.o 
	$(NVCC) $(NVCC_OPTS) test_texture.o -o test_texture
    
test_shared.o: test_shared.cu sgemm_shared.cu sgemm.cu sgemm.cuh
	$(NVCC) $(NVCC_OPTS) test_shared.cu -c test_shared.o

test_texture.o: sgemm_texture_reference.cu sgemm_texture_object.cu test_texture.cu sgemm.cuh
	$(NVCC) $(NVCC_OPTS) test_texture.cu -c test_texture.o

       
constant: constant.cu
	$(NVCC) constant.cu -c constant 
    
clean:
	rm *.o