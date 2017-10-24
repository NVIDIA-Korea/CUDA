
NVCC = nvcc
NVCC_OPTS = --ptxas-options=--verbose -gencode=arch=compute_35,code=sm_35 -I/usr/local/cuda/samples/common/inc

all: sgemm
    
sgemm: sgemm.cu
	$(NVCC) sgemm.cu -gencode=arch=compute_30,code=sm_30 -o sgemm 
    
sgemm_v: 
	$(NVCC) sgemm.cu $(NVCC_OPTS) -o sgemm 
    
sgemm_resize_block: sgemm_resize_block.cu
	$(NVCC) sgemm_resize_block.cu $(NVCC_OPTS) -o sgemm_resize_block
    
sgemm_async_copy: sgemm_async_copy.cu
	$(NVCC) sgemm_async_copy.cu $(NVCC_OPTS) -o sgemm_async_copy
    
sgemm_pinned_memory: sgemm_pinned_memory.cu
	$(NVCC) sgemm_pinned_memory.cu $(NVCC_OPTS) -o sgemm_pinned_memory
    
sgemm_zero_copy: sgemm_zero_copy.cu
	$(NVCC) sgemm_zero_copy.cu $(NVCC_OPTS) -o sgemm_zero_copy

sgemm_stream: sgemm_stream.cu
	$(NVCC) sgemm_stream.cu $(NVCC_OPTS) -o sgemm_stream