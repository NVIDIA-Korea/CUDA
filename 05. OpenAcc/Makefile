
CC=gcc

openmp: saxpy_openmp.cc
	$(CC) -fopenmp saxpy_openmp.cc -O3 -o saxpy_openmp

openacc: saxpy_openacc.c
	$(CC) saxpy_openacc.c -fopenacc -foffload=nvptx-none -foffload="-O3" -O3 -o saxpy_openacc 
    
sum_mp: mat_sum_mp.cc
	$(CC) -fopenmp mat_sum_mp.cc -O3 -o mat_sum_mp

sum_acc: mat_sum_acc.c
	$(CC) mat_sum_acc.c -fopenacc -foffload=nvptx-none -foffload="-O3" -O3 -o mat_sum_acc 