
CC=g++
CCFLAG_MP=-fopenmp -O3
CCFLAG_ACC=-fopenacc -foffload=nvptx-none -foffload="-O3" -O3

PGCC=pgcc
PGFLAG=-acc -Minfo

openmp: saxpy_openmp.cc
	$(CC) $(CCFLAG_MP) saxpy_openmp.cc -O3 -o saxpy_openmp

openacc: saxpy_openacc.c
	$(CC) $(CCFLAG_ACC) saxpy_openacc.c -o saxpy_openacc 
    
sum_mp: mat_sum_mp.cc
	$(CC) $(CCFLAG_MP) mat_sum_mp.cc -o mat_sum_mp

sum_acc: mat_sum_acc.c
	$(CC) $(CCFLAG_ACC) mat_sum_acc.c -o mat_sum_acc 
