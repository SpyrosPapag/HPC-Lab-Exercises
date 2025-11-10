CC=gcc
Q0 = #-Wall
Q1 = -march=native -ftree-vectorize -O3 #-fopt-info-vec-all -Wall
Q2 = -fopenmp -O3 #-Wall
Q3 = -mavx512f -march=native -mavx -O3 -fno-tree-vectorize #-Wall
CFLAGS=  $(Q0)
LIBS=-lm

all: bench

bench: bench.c weno.h
	$(CC) $(CFLAGS) -o bench bench.c $(LIBS)

clean:
	rm -f bench

