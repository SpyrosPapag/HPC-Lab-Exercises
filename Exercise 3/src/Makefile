CC=g++
CFLAGS=-fopenmp -lgomp -foffload=nvptx-none -lcuda -fcf-protection=none -fno-stack-protector

all: CMM

CMM: CMM.cpp
	$(CC) $(CFLAGS) -o CMM CMM.cpp

clean:
	rm -f CMM