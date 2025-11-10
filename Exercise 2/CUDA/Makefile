all: CMM

NVFLAGS = --gpu-architecture=sm_89 -O3 -lineinfo

CMM: CMM.cu
	nvcc $(NVFLAGS) -o CMM CMM.cu

clean:
	rm -f CMM
