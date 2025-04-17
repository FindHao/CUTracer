ARCH=sm_90

all:
	nvcc -ccbin=$(CXX) -O3 -arch=$(ARCH) vectoradd.cu -o vectoradd -g -lineinfo

clean:
	rm -f vectoradd
