CUDA ?= /opt/cuda
PTXAS ?= /opt/cuda-11.4/bin/ptxas
CC ?= gcc

all: probe probes.cubin

probes.cubin: probes.ptx
	$(PTXAS) -arch=sm_52 -O3 -v -o $@ $<

probe: probe.c
	$(CC) -std=c99 -O2 -Wall -Wextra -o $@ $< -I$(CUDA)/include -lcuda

run: all
	./probe all 1

clean:
	rm -f probe probes.cubin

.PHONY: all run clean
