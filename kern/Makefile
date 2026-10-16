CUDA ?= /opt/cuda
PTXAS11 ?= /opt/cuda-11.4/bin/ptxas
MAXAS = PATH=$(CUDA)/bin:$$PATH perl -I../maxas/lib ../maxas/bin/maxas.pl
CC ?= gcc
SHARD1 ?= /mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf

all: gemv_q2_0.cubin gemv_test

# ptxas (11.4, matches driver 470) builds the container; maxas replaces the code.
gemv_q2_0.cubin: gemv_q2_0.sass gemv_q2_0.ptx
	$(PTXAS11) -arch=sm_52 -o skel_q2_0.cubin gemv_q2_0.ptx
	cp skel_q2_0.cubin $@
	$(MAXAS) -i gemv_q2_0.sass $@

# readable listing of what was inserted (control codes included)
gemv_q2_0.dis: gemv_q2_0.cubin
	$(MAXAS) -e $< $@

gemv_test: gemv_test.c ../ref/gguf.c ../ref/quant.c ../ref/gguf.h ../ref/quant.h
	$(CC) -std=gnu99 -O2 -Wall -o $@ gemv_test.c ../ref/gguf.c ../ref/quant.c -I../ref -I$(CUDA)/include -lcuda -lm

test: all
	./gemv_test $(SHARD1)

clean:
	rm -f gemv_q2_0.cubin skel_q2_0.cubin gemv_q2_0.dis gemv_test

.PHONY: all test clean
