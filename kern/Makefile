CUDA ?= /opt/cuda
PTXAS11 ?= /opt/cuda-11.4/bin/ptxas
MAXAS = PATH=$(CUDA)/bin:$$PATH perl -I../maxas/lib ../maxas/bin/maxas.pl
CC ?= gcc
SHARD1 ?= /mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf

all: gemv_q2_0_k2560.cubin gemv_q2_0_k640.cubin gemv_test gemv_bf16_k2560.cubin gemv_bf16_k320.cubin gemv_bf16_test gemv_q4_k_k2560.cubin gemv_q4_k_k6144.cubin gemv_q5_k_k2560.cubin gemv_q5_k_k6144.cubin gemv_k_test

# ptxas (11.4, matches driver 470) builds the container; maxas replaces the code.
# Build switches (GEMV_TRACE, GEMV_NOLOAD, ...) come from the environment.
skel_q2_0.cubin: gemv_q2_0.ptx
	$(PTXAS11) -arch=sm_52 -o $@ $<

gemv_q2_0_k2560.cubin: gemv_q2_0.sass skel_q2_0.cubin
	cp skel_q2_0.cubin $@
	GEMV_K=2560 $(MAXAS) -i gemv_q2_0.sass $@

gemv_q2_0_k640.cubin: gemv_q2_0.sass skel_q2_0.cubin
	cp skel_q2_0.cubin $@
	GEMV_K=640 $(MAXAS) -i gemv_q2_0.sass $@

skel_bf16.cubin: gemv_bf16.ptx
	$(PTXAS11) -arch=sm_52 -o $@ $<

gemv_bf16_k2560.cubin: gemv_bf16.sass skel_bf16.cubin
	cp skel_bf16.cubin $@
	GEMV_K=2560 $(MAXAS) -i gemv_bf16.sass $@

gemv_bf16_k320.cubin: gemv_bf16.sass skel_bf16.cubin
	cp skel_bf16.cubin $@
	GEMV_K=320 $(MAXAS) -i gemv_bf16.sass $@

skel_q4_k.cubin: gemv_q4_k.ptx
	$(PTXAS11) -arch=sm_52 -o $@ $<

gemv_q4_k_k2560.cubin: gemv_q4_k.sass skel_q4_k.cubin
	cp skel_q4_k.cubin $@
	GEMV_K=2560 $(MAXAS) -i gemv_q4_k.sass $@

gemv_q4_k_k6144.cubin: gemv_q4_k.sass skel_q4_k.cubin
	cp skel_q4_k.cubin $@
	GEMV_K=6144 $(MAXAS) -i gemv_q4_k.sass $@

skel_q5_k.cubin: gemv_q5_k.ptx
	$(PTXAS11) -arch=sm_52 -o $@ $<

# the K-quant generator is shared; GEMV_FMT selects the Q5_K high-bit stream
gemv_q5_k_k2560.cubin: gemv_q4_k.sass skel_q5_k.cubin
	cp skel_q5_k.cubin $@
	GEMV_FMT=q5_k GEMV_K=2560 $(MAXAS) -i -k gemv_q5_k gemv_q4_k.sass $@

gemv_q5_k_k6144.cubin: gemv_q4_k.sass skel_q5_k.cubin
	cp skel_q5_k.cubin $@
	GEMV_FMT=q5_k GEMV_K=6144 $(MAXAS) -i -k gemv_q5_k gemv_q4_k.sass $@

gemv_k_test: gemv_k_test.c ../ref/gguf.c ../ref/quant.c ../ref/gguf.h ../ref/quant.h
	$(CC) -std=gnu99 -O2 -Wall -o $@ gemv_k_test.c ../ref/gguf.c ../ref/quant.c -I../ref -I$(CUDA)/include -lcuda -lm

gemv_bf16_test: gemv_bf16_test.c ../ref/gguf.c ../ref/quant.c ../ref/gguf.h ../ref/quant.h
	$(CC) -std=gnu99 -O2 -Wall -o $@ gemv_bf16_test.c ../ref/gguf.c ../ref/quant.c -I../ref -I$(CUDA)/include -lcuda -lm

# readable listing of what was inserted (control codes included)
%.dis: %.cubin
	$(MAXAS) -e $< $@

gemv_test: gemv_test.c ../ref/gguf.c ../ref/quant.c ../ref/gguf.h ../ref/quant.h
	$(CC) -std=gnu99 -O2 -Wall -o $@ gemv_test.c ../ref/gguf.c ../ref/quant.c -I../ref -I$(CUDA)/include -lcuda -lm

test: all
	./gemv_test $(SHARD1) blk.4.ffn_gate_exps.weight 0 20
	./gemv_test $(SHARD1) blk.4.ffn_down_exps.weight 0 20

clean:
	rm -f gemv_q2_0_k2560.cubin gemv_q2_0_k640.cubin skel_q2_0.cubin gemv_bf16_k2560.cubin gemv_bf16_k320.cubin skel_bf16.cubin *.dis gemv_test gemv_bf16_test

.PHONY: all test clean
