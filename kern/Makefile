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

# K=2560 column slice of a K=10240 row, accumulating into y (hc_*_down, hc_*_inject)
gemv_bf16_hc.cubin: gemv_bf16.sass skel_bf16.cubin
	cp skel_bf16.cubin $@
	GEMV_K=2560 GEMV_ROWSTRIDE=20480 GEMV_ACC=1 $(MAXAS) -i gemv_bf16.sass $@

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

skel_q3_k.cubin: gemv_q3_k.ptx
	$(PTXAS11) -arch=sm_52 -o $@ $<

gemv_q3_k_k2560.cubin: gemv_q3_k.sass skel_q3_k.cubin
	cp skel_q3_k.cubin $@
	GEMV_K=2560 $(MAXAS) -i gemv_q3_k.sass $@

gemv_q3_k_k6144.cubin: gemv_q3_k.sass skel_q3_k.cubin
	cp skel_q3_k.cubin $@
	GEMV_K=6144 $(MAXAS) -i gemv_q3_k.sass $@

skel_q6_k.cubin: gemv_q6_k.ptx
	$(PTXAS11) -arch=sm_52 -o $@ $<

gemv_q6_k_k2560.cubin: gemv_q6_k.sass skel_q6_k.cubin
	cp skel_q6_k.cubin $@
	GEMV_K=2560 $(MAXAS) -i gemv_q6_k.sass $@

gemv_q6_k_k6144.cubin: gemv_q6_k.sass skel_q6_k.cubin
	cp skel_q6_k.cubin $@
	GEMV_K=6144 $(MAXAS) -i gemv_q6_k.sass $@

skel_iq4_xs.cubin: gemv_iq4_xs.ptx
	$(PTXAS11) -arch=sm_52 -o $@ $<

gemv_iq4_xs_k2560.cubin: gemv_iq4_xs.sass skel_iq4_xs.cubin
	cp skel_iq4_xs.cubin $@
	GEMV_K=2560 $(MAXAS) -i gemv_iq4_xs.sass $@

gemv_iq4_xs_k6144.cubin: gemv_iq4_xs.sass skel_iq4_xs.cubin
	cp skel_iq4_xs.cubin $@
	GEMV_K=6144 $(MAXAS) -i gemv_iq4_xs.sass $@

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

# block-32 formats at K=640: one generator, GEMV_FMT selects the format
B32 = q4_0 q5_0 q8_0 iq4_nl
B32_CUBINS = $(foreach f,$(B32),gemv_$(f)_k640.cubin)

skel_b32_%.cubin: gemv_skel.ptx
	sed 's/KERNEL/gemv_$*/' $< > skel_b32_$*.ptx
	$(PTXAS11) -arch=sm_52 -o $@ skel_b32_$*.ptx

gemv_%_k640.cubin: gemv_b32.sass skel_b32_%.cubin
	cp skel_b32_$*.cubin $@
	GEMV_FMT=$* GEMV_K=640 $(MAXAS) -i -k gemv_$* gemv_b32.sass $@

b32: $(B32_CUBINS)
.PHONY: b32

# software grid barrier primitive and its visibility test
skel_gridbar.cubin: gridbar.ptx
	$(PTXAS11) -arch=sm_52 -o $@ $<

gridbar.cubin: gridbar.sass skel_gridbar.cubin
	cp skel_gridbar.cubin $@
	$(MAXAS) -i gridbar.sass $@

gridbar_test: gridbar_test.c
	$(CC) -std=gnu99 -O2 -Wall -o $@ gridbar_test.c -I$(CUDA)/include -lcuda

# die-to-die mailbox (40 KB payload + sequence word over P2P), two-device test
skel_mailbox.cubin: mailbox.ptx
	$(PTXAS11) -arch=sm_52 -o $@ $<

mailbox.cubin: mailbox.sass skel_mailbox.cubin
	cp skel_mailbox.cubin $@
	$(MAXAS) -i mailbox.sass $@

mailbox_test: mailbox_test.c
	$(CC) -std=gnu99 -O2 -Wall -o $@ mailbox_test.c -I$(CUDA)/include -lcuda

# activation quantiser (one warp per vector), per vector length
skel_actq.cubin: actq.ptx
	$(PTXAS11) -arch=sm_52 -o $@ $<

actq_k%.cubin: actq.sass skel_actq.cubin
	cp skel_actq.cubin $@
	GEMV_K=$* $(MAXAS) -i actq.sass $@

actq_test: actq_test.c ../ref/quant.c ../ref/quant.h ../ref/gguf.c
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ actq_test.c ../ref/quant.c ../ref/gguf.c -I../ref -I$(CUDA)/include -lcuda -lm

actq: actq_k640.cubin actq_k2560.cubin actq_k6144.cubin actq_k10240.cubin actq_test
.PHONY: actq

# fused RMSNorm + quantiser for the 2560-wide block inputs
skel_normq.cubin: normq.ptx
	$(PTXAS11) -arch=sm_52 -o $@ $<

normq.cubin: normq.sass skel_normq.cubin
	cp skel_normq.cubin $@
	$(MAXAS) -i normq.sass $@

normq_test: normq_test.c ../ref/quant.c ../ref/quant.h ../ref/gguf.c
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ normq_test.c ../ref/quant.c ../ref/gguf.c -I../ref -I$(CUDA)/include -lcuda -lm

# fp32 RMSNorm, per vector length (hc norms 10240, 2560)
skel_rmsnorm.cubin: rmsnorm.ptx
	$(PTXAS11) -arch=sm_52 -o $@ $<

rmsnorm_k%.cubin: rmsnorm.sass skel_rmsnorm.cubin
	cp skel_rmsnorm.cubin $@
	GEMV_K=$* $(MAXAS) -i rmsnorm.sass $@

rmsnorm_test: rmsnorm_test.c ../ref/quant.c ../ref/quant.h ../ref/gguf.c
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ rmsnorm_test.c ../ref/quant.c ../ref/gguf.c -I../ref -I$(CUDA)/include -lcuda -lm

# barrier capacity micro-test
skel_bartest.cubin: bartest.ptx
	$(PTXAS11) -arch=sm_52 -o $@ $<
bartest_test: bartest_test.c
	$(CC) -std=gnu99 -O2 -Wall -o $@ bartest_test.c -I$(CUDA)/include -lcuda

# transcendental library test
skel_fmath.cubin: fmath.ptx
	$(PTXAS11) -arch=sm_52 -o $@ $<

fmath.cubin: fmath.sass fmath.pl skel_fmath.cubin
	cp skel_fmath.cubin $@
	$(MAXAS) -i fmath.sass $@

fmath_test: fmath_test.c ../ref/fmath.c ../ref/fmath.h
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ fmath_test.c ../ref/fmath.c -I../ref -I$(CUDA)/include -lcuda -lm

# DeltaNet head step
skel_dnstep.cubin: dnstep.ptx
	$(PTXAS11) -arch=sm_52 -o $@ $<

dnstep.cubin: dnstep.sass fmath.pl skel_dnstep.cubin
	cp skel_dnstep.cubin $@
	$(MAXAS) -i dnstep.sass $@

dnstep_tiled.cubin: dnstep.sass fmath.pl skel_dnstep.cubin
	cp skel_dnstep.cubin $@
	DN_TILED=1 $(MAXAS) -i dnstep.sass $@

dnstep_test: dnstep_test.c ../ref/deltanet.c ../ref/deltanet.h ../ref/fmath.c
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ dnstep_test.c ../ref/deltanet.c ../ref/fmath.c -I../ref -I$(CUDA)/include -lcuda -lm

# conv1d + SiLU state step
skel_conv1d.cubin: conv1d.ptx
	$(PTXAS11) -arch=sm_52 -o $@ $<

conv1d.cubin: conv1d.sass fmath.pl skel_conv1d.cubin
	cp skel_conv1d.cubin $@
	$(MAXAS) -i conv1d.sass $@

conv1d_test: conv1d_test.c ../ref/deltanet.c ../ref/deltanet.h ../ref/fmath.c
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ conv1d_test.c ../ref/deltanet.c ../ref/fmath.c -I../ref -I$(CUDA)/include -lcuda -lm

# hyper-connection vector ops
skel_hcops.cubin: hcops.ptx
	$(PTXAS11) -arch=sm_52 -o $@ $<

hc_%.cubin: hcops.sass fmath.pl skel_hcops.cubin
	cp skel_hcops.cubin $@
	HC_OP=$* $(MAXAS) -i hcops.sass $@

hcops_test: hcops_test.c ../ref/hc.c ../ref/hc.h ../ref/fmath.c
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ hcops_test.c ../ref/hc.c ../ref/fmath.c -I../ref -I$(CUDA)/include -lcuda -lm

hcops: hc_silu.cubin hc_mix.cubin hc_write.cubin hcops_test
.PHONY: hcops

# MoE router
skel_router.cubin: router.ptx
	$(PTXAS11) -arch=sm_52 -o $@ $<

router.cubin: router.sass fmath.pl skel_router.cubin
	cp skel_router.cubin $@
	$(MAXAS) -i router.sass $@

router_test: router_test.c ../ref/router.c ../ref/router.h ../ref/fmath.c
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ router_test.c ../ref/router.c ../ref/fmath.c -I../ref -I$(CUDA)/include -lcuda -lm

# attention q/k norm + RoPE + KV append
skel_attnqk.cubin: attnqk.ptx
	$(PTXAS11) -arch=sm_52 -o $@ $<

attnqk.cubin: attnqk.sass skel_attnqk.cubin
	cp skel_attnqk.cubin $@
	$(MAXAS) -i attnqk.sass $@

attnqk_test: attnqk_test.c ../ref/attn.c ../ref/attn.h ../ref/fmath.c
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ attnqk_test.c ../ref/attn.c ../ref/fmath.c -I../ref -I$(CUDA)/include -lcuda -lm

# ---- attention core: partials + merge ----
skel_attnpart.cubin: attnpart.ptx
	$(PTXAS11) -arch=sm_52 -o $@ $<

attnpart.cubin: attnpart.sass fmath.pl skel_attnpart.cubin
	cp skel_attnpart.cubin $@
	$(MAXAS) -i attnpart.sass $@

skel_attnmerge.cubin: attnmerge.ptx
	$(PTXAS11) -arch=sm_52 -o $@ $<

attnmerge.cubin: attnmerge.sass fmath.pl skel_attnmerge.cubin
	cp skel_attnmerge.cubin $@
	$(MAXAS) -i attnmerge.sass $@

attn_test: attn_test.c ../ref/attn.c ../ref/attn.h ../ref/fmath.c
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ attn_test.c ../ref/attn.c ../ref/fmath.c -I../ref -I$(CUDA)/include -lcuda -lm

.PHONY: attn
attn: attnqk.cubin attnqk_test attnpart.cubin attnmerge.cubin attn_test
