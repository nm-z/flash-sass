CUDA ?= /opt/cuda
PTXAS11 ?= /opt/cuda-11.4/bin/ptxas
MAXAS = PATH=$(CUDA)/bin:$$PATH perl -I../maxas/lib ../maxas/bin/maxas.pl
CC ?= gcc
SHARD1 ?= /mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf

all: gemv_q2_0_k2560.cubin gemv_q2_0_k640.cubin gemv_test gemv_bf16_k2560.cubin gemv_bf16_k320.cubin gemv_bf16_test gemv_q4_k_k2560.cubin gemv_q4_k_k6144.cubin gemv_q5_k_k2560.cubin gemv_q5_k_k6144.cubin gemv_k_test

ple_ram: ple_ram.c
	$(CC) -std=c11 -O2 -Wall -Wextra -o $@ $<

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
	GEMV_K=2560 GEMV_SLICES=4 $(MAXAS) -i gemv_bf16.sass $@

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

# the rolled-step Q4_K / Q5_K GEMV (gemv_q4_k_c.sass) in skeletons with its staging shared memory; gemv_k_test takes them through GEMV_CUBIN
skel_q4_k_c.cubin: gemv_q4_k_c.ptx
	$(PTXAS11) -arch=sm_52 -o $@ $<

skel_q5_k_c.cubin: gemv_q5_k_c.ptx
	$(PTXAS11) -arch=sm_52 -o $@ $<

gemv_q4_k_c_k%.cubin: gemv_q4_k_c.sass q4_k_math.pl skel_q4_k_c.cubin
	cp skel_q4_k_c.cubin $@
	GEMV_K=$* $(MAXAS) -i -k gemv_q4_k gemv_q4_k_c.sass $@

gemv_q5_k_c_k%.cubin: gemv_q4_k_c.sass q4_k_math.pl skel_q5_k_c.cubin
	cp skel_q5_k_c.cubin $@
	GEMV_FMT=q5_k GEMV_K=$* $(MAXAS) -i -k gemv_q5_k gemv_q4_k_c.sass $@

# the K-quant generator is shared; GEMV_FMT selects the Q5_K high-bit stream
gemv_q5_k_k2560.cubin: gemv_q4_k.sass skel_q5_k.cubin
	cp skel_q5_k.cubin $@
	GEMV_FMT=q5_k GEMV_K=2560 $(MAXAS) -i -k gemv_q5_k gemv_q4_k.sass $@

gemv_q5_k_k6144.cubin: gemv_q4_k.sass skel_q5_k.cubin
	cp skel_q5_k.cubin $@
	GEMV_FMT=q5_k GEMV_K=6144 $(MAXAS) -i -k gemv_q5_k gemv_q4_k.sass $@

skel_q3_k.cubin: gemv_q3_k.ptx
	$(PTXAS11) -arch=sm_52 -o $@ $<

gemv_q3_k_k2560.cubin: gemv_q3_k.sass q3_k_math.pl skel_q3_k.cubin
	cp skel_q3_k.cubin $@
	GEMV_K=2560 $(MAXAS) -i gemv_q3_k.sass $@

gemv_q3_k_k6144.cubin: gemv_q3_k.sass q3_k_math.pl skel_q3_k.cubin
	cp skel_q3_k.cubin $@
	GEMV_K=6144 $(MAXAS) -i gemv_q3_k.sass $@

# the rolled-step Q3_K GEMV (gemv_q3_k_c.sass) in a skeleton with its staging shared memory; gemv_k_test takes it through GEMV_CUBIN
skel_q3_k_c.cubin: gemv_q3_k_c.ptx
	$(PTXAS11) -arch=sm_52 -o $@ $<

gemv_q3_k_c_k2560.cubin: gemv_q3_k_c.sass q3_k_math.pl skel_q3_k_c.cubin
	cp skel_q3_k_c.cubin $@
	GEMV_K=2560 $(MAXAS) -i -k gemv_q3_k gemv_q3_k_c.sass $@

gemv_q3_k_c_k6144.cubin: gemv_q3_k_c.sass q3_k_math.pl skel_q3_k_c.cubin
	cp skel_q3_k_c.cubin $@
	GEMV_K=6144 $(MAXAS) -i -k gemv_q3_k gemv_q3_k_c.sass $@

# Separate opt-in artifacts avoid silently reusing a default image after an
# environment-only switch. Persistent die images also accept Q3_ROW_PREFETCH=1.
gemv_q3_k_prefetch_k2560.cubin: gemv_q3_k.sass q3_k_math.pl skel_q3_k.cubin
	cp skel_q3_k.cubin $@
	Q3_ROW_PREFETCH=1 GEMV_K=2560 $(MAXAS) -i gemv_q3_k.sass $@

gemv_q3_k_prefetch_k6144.cubin: gemv_q3_k.sass q3_k_math.pl skel_q3_k.cubin
	cp skel_q3_k.cubin $@
	Q3_ROW_PREFETCH=1 GEMV_K=6144 $(MAXAS) -i gemv_q3_k.sass $@

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

# the rolled-step IQ4_XS GEMV (gemv_iq4_xs_c.sass) in a skeleton with its staging shared memory; gemv_k_test takes it through GEMV_CUBIN
skel_iq4_xs_c.cubin: gemv_iq4_xs_c.ptx
	$(PTXAS11) -arch=sm_52 -o $@ $<

gemv_iq4_xs_c_k2560.cubin: gemv_iq4_xs_c.sass iq4_xs_math.pl skel_iq4_xs_c.cubin
	cp skel_iq4_xs_c.cubin $@
	GEMV_K=2560 $(MAXAS) -i -k gemv_iq4_xs gemv_iq4_xs_c.sass $@

gemv_iq4_xs_c_k6144.cubin: gemv_iq4_xs_c.sass iq4_xs_math.pl skel_iq4_xs_c.cubin
	cp skel_iq4_xs_c.cubin $@
	GEMV_K=6144 $(MAXAS) -i -k gemv_iq4_xs gemv_iq4_xs_c.sass $@

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

# `make test` is only these two Q2 GEMV checks (they need a free GPU and the model on the NFS share). The rest are the *_test programs, each built and run by name;
# the whole-model check is model_test, the sparse-attention checks qsa_test.
test: test-gemv
test-gemv: all
	./gemv_test $(SHARD1) blk.4.ffn_gate_exps.weight 0 20
	./gemv_test $(SHARD1) blk.4.ffn_down_exps.weight 0 20

clean:
	rm -f *.cubin *.dis *.so *.so.new die.sass die_q2_*.sass die_trace.sass die_trace16.sass dieops.h flashcore flashd flashvis hold memtest fence *_test ple_read_bench ple_read_audit

.PHONY: all test test-gemv clean

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

actq_k%.cubin: actq.sass actq.pl fmath.pl skel_actq.cubin
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

rmsnorm_k%.cubin: rmsnorm.sass rmsnorm.pl skel_rmsnorm.cubin
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

attn_test: attn_test.c test_guards.h ../ref/attn.c ../ref/attn.h ../ref/fmath.c ../ref/checked.h
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ attn_test.c ../ref/attn.c ../ref/fmath.c -I../ref -I$(CUDA)/include -lcuda -lm

.PHONY: attn
attn: attnqk.cubin attnqk_test attnpart.cubin attnmerge.cubin attn_test

# ---- MoE expert activation: SiLU(g) * u + per-32 quantiser ----
skel_ffnact.cubin: ffnact.ptx
	$(PTXAS11) -arch=sm_52 -o $@ $<

ffnact_k%.cubin: ffnact.sass actq.pl fmath.pl skel_ffnact.cubin
	cp skel_ffnact.cubin $@
	GEMV_K=$* $(MAXAS) -i ffnact.sass $@

ffnact_test: ffnact_test.c ../ref/quant.c ../ref/quant.h ../ref/gguf.c ../ref/fmath.c
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ ffnact_test.c ../ref/quant.c ../ref/gguf.c ../ref/fmath.c -I../ref -I$(CUDA)/include -lcuda -lm

.PHONY: ffnact
ffnact: ffnact_k640.cubin ffnact_test actq_k640.cubin actq_k2560.cubin actq_test

# ---- token embedding row (raw Q3_K decode, broadcast to the 4 lanes) ----
skel_embrow.cubin: embrow.ptx
	$(PTXAS11) -arch=sm_52 -o $@ $<

embrow.cubin: embrow.sass skel_embrow.cubin
	cp skel_embrow.cubin $@
	$(MAXAS) -i embrow.sass $@

embrow_test: embrow_test.c ../ref/quant.c ../ref/quant.h ../ref/gguf.c
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ embrow_test.c ../ref/quant.c ../ref/gguf.c -I../ref -I$(CUDA)/include -lcuda -lm

.PHONY: embrow
embrow: embrow.cubin embrow_test

# ---- argmax over the logits ----
skel_argmax.cubin: argmax.ptx
	$(PTXAS11) -arch=sm_52 -o $@ $<

argmax.cubin: argmax.sass skel_argmax.cubin
	cp skel_argmax.cubin $@
	$(MAXAS) -i argmax.sass $@

argmax_test: argmax_test.c
	$(CC) -std=gnu99 -O2 -Wall -o $@ argmax_test.c -I$(CUDA)/include -lcuda -lm

.PHONY: argmax
argmax: argmax.cubin argmax_test

# ---- PLE row hash + host-RAM gather + IQ4_NL decode ----
skel_plerows.cubin: plerows.ptx
	$(PTXAS11) -arch=sm_52 -o $@ $<

plerows.cubin: plerows.sass skel_plerows.cubin
	cp skel_plerows.cubin $@
	$(MAXAS) -i plerows.sass $@

plerows_test: plerows_test.c ../ref/ple.c ../ref/ple.h ../ref/quant.c ../ref/gguf.c ../ref/fmath.c
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ plerows_test.c ../ref/ple.c ../ref/quant.c ../ref/gguf.c ../ref/fmath.c -I../ref -I$(CUDA)/include -lcuda -lm

.PHONY: plerows
plerows: plerows.cubin plerows_test

# ---- PLE dilated conv + SiLU + residual + stream add ----
skel_pleconv.cubin: pleconv.ptx
	$(PTXAS11) -arch=sm_52 -o $@ $<

pleconv.cubin: pleconv.sass fmath.pl skel_pleconv.cubin
	cp skel_pleconv.cubin $@
	$(MAXAS) -i pleconv.sass $@

pleconv_test: pleconv_test.c ../ref/ple.c ../ref/ple.h ../ref/fmath.c ../ref/quant.c ../ref/gguf.c
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ pleconv_test.c ../ref/ple.c ../ref/fmath.c ../ref/quant.c ../ref/gguf.c -I../ref -I$(CUDA)/include -lcuda -lm

.PHONY: pleconv
pleconv: pleconv.cubin pleconv_test

plerows_ng.cubin: plerows.sass skel_plerows.cubin
	cp skel_plerows.cubin $@
	PLE_NOGATHER=1 $(MAXAS) -i plerows.sass $@

# ---- PLE lane gate ----
skel_plegate.cubin: plegate.ptx
	$(PTXAS11) -arch=sm_52 -o $@ $<

plegate.cubin: plegate.sass fmath.pl skel_plegate.cubin
	cp skel_plegate.cubin $@
	$(MAXAS) -i plegate.sass $@

plegate_test: plegate_test.c ../ref/ple.c ../ref/ple.h ../ref/fmath.c
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ plegate_test.c ../ref/ple.c ../ref/fmath.c -I../ref -I$(CUDA)/include -lcuda -lm

.PHONY: plegate ple
plegate: plegate.cubin plegate_test
ple: plerows.cubin plerows_ng.cubin plerows_test plegate pleconv

# ---- the persistent per-die kernel (die.pl stitches the ops together) ----
DIE_OPS = $(filter-out die.sass die_q2_%.sass die_q3_%.sass die_trace%.sass semtest%.sass bartest.sass fmath.sass gridbar.sass mailbox.sass xmadbench.sass, $(wildcard *.sass)) actq.pl rmsnorm.pl fmath.pl q3_k_math.pl
die.sass: die.pl trace.pl $(DIE_OPS)
	DIE_TRACE=0 perl die.pl > $@

dieops.h: die.pl
	perl die.pl header > $@

skel_die.cubin: die.ptx
	$(PTXAS11) -arch=sm_52 -o $@ $<

die.cubin: die.sass skel_die.cubin
	cp skel_die.cubin $@
	$(MAXAS) -i die.sass $@


die_q2_skip.sass: die.pl trace.pl $(DIE_OPS)
	Q2_RESIDENT_INPUT=0 Q2_COMPACT_ACTIVE=0 Q2_WEIGHT_TILE=0 Q2_POSITION_LOOP=0 Q2_SKIP_ABSENT=1 DIE_TRACE=0 perl die.pl > $@
die_q2_skip.cubin: die_q2_skip.sass skel_die.cubin
	cp skel_die.cubin $@
	$(MAXAS) -i die_q2_skip.sass $@

die_q2_loop.sass: die.pl trace.pl $(DIE_OPS)
	Q2_RESIDENT_INPUT=0 Q2_COMPACT_ACTIVE=0 Q2_WEIGHT_TILE=0 Q2_POSITION_LOOP=1 Q2_SKIP_ABSENT=0 DIE_TRACE=0 perl die.pl > $@
die_q2_loop_skip.sass: die.pl trace.pl $(DIE_OPS)
	Q2_RESIDENT_INPUT=0 Q2_COMPACT_ACTIVE=0 Q2_WEIGHT_TILE=0 Q2_POSITION_LOOP=1 Q2_SKIP_ABSENT=1 DIE_TRACE=0 perl die.pl > $@
die_q2_loop.cubin: die_q2_loop.sass skel_die.cubin
	cp skel_die.cubin $@
	$(MAXAS) -i die_q2_loop.sass $@
die_q2_loop_skip.cubin: die_q2_loop_skip.sass skel_die.cubin
	cp skel_die.cubin $@
	$(MAXAS) -i die_q2_loop_skip.sass $@

die_test: die_test.c hcup_batch_test.inc hcr_batch_test.inc prog.c dieops.h prog.h test_guards.h ../ref/quant.c ../ref/gguf.c ../ref/fmath.c ../ref/ple.c ../ref/ple.h ../ref/hc.c ../ref/hc.h ../ref/split.c ../ref/split.h
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ die_test.c prog.c ../ref/quant.c ../ref/gguf.c ../ref/fmath.c ../ref/ple.c ../ref/hc.c ../ref/split.c -I../ref -I$(CUDA)/include -lcuda -lm

.PHONY: die
die: die.cubin die_test

# ---- the resident weight holder (load once; tests map the arena over CUDA IPC) ----
hold: hold.c prog.c prog.h dieops.h ../ref/quant.c ../ref/gguf.c ../ref/quant.h ../ref/gguf.h ../ref/split.c ../ref/split.h
	$(CC) -std=gnu99 -O2 -Wall -o $@ hold.c prog.c ../ref/quant.c ../ref/gguf.c ../ref/split.c -I../ref -I$(CUDA)/include -lcuda -lm

# one process for all six dies, routed experts in 64-row units (ref/split2.h)
hold2: hold2.c prog.c prog.h dieops.h ../ref/quant.c ../ref/gguf.c ../ref/quant.h ../ref/gguf.h ../ref/split.c ../ref/split.h
	$(CC) -std=gnu99 -O2 -Wall -o $@ hold2.c prog.c ../ref/quant.c ../ref/gguf.c ../ref/split.c -I../ref -I$(CUDA)/include -lcuda -lm

# ---- DeltaNet readout norm + gate ----
skel_dnnorm.cubin: dnnorm.ptx
	$(PTXAS11) -arch=sm_52 -o $@ $<

dnnorm.cubin: dnnorm.sass fmath.pl skel_dnnorm.cubin
	cp skel_dnnorm.cubin $@
	$(MAXAS) -i dnnorm.sass $@

skel_dnnorm_xout.cubin: dnnorm_xout.ptx
	$(PTXAS11) -arch=sm_52 -o $@ $<

dnnorm_xout.cubin: dnnorm.sass fmath.pl skel_dnnorm_xout.cubin
	cp skel_dnnorm_xout.cubin $@
	DN_XOUT=1 $(MAXAS) -i dnnorm.sass $@

skel_hcops_xout.cubin: hcops_xout.ptx
	$(PTXAS11) -arch=sm_52 -o $@ $<

hc_mix_xout.cubin: hcops.sass fmath.pl skel_hcops_xout.cubin
	cp skel_hcops_xout.cubin $@
	HC_OP=mix HC_XOUT=1 $(MAXAS) -i hcops.sass $@

dnnorm_test: dnnorm_test.c ../ref/deltanet.c ../ref/deltanet.h ../ref/hc.c ../ref/hc.h ../ref/fmath.c
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ dnnorm_test.c ../ref/deltanet.c ../ref/hc.c ../ref/fmath.c -I../ref -I$(CUDA)/include -lcuda -lm

.PHONY: dnnorm
dnnorm: dnnorm.cubin dnnorm_xout.cubin hc_mix_xout.cubin dnnorm_test

# ---- MoE / accumulate / bias GEMV variants ----
gemv_q2_0_moeg.cubin: gemv_q2_0.sass skel_q2_0.cubin
	cp skel_q2_0.cubin $@
	GEMV_K=2560 GEMV_MOE=gather $(MAXAS) -i gemv_q2_0.sass $@

gemv_q2_0_moed.cubin: gemv_q2_0.sass skel_q2_0.cubin
	cp skel_q2_0.cubin $@
	GEMV_K=640 GEMV_MOE=down $(MAXAS) -i gemv_q2_0.sass $@

gemv_q2_0_acc640.cubin: gemv_q2_0.sass skel_q2_0.cubin
	cp skel_q2_0.cubin $@
	GEMV_K=640 GEMV_ACC=1 $(MAXAS) -i gemv_q2_0.sass $@

gemv_bf16_bias.cubin: gemv_bf16.sass skel_bf16.cubin
	cp skel_bf16.cubin $@
	GEMV_K=2560 GEMV_BIAS=1 $(MAXAS) -i gemv_bf16.sass $@

moe_test: moe_test.c ../ref/quant.c ../ref/gguf.c ../ref/fmath.c
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ moe_test.c ../ref/quant.c ../ref/gguf.c ../ref/fmath.c -I../ref -I$(CUDA)/include -lcuda -lm

.PHONY: moe
moe: gemv_q2_0_moeg.cubin gemv_q2_0_moed.cubin gemv_q2_0_acc640.cubin gemv_bf16_bias.cubin moe_test

# ---- 6-die all-gather microbenchmark ----
skel_allgather.cubin: allgather.ptx
	$(PTXAS11) -arch=sm_52 -o $@ $<

allgather.cubin: allgather.sass skel_allgather.cubin
	cp skel_allgather.cubin $@
	$(MAXAS) -i allgather.sass $@

allgather_ag4.cubin: allgather.sass skel_allgather.cubin
	cp skel_allgather.cubin $@
	AG_CHUNKS=4 $(MAXAS) -i allgather.sass $@

allgather_test: allgather_test.c
	$(CC) -std=gnu99 -O2 -Wall -o $@ allgather_test.c -I$(CUDA)/include -lcuda -lm

.PHONY: allgather
allgather: allgather.cubin allgather_ag4.cubin allgather_test

allgather_flags.cubin: allgather.sass skel_allgather.cubin
	cp skel_allgather.cubin $@
	AG_NOPAY=1 $(MAXAS) -i allgather.sass $@

allgather_nover.cubin: allgather.sass skel_allgather.cubin
	cp skel_allgather.cubin $@
	AG_NOVERIFY=1 $(MAXAS) -i allgather.sass $@

# ---- a block through the persistent kernel(s) vs the C block twin: n dies ----
dies_test: dies_test.c prog.c prog.h holdc.c hold.h dieops.h ../ref/block.c ../ref/block.h ../ref/attn.c ../ref/ple.c ../ref/quant.c ../ref/gguf.c ../ref/fmath.c ../ref/hc.c ../ref/deltanet.c ../ref/router.c ../ref/split.c ../ref/split.h
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ dies_test.c prog.c holdc.c ../ref/block.c ../ref/attn.c ../ref/ple.c ../ref/quant.c ../ref/gguf.c ../ref/fmath.c ../ref/hc.c ../ref/deltanet.c ../ref/router.c ../ref/split.c -I../ref -I$(CUDA)/include -lcuda -lm
chunk_test: chunk_test.c prog_chunk.c prog_chunk.h prog.c prog.h holdc.c hold.h dieops.h ../ref/block.c ../ref/block.h ../ref/attn.c ../ref/ple.c ../ref/quant.c ../ref/gguf.c ../ref/fmath.c ../ref/hc.c ../ref/deltanet.c ../ref/router.c ../ref/split.c ../ref/split.h
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ chunk_test.c prog_chunk.c prog.c holdc.c ../ref/block.c ../ref/attn.c ../ref/ple.c ../ref/quant.c ../ref/gguf.c ../ref/fmath.c ../ref/hc.c ../ref/deltanet.c ../ref/router.c ../ref/split.c -I../ref -I$(CUDA)/include -lcuda -lm

# ---- whole tokens (embed, blocks [b0,b1), head) on n dies vs the twins ----
model_test: model_test.c prog.c prog.h holdc.c hold.h dieops.h ../ref/block.c ../ref/block.h ../ref/attn.c ../ref/ple.c ../ref/quant.c ../ref/gguf.c ../ref/fmath.c ../ref/hc.c ../ref/deltanet.c ../ref/router.c ../ref/split.c ../ref/split.h
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ model_test.c prog.c holdc.c ../ref/block.c ../ref/attn.c ../ref/ple.c ../ref/quant.c ../ref/gguf.c ../ref/fmath.c ../ref/hc.c ../ref/deltanet.c ../ref/router.c ../ref/split.c -I../ref -I$(CUDA)/include -lcuda -lm

# the persistent core and the reloadable engine library (see core.h); the library is built to a new name and moved
# over the old one, so a running core that has the old file mapped keeps a consistent image
ENGINE_SRC = engine.c prog.c prog_chunk.c holdc.c ../ref/qsa.c ../ref/block.c ../ref/attn.c ../ref/ple.c ../ref/quant.c ../ref/gguf.c ../ref/fmath.c ../ref/hc.c ../ref/deltanet.c ../ref/router.c ../ref/split.c

trace_session_test: trace_session_test.c $(ENGINE_SRC) prefill.h trace_host.inc timings.h engine.h core.h prog.h hold.h dieops.h ple_read_pool.h ple_read_reload.h
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -UNDEBUG -pthread -o $@ trace_session_test.c $(filter-out engine.c,$(ENGINE_SRC)) -I../ref -I$(CUDA)/include -lcuda -lm
# the commit (and -dirty when the tree differs from it) the library is built from: the engine reports it with its `config` command
FLASH_REV ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)$(shell git diff --quiet HEAD -- .. 2>/dev/null || echo -dirty)
libengine.so: trace_host.inc engine_chunk.c prog_chunk.h $(ENGINE_SRC) prefill.h prog.h hold.h core.h engine.h dieops.h ple_read_pool.h ple_read_reload.h ../ref/split.h ../ref/qsa.h ../ref/block.h ../ref/attn.h ../ref/ple.h ../ref/gguf.h ../ref/quant.h ../ref/fmath.h ../ref/hc.h ../ref/deltanet.h ../ref/router.h timings.h
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -fPIC -shared -pthread -DFLASH_REV='"$(FLASH_REV)"' -o $@.new $(ENGINE_SRC) -I../ref -I$(CUDA)/include -lcuda -lm && mv -f $@.new $@

flashcore: flashcore.c core.h hold.h holdc.c ../ref/gguf.c ../ref/gguf.h
	$(CC) -std=gnu99 -O2 -Wall -o $@ flashcore.c holdc.c ../ref/gguf.c -I../ref -I$(CUDA)/include -lcuda -ldl -lm

nullengine.so: nullengine.c core.h
	$(CC) -std=gnu99 -O2 -Wall -fPIC -shared -o $@.new nullengine.c -I../ref -I$(CUDA)/include && mv -f $@.new $@

tok_test: tok_test.c tok.c tok.h ucd.h ../ref/gguf.c ../ref/gguf.h
	$(CC) -std=gnu99 -O2 -Wall -o $@ tok_test.c tok.c ../ref/gguf.c -I../ref

jinja_test: jinja_test.c jinja.c jinja.h json.c json.h
	$(CC) -std=gnu99 -O2 -Wall -o $@ jinja_test.c jinja.c json.c -lm

flashd: timings.h flashd.c tok.c tok.h jinja.c jinja.h json.c json.h ucd.h ../ref/gguf.c ../ref/gguf.h
	$(CC) -std=gnu99 -O2 -Wall -o $@ flashd.c tok.c jinja.c json.c ../ref/gguf.c -I../ref -lm -lpthread

sample_test: sample_test.c dieops.h ../ref/sample.c ../ref/sample.h ../ref/fmath.c
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ sample_test.c ../ref/sample.c ../ref/fmath.c -I../ref -I$(CUDA)/include -lcuda -lm

# ---- QSA: int8 KV cache, indexer, selection, gather through the persistent kernel vs ref/qsa.c ----
qsa_test: qsa_test.c quant_range_cases.h test_guards.h qsa_padding_test.h dieops.h ../ref/qsa.c ../ref/qsa.h ../ref/attn.c ../ref/attn.h ../ref/fmath.c ../ref/quant.c ../ref/gguf.c ../ref/checked.h
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ qsa_test.c ../ref/qsa.c ../ref/attn.c ../ref/fmath.c ../ref/quant.c ../ref/gguf.c -I../ref -I$(CUDA)/include -lcuda -lm

attnpartn_unitmap: attnpartn_unitmap.c
	$(CC) -std=gnu99 -O2 -Wall -Wextra -o $@ $<

# ---- the vision encoder daemon (libmtmd from a llama.cpp build; LLAMA_CPP names its tree) ----
LLAMA_CPP ?= $(HOME)/claude/llama.cpp-up
flashvis: flashvis.c
	$(CC) -std=gnu99 -O2 -Wall -o $@ flashvis.c -I$(LLAMA_CPP)/include -I$(LLAMA_CPP)/ggml/include -I$(LLAMA_CPP)/tools/mtmd -L$(LLAMA_CPP)/build/bin -lmtmd -lllama -lggml -lggml-base -Wl,-rpath,$(LLAMA_CPP)/build/bin

memtest: memtest.c
	$(CC) -std=gnu99 -O2 -Wall -o $@ memtest.c -I$(CUDA)/include -lcuda

fence: fence.c
	$(CC) -std=gnu99 -O2 -Wall -o $@ fence.c -I$(CUDA)/include -lcuda

quant_range_test: quant_range_test.c quant_range_cases.h ../ref/qsa.c ../ref/qsa.h ../ref/fmath.c ../ref/fmath.h
	$(CC) -std=gnu99 -O2 -ffp-contract=off -UNDEBUG -Wall -Wextra -o $@ quant_range_test.c ../ref/qsa.c ../ref/fmath.c -I../ref -lm
.PHONY: test-quant-range
quant_bench: quant_bench.c dieops.h
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -Wextra -DFLASH_REV='"$(FLASH_REV)"' -o $@ quant_bench.c -I$(CUDA)/include -lcuda -lm

test-quant-range: quant_range_test
	./quant_range_test

# Disk PLE readers and read-only file I/O measurements.
ple_read_test: ple_read_test.c ple_read_pool.h ple_read_reload.h
	$(CC) -std=gnu99 -O2 -Wall -Wextra -pthread -o $@ $<

ple_read_bench: ple_read_bench.c ple_read_pool.h
	$(CC) -std=gnu99 -O2 -Wall -Wextra -pthread -DFLASH_REV='"$(FLASH_REV)"' -o $@ $<

ple_read_audit: ple_read_bench.c ple_read_pool.h
	$(CC) -std=gnu99 -O2 -Wall -Wextra -pthread -DPLE_READ_AUDIT -DFLASH_REV='"$(FLASH_REV)"' -o $@ $<

qsa_direct_test: qsa_direct_test.c qsa_direct_projection.h qsa_direct_fixtures.h qsa_test.c quant_range_cases.h test_guards.h qsa_padding_test.h prog.c prog.h dieops.h ../ref/qsa.c ../ref/qsa.h ../ref/block.c ../ref/block.h ../ref/attn.c ../ref/ple.c ../ref/quant.c ../ref/gguf.c ../ref/fmath.c ../ref/hc.c ../ref/deltanet.c ../ref/router.c ../ref/split.c ../ref/checked.h
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -DFLASH_REV='"$(FLASH_REV)"' -o $@ qsa_direct_test.c prog.c ../ref/qsa.c ../ref/block.c ../ref/attn.c ../ref/ple.c ../ref/quant.c ../ref/gguf.c ../ref/fmath.c ../ref/hc.c ../ref/deltanet.c ../ref/router.c ../ref/split.c -I../ref -I$(CUDA)/include -lcuda -lm

# Two-position descriptor and state equivalence tests.
pair_prog_test: pair_prog_test.c prog.c prog.h dieops.h ../ref/quant.c ../ref/gguf.c ../ref/split.c
	$(CC) -std=gnu99 -O2 -Wall -o $@ pair_prog_test.c prog.c ../ref/quant.c ../ref/gguf.c ../ref/split.c -I../ref -I$(CUDA)/include -lm

pair_state_test: pair_state_test.c test_guards.h prog.c prog.h dieops.h ../ref/quant.c ../ref/gguf.c ../ref/split.c ../ref/deltanet.c ../ref/deltanet.h ../ref/fmath.c
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ pair_state_test.c prog.c ../ref/quant.c ../ref/gguf.c ../ref/split.c ../ref/deltanet.c ../ref/fmath.c -I../ref -I$(CUDA)/include -lcuda -lm

group_attention_test: group_attention_test.c test_guards.h prog.c prog.h dieops.h ../ref/qsa.c ../ref/attn.c ../ref/quant.c ../ref/gguf.c ../ref/fmath.c ../ref/split.c
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -Wextra -o $@ group_attention_test.c prog.c ../ref/qsa.c ../ref/attn.c ../ref/quant.c ../ref/gguf.c ../ref/fmath.c ../ref/split.c -I../ref -I$(CUDA)/include -lcuda -lm

pair_bf16_test: pair_bf16_test.c dieops.h
	$(CC) -std=gnu99 -O2 -Wall -o $@ pair_bf16_test.c -I$(CUDA)/include -lcuda

pair_bf16_plan_test: pair_bf16_plan_test.c prog.c prog.h dieops.h ../ref/quant.c ../ref/gguf.c ../ref/split.c
	$(CC) -std=gnu99 -O2 -Wall -o $@ pair_bf16_plan_test.c prog.c ../ref/quant.c ../ref/gguf.c ../ref/split.c -I../ref -I$(CUDA)/include -lm

# Profiling images use separate generated files; changing a shell variable cannot leave a stale production image.
die_trace.sass: die.pl trace.pl $(DIE_OPS)
	DIE_TRACE=1 perl die.pl > $@
die_trace16.sass: die.pl trace.pl $(DIE_OPS)
	DIE_TRACE=16 perl die.pl > $@
die_trace.cubin: die_trace.sass skel_die.cubin
	cp skel_die.cubin $@
	$(MAXAS) -i die_trace.sass $@
die_trace16.cubin: die_trace16.sass skel_die.cubin
	cp skel_die.cubin $@
	$(MAXAS) -i die_trace16.sass $@

timings_test: timings_test.c timings.h
	$(CC) -std=gnu99 -O2 -Wall -Wextra -UNDEBUG -o $@ timings_test.c -lm

flashd_timing_test: flashd_timing_test.c flashd.c timings.h tok.c tok.h jinja.c jinja.h json.c json.h ucd.h ../ref/gguf.c ../ref/gguf.h
	$(CC) -std=gnu99 -O2 -Wall -UNDEBUG -o $@ flashd_timing_test.c tok.c jinja.c json.c ../ref/gguf.c -I../ref -lm -lpthread
# This planner test needs only a C compiler.
prefill_plan_test: prefill_plan_test.c prefill.h
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror -o $@ prefill_plan_test.c

output_tokens_test: output_tokens_test.c flashd.c tok.c tok.h jinja.c json.c ../ref/gguf.c
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ output_tokens_test.c tok.c jinja.c json.c ../ref/gguf.c -I../ref -lm -lpthread

dn_l2norm_test: dn_l2norm_test.c ../ref/deltanet.c ../ref/fmath.c
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ dn_l2norm_test.c ../ref/deltanet.c ../ref/fmath.c -I../ref -lm

ple_eos_test: ple_eos_test.c ../ref/ple.c ../ref/fmath.c
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ ple_eos_test.c ../ref/ple.c ../ref/fmath.c -I../ref -lm

checked_test: checked_test.c ../ref/checked.c ../ref/checked.h ../ref/attn.c ../ref/qsa.c ../ref/fmath.c
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ checked_test.c ../ref/checked.c ../ref/attn.c ../ref/qsa.c ../ref/fmath.c -I../ref -lm

# Validation-only library: real engine requests with allocation canaries.
libengine_reuse_test.so: engine_reuse_test.c $(ENGINE_SRC) prefill.h prog.h hold.h core.h engine.h dieops.h trace_host.inc timings.h ple_read_pool.h ple_read_reload.h ../ref/split.h ../ref/qsa.h ../ref/block.h ../ref/attn.h ../ref/ple.h ../ref/gguf.h ../ref/quant.h ../ref/fmath.h ../ref/hc.h ../ref/deltanet.h ../ref/router.h
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -fPIC -shared -pthread -DFLASH_REV='"$(FLASH_REV)"' -o $@.new engine_reuse_test.c $(filter-out engine.c,$(ENGINE_SRC)) -I../ref -I$(CUDA)/include -lcuda -lm && mv -f $@.new $@

# CPU-only validation of the PLE reader, benchmark records and launch guards.
ple_read_bench_test: ple_read_bench_test.c ple_read_bench.c ple_read_pool.h
	$(CC) -std=gnu99 -O2 -Wall -Wextra -pthread -o $@ $<

ple_read_audit_test: ple_read_bench_test.c ple_read_bench.c ple_read_pool.h
	$(CC) -std=gnu99 -O2 -Wall -Wextra -pthread -DPLE_READ_AUDIT -o $@ $<

.PHONY: test-ple
test-ple: ple_read_test ple_read_bench_test ple_read_audit_test
	./ple_read_test
	./ple_read_bench_test
	./ple_read_audit_test
	bash ./ple_prewarm_test.sh
	bash ./ple_io_run_test.sh
	bash ./ple_io_validate_test.sh

# Run this GPU fixture only through gpuq; --cpu-only checks host eligibility.
batch_rms_test: batch_rms_test.c prog.c prog.h dieops.h test_guards.h ../ref/quant.c ../ref/gguf.c ../ref/split.c
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ batch_rms_test.c prog.c ../ref/quant.c ../ref/gguf.c ../ref/split.c -I../ref -I$(CUDA)/include -lcuda -lm

batch_actq_test: batch_actq_test.c prog.c prog.h dieops.h test_guards.h ../ref/quant.c ../ref/gguf.c ../ref/split.c
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ batch_actq_test.c prog.c ../ref/quant.c ../ref/gguf.c ../ref/split.c -I../ref -I$(CUDA)/include -lcuda -lm

batch_q2_test: batch_q2_test.c prog.c prog.h dieops.h ../ref/quant.c ../ref/gguf.c ../ref/split.c
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ batch_q2_test.c prog.c ../ref/quant.c ../ref/gguf.c ../ref/split.c -I../ref -I$(CUDA)/include -lcuda -lm

# Dense-only position-parallel component image; routed bodies retain control.
die_q2_parallel_dense.sass: die.pl trace.pl $(DIE_OPS)
	Q2_RESIDENT_INPUT=0 Q2_COMPACT_ACTIVE=0 Q2_WEIGHT_TILE=0 Q2_PARALLEL_DENSE=1 Q2_POSITION_PARALLEL=0 Q2_POSITION_LOOP=0 Q2_SKIP_ABSENT=0 DIE_TRACE=0 perl die.pl > $@
die_q2_parallel_dense.cubin: die_q2_parallel_dense.sass skel_die.cubin
	cp skel_die.cubin $@
	$(MAXAS) -i die_q2_parallel_dense.sass $@

# Full position-parallel image includes packed routed incidence and rank tiles.
die_q2_parallel.sass: die.pl trace.pl $(DIE_OPS)
	Q2_RESIDENT_INPUT=0 Q2_COMPACT_ACTIVE=0 Q2_WEIGHT_TILE=0 Q2_POSITION_PARALLEL=1 Q2_PARALLEL_DENSE=0 Q2_POSITION_LOOP=0 Q2_SKIP_ABSENT=0 DIE_TRACE=0 perl die.pl > $@
die_q2_parallel.cubin: die_q2_parallel.sass skel_die.cubin
	cp skel_die.cubin $@
	$(MAXAS) -i die_q2_parallel.sass $@

# Canonical signed-byte inputs leave room for a complete shared weight tile.
die_q2_tile.sass: die.pl trace.pl $(DIE_OPS)
	Q2_RESIDENT_INPUT=0 Q2_COMPACT_ACTIVE=0 Q2_WEIGHT_TILE=1 Q2_POSITION_PARALLEL=0 Q2_PARALLEL_DENSE=0 Q2_POSITION_LOOP=0 Q2_SKIP_ABSENT=0 DIE_TRACE=0 perl die.pl > $@
die_q2_tile.cubin: die_q2_tile.sass skel_die.cubin
	cp skel_die.cubin $@
	$(MAXAS) -i die_q2_tile.sass $@

# Compacted routed groups visit one/two/four/eight active-position slots.
die_q2_compact.sass: die.pl trace.pl $(DIE_OPS)
	Q2_RESIDENT_INPUT=0 Q2_COMPACT_ACTIVE=1 Q2_WEIGHT_TILE=1 Q2_POSITION_PARALLEL=0 Q2_PARALLEL_DENSE=0 Q2_POSITION_LOOP=0 Q2_SKIP_ABSENT=0 DIE_TRACE=0 perl die.pl > $@
die_q2_compact.cubin: die_q2_compact.sass skel_die.cubin
	cp skel_die.cubin $@
	$(MAXAS) -i die_q2_compact.sass $@
# Focused Q3 B8 scalar GPU / independent C reference parity (run via gpuq).
batch_q3_test: batch_q3_test.c prog.c prog.h dieops.h ../ref/quant.c ../ref/gguf.c ../ref/split.c
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ batch_q3_test.c prog.c ../ref/quant.c ../ref/gguf.c ../ref/split.c -I../ref -I$(CUDA)/include -lcuda -lm

# Direct int16 inputs stay in registers across all rows for the current expert.
die_q2_resident.sass: die.pl trace.pl $(DIE_OPS)
	Q2_RESIDENT_INPUT=1 Q2_COMPACT_ACTIVE=1 Q2_WEIGHT_TILE=1 Q2_POSITION_PARALLEL=0 Q2_PARALLEL_DENSE=0 Q2_POSITION_LOOP=0 Q2_SKIP_ABSENT=0 DIE_TRACE=0 perl die.pl > $@
die_q2_resident.cubin: die_q2_resident.sass skel_die.cubin
	cp skel_die.cubin $@
	$(MAXAS) -i die_q2_resident.sass $@
# Same Q3 B8 opcode ABI, with integer weights decoded once per shared tile.
die_q3_decode.sass: die.pl trace.pl $(DIE_OPS)
	Q3_WEIGHT_DECODE=1 DIE_TRACE=0 perl die.pl > $@
die_q3_decode.cubin: die_q3_decode.sass skel_die.cubin
	cp skel_die.cubin $@
	$(MAXAS) -i die_q3_decode.sass $@

# the K-split MoE ops of the unit layout (needs a die cubin built with them)
moe_units_test: moe_units_test.c test_guards.h dieops.h ../ref/quant.c ../ref/quant.h ../ref/fmath.c ../ref/fmath.h ../ref/gguf.c
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ moe_units_test.c ../ref/quant.c ../ref/fmath.c ../ref/gguf.c -I../ref -I$(CUDA)/include -lcuda -lm

# the prefill GEMMs (GEMM_FMT=<fmt> gen_gemm.lua -> gemm_<fmt>.sass; the skeletons carry the 16 KB of tile buffers)
gemm_%.sass: gen_gemm.lua
	GEMM_FMT=$* lua5.4 gen_gemm.lua > $@

skel_gemm_%.cubin: gemm_%.ptx
	$(PTXAS11) -arch=sm_52 -o $@ $<

gemm_%.cubin: gemm_%.sass skel_gemm_%.cubin
	cp skel_gemm_$*.cubin $@
	$(MAXAS) -i gemm_$*.sass $@

.SECONDARY: skel_gemm_q4k.cubin skel_gemm_q5k.cubin skel_gemm_q6k.cubin skel_gemm_q40.cubin skel_gemm_q50.cubin skel_gemm_iq4nl.cubin skel_gemm_q80.cubin skel_gemm_bf16.cubin skel_gemm_q2.cubin skel_gemm_q3.cubin skel_gemm_iq4.cubin   # kept for debug builds
gemm_q4k_test: gemm_q4k_test.c ../ref/gguf.c ../ref/quant.c ../ref/gguf.h ../ref/quant.h
	$(CC) -std=gnu99 -O2 -Wall -o $@ gemm_q4k_test.c ../ref/gguf.c ../ref/quant.c -I../ref -I$(CUDA)/include -lcuda -lm
.PHONY: gemm
gemm: gemm_q4k.cubin gemm_q5k.cubin gemm_q6k.cubin gemm_q40.cubin gemm_q50.cubin gemm_iq4nl.cubin gemm_q80.cubin gemm_bf16.cubin gemm_q2.cubin gemm_q3.cubin gemm_iq4.cubin gemm_q4k_test

# the same GEMMs as ops of the die kernel (die.pl lists gemm_<fmt>_op.sass) and their test through the die kernel
gemm_%_op.sass: gen_gemm.lua
	GEMM_OP=1 GEMM_FMT=$* lua5.4 gen_gemm.lua > $@
GEMM_OPS = gemm_q4k_op.sass gemm_bf16_op.sass gemm_q2_op.sass gemm_q3_op.sass gemm_iq4_op.sass gemm_q5k_op.sass gemm_q6k_op.sass gemm_q40_op.sass gemm_q50_op.sass gemm_iq4nl_op.sass gemm_q80_op.sass
die.sass: $(GEMM_OPS)

# the prefill chunk program's slab ops
slab_%.sass: slab.lua
	SLAB_OP=$* lua5.4 slab.lua > $@
SLAB_OPS = slab_silu.sass slab_mix.sass slab_write.sass slab_rms.sass slab_conv.sass slab_bias.sass slab_router.sass slab_tr.sass slab_group.sass slab_swiglu.sass slab_saxpy.sass slab_zero.sass slab_rs.sass slab_sum.sass slab_pgate.sass slab_pconv.sass slab_xs.sass slab_xh.sass slab_uh.sass
die.sass: $(SLAB_OPS)
ebatch_%.sass: ebatch.lua
	EB_OP=$* lua5.4 ebatch.lua > $@
EBATCH_OPS = ebatch_gu.sass ebatch_dn.sass ebatch_gu2.sass
die.sass: $(EBATCH_OPS)
cattn_%.sass: cattn.lua
	CA_OP=$* lua5.4 cattn.lua > $@
CATTN_OPS = cattn_kv.sass cattn_idx.sass cattn_sel.sass cattn_app.sass cattn_attn.sass
die.sass: $(CATTN_OPS)
slab_test: slab_test.c dieops.h ../ref/deltanet.c ../ref/router.c ../ref/fmath.c ../ref/ple.c ../ref/qsa.c
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ slab_test.c ../ref/deltanet.c ../ref/router.c ../ref/fmath.c ../ref/ple.c ../ref/qsa.c -I../ref -I$(CUDA)/include -lcuda -lm
gemm_die_test: gemm_die_test.c dieops.h ../ref/gguf.c ../ref/quant.c ../ref/gguf.h ../ref/quant.h
	$(CC) -std=gnu99 -O2 -Wall -o $@ gemm_die_test.c ../ref/gguf.c ../ref/quant.c -I../ref -I$(CUDA)/include -lcuda -lm
cattn_test: cattn_test.c dieops.h ../ref/attn.c ../ref/qsa.c ../ref/fmath.c
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ cattn_test.c ../ref/attn.c ../ref/qsa.c ../ref/fmath.c -I../ref -I$(CUDA)/include -lcuda -lm

# dief_<name>.cubin: the die kernel with the env in DIE_ENV (make dief_x.cubin DIE_ENV="DIE_BEACON=1 ..."), every thread
# fencing its stores at the phase end (diefence.lua); a MaxAs complaint removes the cubin
DIEF_DEPS = die.pl trace.pl diefence.lua $(filter-out dief_%.sass die_%.sass %_dbg.sass,$(DIE_OPS)) $(SLAB_OPS) $(EBATCH_OPS) $(CATTN_OPS) $(GEMM_OPS)
dief_%.sass: $(DIEF_DEPS)
	env ATTN_SHORT_TILE=1 ROUTER_LOOP=1 DIE_TRACE=0 $(DIE_ENV) perl die.pl | lua5.4 diefence.lua > $@
.PRECIOUS: dief_%.sass
dief_%.cubin: dief_%.sass skel_die.cubin
	cp skel_die.cubin $@ && chmod 644 $@
	$(MAXAS) -i $< $@ > dief_$*.maxas.log 2>&1; tail -1 dief_$*.maxas.log
	@if grep -qi "badly formed\|error\|unknown\|not a \|called at" dief_$*.maxas.log; then rm -f $@; echo "dief: maxas failed for $*"; exit 1; fi
	@sha256sum $@
