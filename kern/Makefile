CUDA ?= /opt/cuda
.DEFAULT_GOAL := all
PTXAS11 ?= /opt/cuda-11.4/bin/ptxas
MAXAS = PATH=$(CUDA)/bin:$$PATH perl -I../maxas/lib ../maxas/bin/maxas.pl
CC ?= gcc
SHARD1 ?= /mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf

.PHONY: vision-all vision-core
vision-all:
	env FLASH_MAXIMG=16384 ./serve.sh all
vision-core:
	env FLASH_MAXIMG=16384 ./serve.sh core

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

# `make test` is only these two Q2 GEMV checks (they need a free GPU and the model on the NFS share). The rest are the *_test programs, each built and run by name;
# the whole-model check is model_test, the sparse-attention checks qsa_test.
test: test-gemv
test-gemv: all
	./gemv_test $(SHARD1) blk.4.ffn_gate_exps.weight 0 20
	./gemv_test $(SHARD1) blk.4.ffn_down_exps.weight 0 20

clean:
	rm -f *.cubin *.dis *.so *.so.new die.sass die_trace.sass die_trace16.sass dieops.h flashcore flashd flashvis hold memtest fence *_test ple_read_bench ple_read_audit

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
DIE_OPS = $(filter-out die.sass die_fused.sass die_chunk.sass die_chunk_fp8.sass die_exact.sass die_gr.sass die_phase.sass die_balanced%.sass die_phase_replica.sass die_phase_replica_xtrace.sass die_phase_grhash.sass die_phase_serial.sass die_deterministic.sass die_phase_xtrace.sass die_deterministic_xtrace.sass die_gr_fp8.sass die_verify.sass die_mtp.sass die_collective.sass mtp_q8_%.sass die_trace%.sass semtest%.sass bartest.sass fmath.sass gridbar.sass mailbox.sass xmadbench.sass, $(wildcard *.sass)) actq.pl fmath.pl
die.sass: die.pl trace.pl $(DIE_OPS)
	DIE_TRACE=0 perl die.pl > $@

dieops.h: die.pl
	perl die.pl header > $@

skel_die.cubin: die.ptx
	$(PTXAS11) -arch=sm_52 -o $@ $<

die.cubin: die.sass skel_die.cubin
	cp skel_die.cubin $@
	$(MAXAS) -i die.sass $@

decode_fuse: decode_fuse.c
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror -o $@ $<

die_fused.sass: die.sass decode_fuse
	./decode_fuse die.sass > $@.tmp
	mv $@.tmp $@

skel_die_fused.cubin: die_fused.ptx
	$(PTXAS11) -arch=sm_52 -o $@ $<

die_fused.cubin: die_fused.sass skel_die_fused.cubin
	cp skel_die_fused.cubin $@
	$(MAXAS) -i -k die die_fused.sass $@

moe_fused_test: moe_fused_test.c prog.c prog.h dieops.h ../ref/quant.c ../ref/gguf.c ../ref/fmath.c ../ref/split.c
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -Wextra -o $@ moe_fused_test.c prog.c ../ref/quant.c ../ref/gguf.c ../ref/fmath.c ../ref/split.c -I../ref -I$(CUDA)/include -lcuda -lm

resident_decode_bench: resident_decode_bench.c
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror -o $@ $<

balance_report: balance/report.c json.c json.h
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror -o $@ balance/report.c json.c -lm

fence_hold: balance/fence_hold.c
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror -I$(CUDA)/include -o $@ $< -lcuda

mtp_q8_gen: mtp_q8_gen.c
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror -o $@ $<

mtp_q8_%.sass: mtp_q8_gen
	./mtp_q8_gen $* > $@

skel_mtp_q8.cubin: mtp_q8.ptx
	$(PTXAS11) -arch=sm_52 -o $@ $<

mtp_q8_%.cubin: mtp_q8_%.sass skel_mtp_q8.cubin
	cp skel_mtp_q8.cubin $@
	$(MAXAS) -i -k mtp_q8 $< $@

die_test: die_test.c dieops.h ../ref/quant.c ../ref/gguf.c ../ref/fmath.c ../ref/ple.c ../ref/ple.h
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ die_test.c ../ref/quant.c ../ref/gguf.c ../ref/fmath.c ../ref/ple.c -I../ref -I$(CUDA)/include -lcuda -lm

.PHONY: die
die: die.cubin die_test

# ---- the resident weight holder (load once; tests map the arena over CUDA IPC) ----
hold: hold.c hold_selection.h prog.c prog.h dieops.h balance/layout.h balance/embedding.h balance/program.h ../ref/quant.c ../ref/gguf.c ../ref/quant.h ../ref/gguf.h ../ref/split.c ../ref/split.h
	$(CC) -std=gnu99 -O2 -Wall -o $@ hold.c prog.c ../ref/quant.c ../ref/gguf.c ../ref/split.c -I../ref -I$(CUDA)/include -lcuda -lm

native_weights: balance/native_weights.c hold_selection.h holdc.c hold.h prog.c prog.h balance/layout.h balance/embedding.h unique/layout.h unique/columns.h ../ref/split.c ../ref/split.h ../ref/quant.c ../ref/gguf.c
	$(CC) -std=gnu99 -O2 -Wall -o $@ balance/native_weights.c holdc.c prog.c ../ref/split.c ../ref/quant.c ../ref/gguf.c -I. -I../ref -I$(CUDA)/include -lcuda -lm

# ---- DeltaNet readout norm + gate ----
skel_dnnorm.cubin: dnnorm.ptx
	$(PTXAS11) -arch=sm_52 -o $@ $<

dnnorm.cubin: dnnorm.sass fmath.pl skel_dnnorm.cubin
	cp skel_dnnorm.cubin $@
	$(MAXAS) -i dnnorm.sass $@

dnnorm_test: dnnorm_test.c ../ref/deltanet.c ../ref/deltanet.h ../ref/fmath.c
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ dnnorm_test.c ../ref/deltanet.c ../ref/fmath.c -I../ref -I$(CUDA)/include -lcuda -lm

.PHONY: dnnorm
dnnorm: dnnorm.cubin dnnorm_test

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
dies_test: dies_test.c prog.c prog.h holdc.c hold.h dieops.h ../ref/block.c ../ref/block.h ../ref/attn.c ../ref/ple.c ../ref/quant.c ../ref/gguf.c ../ref/fmath.c ../ref/hc.c ../ref/deltanet.c ../ref/router.c ../ref/split.c ../ref/split.h ../ref/qsa.c ../ref/qsa.h
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ dies_test.c prog.c holdc.c ../ref/qsa.c ../ref/block.c ../ref/attn.c ../ref/ple.c ../ref/quant.c ../ref/gguf.c ../ref/fmath.c ../ref/hc.c ../ref/deltanet.c ../ref/router.c ../ref/split.c -I../ref -I$(CUDA)/include -lcuda -lm

# ---- whole tokens (embed, blocks [b0,b1), head) on n dies vs the twins ----
model_test: model_test.c prog.c prog.h holdc.c hold.h dieops.h ../ref/block.c ../ref/block.h ../ref/attn.c ../ref/ple.c ../ref/quant.c ../ref/gguf.c ../ref/fmath.c ../ref/hc.c ../ref/deltanet.c ../ref/router.c ../ref/split.c ../ref/split.h ../ref/qsa.c ../ref/qsa.h model_qsa_test.h qsa_state.h
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ model_test.c prog.c holdc.c ../ref/qsa.c ../ref/block.c ../ref/attn.c ../ref/ple.c ../ref/quant.c ../ref/gguf.c ../ref/fmath.c ../ref/hc.c ../ref/deltanet.c ../ref/router.c ../ref/split.c -I../ref -I$(CUDA)/include -lcuda -lm

# the persistent core and the reloadable engine library (see core.h); the library is built to a new name and moved
# over the old one, so a running core that has the old file mapped keeps a consistent image
ENGINE_SRC = engine.c prog.c holdc.c ../ref/qsa.c ../ref/block.c ../ref/attn.c ../ref/ple.c ../ref/quant.c ../ref/gguf.c ../ref/fmath.c ../ref/hc.c ../ref/deltanet.c ../ref/router.c ../ref/split.c

mtp_q8_test: mtp_q8_test.c ../ref/quant.c ../ref/gguf.c ../ref/quant.h ../ref/gguf.h
	$(CC) -std=c11 -D_POSIX_C_SOURCE=200809L -O2 -ffp-contract=off -Wall -o $@ mtp_q8_test.c ../ref/quant.c ../ref/gguf.c -I../ref -I$(CUDA)/include -lcuda -lm
include chunk/Makefile.inc
include exactbatch/Makefile.inc
include mtp/Makefile.inc
include comm/Makefile.inc

trace_session_test: trace_session_test.c $(ENGINE_SRC) prefill.h trace_host.inc timings.h engine.h core.h prog.h hold.h dieops.h ple_read_pool.h ple_read_reload.h
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -UNDEBUG -pthread -o $@ trace_session_test.c $(filter-out engine.c,$(ENGINE_SRC)) -I../ref -I$(CUDA)/include -lcuda -lm
# the commit (and -dirty when the tree differs from it) the library is built from: the engine reports it with its `config` command
FLASH_REV ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)$(shell git diff --quiet HEAD -- .. 2>/dev/null || echo -dirty)
libengine.so: vision.h trace_host.inc $(ENGINE_SRC) prefill.h prog.h hold.h core.h engine.h dieops.h balance/layout.h ple_read_pool.h ple_read_reload.h ../ref/split.h ../ref/qsa.h ../ref/block.h ../ref/attn.h ../ref/ple.h ../ref/gguf.h ../ref/quant.h ../ref/fmath.h ../ref/hc.h ../ref/deltanet.h ../ref/router.h timings.h qsa_state.h
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -fPIC -shared -pthread -DFLASH_REV='"$(FLASH_REV)"' -o $@.new $(ENGINE_SRC) -I../ref -I$(CUDA)/include -lcuda -lm && mv -f $@.new $@

flashcore: flashcore.c flashcore_prefill.h core.h hold.h holdc.c ../ref/gguf.c ../ref/gguf.h
	$(CC) -std=gnu99 -O2 -Wall -o $@ flashcore.c holdc.c ../ref/gguf.c -I../ref -I$(CUDA)/include -lcuda -ldl -lm

flashcore_prefill_test: flashcore_prefill_test.c flashcore_prefill.h
	$(CC) -std=gnu99 -O2 -Wall -Wextra -Werror -UNDEBUG -o $@ $<

.PHONY: test-flashcore-prefill
test-flashcore-prefill: flashcore_prefill_test
	./flashcore_prefill_test

nullengine.so: nullengine.c core.h
	$(CC) -std=gnu99 -O2 -Wall -fPIC -shared -o $@.new nullengine.c -I../ref -I$(CUDA)/include && mv -f $@.new $@

tok_test: tok_test.c tok.c tok.h ucd.h ../ref/gguf.c ../ref/gguf.h
	$(CC) -std=gnu99 -O2 -Wall -o $@ tok_test.c tok.c ../ref/gguf.c -I../ref

jinja_test: jinja_test.c jinja.c jinja.h json.c json.h
	$(CC) -std=gnu99 -O2 -Wall -o $@ jinja_test.c jinja.c json.c -lm

flashd: vision.h timings.h flashd.c tok.c tok.h jinja.c jinja.h json.c json.h ucd.h ../ref/gguf.c ../ref/gguf.h
	$(CC) -std=gnu99 -O2 -Wall -o $@ flashd.c tok.c jinja.c json.c ../ref/gguf.c -I../ref -lm -lpthread

sample_test: sample_test.c dieops.h ../ref/sample.c ../ref/sample.h ../ref/fmath.c
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ sample_test.c ../ref/sample.c ../ref/fmath.c -I../ref -I$(CUDA)/include -lcuda -lm

# ---- QSA: int8 KV cache, indexer, selection, gather through the persistent kernel vs ref/qsa.c ----
qsa_test: qsa_test.c quant_range_cases.h test_guards.h qsa_padding_test.h dieops.h ../ref/qsa.c ../ref/qsa.h ../ref/attn.c ../ref/attn.h ../ref/fmath.c ../ref/quant.c ../ref/gguf.c ../ref/checked.h
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ qsa_test.c ../ref/qsa.c ../ref/attn.c ../ref/fmath.c ../ref/quant.c ../ref/gguf.c -I../ref -I$(CUDA)/include -lcuda -lm

attnpartn_unitmap: attnpartn_unitmap.c
	$(CC) -std=gnu99 -O2 -Wall -Wextra -o $@ $<

# ---- the vision encoder daemon (libmtmd from a llama.cpp build; LLAMA_CPP names its tree) ----
# The vision service requires the installed CUDA backend.
LLAMA_CPP ?= $(HOME)/claude/llama.cpp-cuda
LLAMA_BUILD ?= $(LLAMA_CPP)/build
flashvis: flashvis.c vision.h vis_ggml.c vis_ggml.h vis_fa.c vis_fa.h vis_fa.cubin vis_supervisor.c vis_supervisor.h $(LLAMA_BUILD)/bin/libggml-cuda.so
	$(CC) -std=gnu11 -O2 -Wall -Wl,--export-dynamic -o $@ flashvis.c vis_ggml.c vis_fa.c vis_supervisor.c -DVIS_FA_PTX_PATH='"$(abspath vis_fa.cubin)"' -DFLASHVIS_BACKEND_DIR='"$(abspath $(LLAMA_BUILD))/bin"' -I$(CUDA)/include -I$(LLAMA_CPP)/include -I$(LLAMA_CPP)/ggml/include -I$(LLAMA_CPP)/tools/mtmd -L$(LLAMA_BUILD)/bin -lmtmd -lllama -lggml -lggml-base -lcuda -ldl -lm -Wl,-rpath,$(abspath $(LLAMA_BUILD))/bin

vis_fa.cubin: vis_fa.ptx
	$(PTXAS11) -arch=sm_52 -o $@ $<

memtest: memtest.c
	$(CC) -std=gnu99 -O2 -Wall -o $@ memtest.c -I$(CUDA)/include -lcuda

fence: fence.c
	$(CC) -std=gnu99 -O2 -Wall -o $@ fence.c -I$(CUDA)/include -lcuda

# Same packing implementation as the holder; no CUDA driver link or calls.
hold_cache: hold.c hold_selection.h prog.c prog.h dieops.h balance/layout.h balance/embedding.h balance/program.h ../ref/quant.c ../ref/gguf.c ../ref/quant.h ../ref/gguf.h ../ref/split.c ../ref/split.h
	$(CC) -std=gnu99 -O2 -Wall -DHOLD_CACHE_ONLY -o $@ hold.c prog.c ../ref/quant.c ../ref/gguf.c ../ref/split.c -I../ref -I$(CUDA)/include -lm

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

pair_state_test: pair_state_test.c prog.c prog.h dieops.h ../ref/quant.c ../ref/gguf.c ../ref/split.c
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ pair_state_test.c prog.c ../ref/quant.c ../ref/gguf.c ../ref/split.c -I../ref -I$(CUDA)/include -lcuda -lm

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

.PHONY: test-flashd-deadline
test-flashd-deadline: flashd flashd_timing_test
	./flashd_timing_test --unit
	perl flashd_deadline_test.pl ./flashd
# This planner test needs only a C compiler.
prefill_plan_test: prefill_plan_test.c prefill.h
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror -o $@ prefill_plan_test.c

attnpartn_regress: attnpartn_regress.c cta_trace.h dieops.h ../ref/attn.c ../ref/fmath.c ../ref/attn.h ../ref/fmath.h
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ attnpartn_regress.c ../ref/attn.c ../ref/fmath.c -I../ref -I$(CUDA)/include -lcuda -lm

# Validation-only engine: uses the production engine and saves paired decode
# callbacks plus per-CTA traces. Load only in a validator-owned flashcore.
libengine_cta_remap.so: cta_remap_bench.c cta_trace.h $(ENGINE_SRC) core.h engine.h prefill.h dieops.h
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -fPIC -shared -DFLASH_REV='"$(FLASH_REV)"' -o $@.new cta_remap_bench.c $(filter-out engine.c,$(ENGINE_SRC)) -I../ref -I$(CUDA)/include -lcuda -lm && mv -f $@.new $@

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

# Offline checks for the six-die balance experiment; no CUDA or model required.
.PHONY: die_balance_test
die_balance_test:
	ruby ../validation/issue54/test_die_balance.rb
	ruby ../validation/issue54/test_die_balance_capture.rb
