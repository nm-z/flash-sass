ISSUE110_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
include $(ISSUE110_DIR)/../issue9/Makefile
.DEFAULT_GOAL := phase-check
PHASE_OUT := $(ISSUE110_DIR)/bin
PHASE_MAXAS := env PATH=/opt/cuda/bin:/usr/bin:/bin perl -I$(MAXAS_ROOT)/lib $(MAXAS_ROOT)/bin/maxas.pl
XCHG_FENCE_SASS := $(PHASE_OUT)/die-xchg-sys-fence.sass
XCHG_FENCE_CUBIN := $(PHASE_OUT)/die-xchg-sys-fence.cubin
XCHG_PEER_FENCE_SASS := $(PHASE_OUT)/die-xchg-peer-sys-fence.sass
XCHG_PEER_FENCE_CUBIN := $(PHASE_OUT)/die-xchg-peer-sys-fence.cubin
HOLDER_PIDS ?= $(shell ps -C hold -o pid=,args= | gawk -v gguf='$(GGUF)' -f $(ISSUE110_DIR)/holders.awk)

ifneq ($(GPUQ_JOB_ID),)
ifneq ($(filter phase-check phase-fixture-build phase-engine-build phase-build,$(MAKECMDGOALS)),)
$(error Run phase probe builds/checks through dsh build outside gpuq)
endif
endif

.PHONY: phase-check phase-fixture-build phase-engine-build phase-build phase-gpu phase-canary phase-matrix phase-failure-gpu xchg-fence-build xchg-peer-fence-build
$(PHASE_OUT):
	mkdir -p $@

$(PHASE_OUT)/phase_probe_test: $(ROOT)/kern/phase_probe_test.c $(ROOT)/kern/phase_probe.c $(ROOT)/kern/phase_probe.h $(ISSUE110_DIR)/dieops.h | $(PHASE_OUT)
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror -I$(ROOT)/kern -I$(ISSUE110_DIR) -I/opt/cuda/include -o $@ $(ROOT)/kern/phase_probe_test.c $(ROOT)/kern/phase_probe.c -lcuda

phase-check: $(PHASE_OUT)/phase_probe_test
	$(PHASE_OUT)/phase_probe_test
	sha256sum $(PHASE_OUT)/phase_probe_test $(ROOT)/kern/phase_probe.c $(ROOT)/kern/phase_probe.h

phase-engine-build:
	$(MAKE) -B -C $(ROOT)/kern FLASH_REV=$(ENGINE_REVISION) libengine.so flashcore

$(PHASE_OUT)/phase_probe_gpu: $(ISSUE110_DIR)/phase_probe_gpu.c $(ROOT)/kern/phase_probe.c $(ROOT)/kern/phase_probe.h $(ROOT)/kern/prog.c $(ISSUE110_DIR)/dieops.h | $(PHASE_OUT)
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -Wextra -Werror -I$(ROOT)/kern -I$(ISSUE110_DIR) -I$(ROOT)/ref -I/opt/cuda/include -o $@ $< $(ROOT)/kern/phase_probe.c $(ROOT)/kern/prog.c $(ROOT)/ref/quant.c $(ROOT)/ref/gguf.c $(ROOT)/ref/split.c -lcuda -lm

phase-fixture-build: $(PHASE_OUT)/phase_probe_gpu

phase-build: kernel-build $(PHASE_OUT)/phase_probe_gpu | $(PHASE_OUT)
	env --chdir=$(ROOT) sha256sum -c validation/issue110/baseline.sha256
	cp $(ROOT)/kern/die.cubin $(PHASE_OUT)/baseline.cubin
	gawk -f $(ISSUE110_DIR)/phase_probe.awk $(ROOT)/kern/die.sass > $(PHASE_OUT)/die-phase-probe.sass
	cp $(ROOT)/kern/skel_die.cubin $(ROOT)/kern/die.cubin
	$(PHASE_MAXAS) -i $(PHASE_OUT)/die-phase-probe.sass $(ROOT)/kern/die.cubin
	env --chdir=$(ROOT) sha256sum kern/die.cubin kern/die.sass kern/dieops.h kern/phase_probe.c kern/phase_probe.h validation/issue110/phase_probe.awk validation/issue110/bin/die-phase-probe.sass $(patsubst $(ROOT)/%,%,$(KERNEL_INPUTS)) $(MAXAS_ROOT)/lib/MaxAs/MaxAsGrammar.pm $(MAXAS_ROOT)/lib/MaxAs/MaxAs.pm $(MAXAS_ROOT)/lib/MaxAs/Cubin.pm > $(KERNEL_RECEIPT)
	$(MAKE) -f $(ISSUE110_DIR)/../issue9/Makefile ENGINE_REVISION=$(ENGINE_REVISION) engine-build engine-check
	$(MAKE) -C $(ROOT)/kern model_test
	sha256sum $(ROOT)/kern/die.cubin $(ROOT)/kern/libengine.so $(ROOT)/kern/flashcore $(OUT_DIR)/engine_check $(PHASE_OUT)/phase_probe_gpu

# Diagnostic cubin only: add a system-scope fence from each xsend writer thread
# before its CTA rendezvous and the framework's peer completion flag publication.
xchg-fence-build: kernel-build | $(PHASE_OUT)
	gawk -f $(ISSUE110_DIR)/xchg_sys_fence.awk $(ROOT)/kern/die.sass > $(XCHG_FENCE_SASS)
	cp $(ROOT)/kern/skel_die.cubin $(XCHG_FENCE_CUBIN)
	$(PHASE_MAXAS) -i $(XCHG_FENCE_SASS) $(XCHG_FENCE_CUBIN)
	sha256sum $(XCHG_FENCE_SASS) $(XCHG_FENCE_CUBIN)

# Diagnostic cubin: fence every operation that writes peer inbox memory before
# its CTA barrier; the framework publishes the send flag after that barrier.
xchg-peer-fence-build: kernel-build | $(PHASE_OUT)
	gawk -f $(ISSUE110_DIR)/xchg_peer_fence.awk $(ROOT)/kern/die.sass > $(XCHG_PEER_FENCE_SASS)
	cp $(ROOT)/kern/skel_die.cubin $(XCHG_PEER_FENCE_CUBIN)
	$(PHASE_MAXAS) -i $(XCHG_PEER_FENCE_SASS) $(XCHG_PEER_FENCE_CUBIN)
	sha256sum $(XCHG_PEER_FENCE_SASS) $(XCHG_PEER_FENCE_CUBIN)

phase-gpu: identity
	env --chdir=$(ROOT)/kern CUDA_VISIBLE_DEVICES=6 $(PHASE_OUT)/phase_probe_gpu $(ROOT)/kern/die.cubin

# Deliberate invalid read on otherwise unused GPU6; each expected exit occurs
# in a fresh child. The parent never initializes CUDA. Not a model fault repro.
phase-failure-gpu: identity
	env --chdir=$(ROOT)/kern CUDA_VISIBLE_DEVICES=6 $(PHASE_OUT)/phase_probe_gpu --failure-paths $(ROOT)/kern/die.cubin

# The framework hook touches dispatch scratch/predicate state for every op.
# Check one complete short scalar-oracle case before the long diagnostic.
phase-canary: phase-gpu
	$(MAKE) -f $(ISSUE110_DIR)/../issue9/Makefile TOKENS=9 BLOCKS=1 HOST_KV=1 FLAGS=eh ORACLE_REVISION=e2ff58ff1b0e956d61a1c93830c5f66210d27788 ORACLE_DATA=/mnt/stop-hook-hdd/stop-hook/flash-sass-issue9-oracle/par-e2ff-b1-t9 model-oracle

phase-matrix: identity
	$(if $(filter 6,$(words $(HOLDER_PIDS))),,$(error Supply the six verified live holder PIDs in logical-die order))
	$(OUT_DIR)/engine_check --phase-probe --repo $(ROOT) --out $(GPUQ_RUN_DIR)/engine-matrix --ple $(PLE_GGUF) --revision $(ENGINE_REVISION) $(foreach path,$(MANIFESTS),--manifest $(path)) $(foreach pid,$(HOLDER_PIDS),--holder-pid $(pid)) --cuda-visible 0,1,2,3,4,5 --split-shares $(SHARES) --attn-map $(ATTENTION_MAP) --tokens $(TOKENS) --snapshots $(SNAPSHOTS) --nblk $(BLOCKS) --lock-fd $(GPUQ_LOCK_FD) --timeout $(ENGINE_TIMEOUT) --weight-manifest $(CACHE_ROOT)/checkpoints.sha256
