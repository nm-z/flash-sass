ISSUE110_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
include $(ISSUE110_DIR)/../issue9/Makefile
.DEFAULT_GOAL := phase-check
PHASE_OUT := $(ISSUE110_DIR)/bin
PHASE_MAXAS := env PATH=/opt/cuda/bin:/usr/bin:/bin perl -I$(MAXAS_ROOT)/lib $(MAXAS_ROOT)/bin/maxas.pl
HOLDER_PIDS ?= $(shell ps -C hold -o pid=,args= | gawk -v gguf='$(GGUF)' -f $(ISSUE110_DIR)/holders.awk)

ifneq ($(GPUQ_JOB_ID),)
ifneq ($(filter phase-check phase-engine-build phase-build,$(MAKECMDGOALS)),)
$(error Run phase probe builds/checks through dsh build outside gpuq)
endif
endif

.PHONY: phase-check phase-engine-build phase-build phase-gpu phase-canary phase-matrix
$(PHASE_OUT):
	mkdir -p $@

$(PHASE_OUT)/phase_probe_test: $(ROOT)/kern/phase_probe_test.c $(ROOT)/kern/phase_probe.c $(ROOT)/kern/phase_probe.h | $(PHASE_OUT) oracle-header
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror -I$(ROOT)/kern -I/opt/cuda/include -o $@ $(ROOT)/kern/phase_probe_test.c $(ROOT)/kern/phase_probe.c -lcuda

phase-check: $(PHASE_OUT)/phase_probe_test
	$(PHASE_OUT)/phase_probe_test
	sha256sum $(PHASE_OUT)/phase_probe_test $(ROOT)/kern/phase_probe.c $(ROOT)/kern/phase_probe.h

phase-engine-build:
	$(MAKE) -B -C $(ROOT)/kern FLASH_REV=$(ENGINE_REVISION) libengine.so flashcore

$(PHASE_OUT)/phase_probe_gpu: $(ISSUE110_DIR)/phase_probe_gpu.c $(ROOT)/kern/phase_probe.c $(ROOT)/kern/phase_probe.h $(ROOT)/kern/prog.c | $(PHASE_OUT) oracle-header
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -Wextra -Werror -I$(ROOT)/kern -I$(ROOT)/ref -I/opt/cuda/include -o $@ $< $(ROOT)/kern/phase_probe.c $(ROOT)/kern/prog.c $(ROOT)/ref/quant.c $(ROOT)/ref/gguf.c $(ROOT)/ref/split.c -lcuda -lm

phase-build: kernel-build $(PHASE_OUT)/phase_probe_gpu | $(PHASE_OUT)
	env --chdir=$(ROOT) sha256sum -c validation/issue110/baseline.sha256
	cp $(ROOT)/kern/die.cubin $(PHASE_OUT)/baseline.cubin
	gawk -f $(ISSUE110_DIR)/phase_probe.awk $(ROOT)/kern/die.sass > $(PHASE_OUT)/die-phase-probe.sass
	cp $(ROOT)/kern/skel_die.cubin $(ROOT)/kern/die.cubin
	$(PHASE_MAXAS) -i $(PHASE_OUT)/die-phase-probe.sass $(ROOT)/kern/die.cubin
	env --chdir=$(ROOT) sha256sum kern/die.cubin kern/die.sass kern/phase_probe.c kern/phase_probe.h validation/issue110/phase_probe.awk validation/issue110/bin/die-phase-probe.sass $(patsubst $(ROOT)/%,%,$(KERNEL_INPUTS)) $(MAXAS_ROOT)/lib/MaxAs/MaxAsGrammar.pm $(MAXAS_ROOT)/lib/MaxAs/MaxAs.pm $(MAXAS_ROOT)/lib/MaxAs/Cubin.pm > $(KERNEL_RECEIPT)
	$(MAKE) -f $(ISSUE110_DIR)/../issue9/Makefile ENGINE_REVISION=$(ENGINE_REVISION) engine-build engine-check
	$(MAKE) -C $(ROOT)/kern model_test
	sha256sum $(ROOT)/kern/die.cubin $(ROOT)/kern/libengine.so $(ROOT)/kern/flashcore $(OUT_DIR)/engine_check $(PHASE_OUT)/phase_probe_gpu

phase-gpu: identity
	env --chdir=$(ROOT)/kern CUDA_VISIBLE_DEVICES=6 $(PHASE_OUT)/phase_probe_gpu $(ROOT)/kern/die.cubin

# The framework hook touches dispatch scratch/predicate state for every op.
# Check one complete short scalar-oracle case before the long diagnostic.
phase-canary: phase-gpu
	$(MAKE) -f $(ISSUE110_DIR)/../issue9/Makefile TOKENS=9 BLOCKS=1 HOST_KV=1 FLAGS=eh ORACLE_REVISION=e2ff58ff1b0e956d61a1c93830c5f66210d27788 ORACLE_DATA=/mnt/stop-hook-hdd/stop-hook/flash-sass-issue9-oracle/par-e2ff-b1-t9 model-oracle

phase-matrix: identity
	$(if $(filter 6,$(words $(HOLDER_PIDS))),,$(error Supply the six verified live holder PIDs in logical-die order))
	$(OUT_DIR)/engine_check --phase-probe --repo $(ROOT) --out $(GPUQ_RUN_DIR)/engine-matrix --ple $(PLE_GGUF) --revision $(ENGINE_REVISION) $(foreach path,$(MANIFESTS),--manifest $(path)) $(foreach pid,$(HOLDER_PIDS),--holder-pid $(pid)) --cuda-visible 0,1,2,3,4,5 --split-shares $(SHARES) --attn-map $(ATTENTION_MAP) --tokens $(TOKENS) --snapshots $(SNAPSHOTS) --nblk $(BLOCKS) --lock-fd $(GPUQ_LOCK_FD) --timeout $(ENGINE_TIMEOUT) --weight-manifest $(CACHE_ROOT)/checkpoints.sha256
