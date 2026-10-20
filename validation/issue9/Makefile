SOURCE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ROOT := $(abspath $(SOURCE_DIR)/../..)
OUT_DIR ?= $(SOURCE_DIR)/bin
CC ?= cc
MAXAS_ROOT ?= /home/nate/work/fs9-maxas
KERNEL_INPUTS := $(filter-out $(ROOT)/kern/die.sass $(ROOT)/kern/die_trace%.sass,$(wildcard $(ROOT)/kern/*.sass)) $(ROOT)/kern/die.pl $(ROOT)/kern/trace.pl $(ROOT)/kern/actq.pl $(ROOT)/kern/fmath.pl $(ROOT)/kern/die.ptx $(wildcard $(ROOT)/maxas-patch/*.pm)
KERNEL_RECEIPT ?= $(ROOT)/kern/issue9-kernel.sha256
CACHE_ROOT ?= /mnt/stop-hook-hdd/stop-hook/flash-sass-issue9-cache-v3
MODEL_CACHE ?= $(CACHE_ROOT)/tensors
GGUF ?= /mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
PLE_GGUF ?= /mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf
TOKENS ?= 1
BLOCKS ?= 512
HOST_KV ?= 1
FLAGS ?= eh
CAPTURE ?= 0
SHARES ?= 112,97,81,116,97,97
ATTENTION_MAP ?= 1,1,1,1,4,4,4,4,5,5,5,5
MANIFESTS := /tmp/hold-h0.manifest /tmp/hold-h1.manifest /tmp/hold-h2.manifest /tmp/hold-h3.manifest /tmp/hold-h4.manifest /tmp/hold-h5.manifest
ORACLE_SOURCES := $(ROOT)/kern/model_oracle.c $(addprefix $(ROOT)/ref/,qsa.c block.c attn.c ple.c quant.c gguf.c fmath.c hc.c deltanet.c router.c split.c)
ORACLE_HEADERS := $(ROOT)/kern/model_oracle.h $(wildcard $(ROOT)/ref/*.h)
ORACLE_CFLAGS := -std=c11 -D_XOPEN_SOURCE=700 -O2 -ffp-contract=off -Wall -Wextra -Werror
ORACLE_ARCH_FLAGS ?=
ORACLE_SOURCE_FILES := kern/model_oracle.c kern/model_oracle.h kern/model_oracle_gpu.h kern/model_test.c kern/model_qsa_test.h kern/qsa_state.h kern/prog.c kern/prog.h kern/holdc.c kern/hold.h kern/dieops.h $(patsubst $(ROOT)/%,%,$(wildcard $(ROOT)/ref/*.c $(ROOT)/ref/*.h))
ORACLE_SOURCE_MANIFEST ?= $(OUT_DIR)/oracle-source.sha256
ORACLE_REVISION ?= $(shell git -C $(ROOT) rev-parse HEAD)
ENGINE_REVISION ?= $(ORACLE_REVISION)
ORACLE_DATA ?= /mnt/stop-hook-hdd/stop-hook/flash-sass-issue9-oracle/b$(BLOCKS)-t$(TOKENS)
ORACLE_IDS ?= $(ORACLE_DATA)/ids.txt
ORACLE_CHUNK ?= 32
ORACLE_DIES ?= 0,1,2,3,4,5
CACHE_SECOND_ROOT ?= /home/nate/work/fs9-sol-cache-tail
SNAPSHOTS ?= 1,2,3,5
ENGINE_TIMEOUT ?= 3600
.NOTPARALLEL:
ifneq ($(GPUQ_JOB_ID),)
ifeq ($(strip $(MAKECMDGOALS)),)
$(error Default target builds CPU code; invoke a GPU execution target explicitly)
endif
ifneq ($(filter all cpu-check engine-build engine-check oracle-header oracle-build oracle-check oracle-chunk oracle-batch oracle-validate oracle-parity cache-export cache-export-two-volumes cache-identity kernel-build oracle-step-%,$(MAKECMDGOALS)),)
$(error Run CPU builds/reference work through dsh build outside gpuq)
endif
endif

.PHONY: all cpu-check engine-build engine-check engine-matrix oracle-header oracle-build oracle-check oracle-source-identity oracle-chunk oracle-batch oracle-validate oracle-parity model-oracle assembler-check kernel-build cache-export cache-export-two-volumes cache-identity identity model direct-identity direct-matrix
all:
	$(MAKE) -C $(ROOT)/kern FLASH_REV=$(ENGINE_REVISION) model_test hold_cache libengine.so flashcore

assembler-check:
	cmp $(ROOT)/maxas-patch/MaxAsGrammar.pm $(MAXAS_ROOT)/lib/MaxAs/MaxAsGrammar.pm
	cmp $(ROOT)/maxas-patch/MaxAs.pm $(MAXAS_ROOT)/lib/MaxAs/MaxAs.pm
	cmp $(ROOT)/maxas-patch/Cubin.pm $(MAXAS_ROOT)/lib/MaxAs/Cubin.pm

kernel-build: assembler-check
	$(MAKE) -B -C $(ROOT)/kern 'MAXAS=env PATH=/opt/cuda/bin:/usr/bin:/bin perl -I$(MAXAS_ROOT)/lib $(MAXAS_ROOT)/bin/maxas.pl' die.cubin
	env --chdir=$(ROOT) sha256sum kern/die.cubin $(patsubst $(ROOT)/%,%,$(KERNEL_INPUTS)) $(MAXAS_ROOT)/lib/MaxAs/MaxAsGrammar.pm $(MAXAS_ROOT)/lib/MaxAs/MaxAs.pm $(MAXAS_ROOT)/lib/MaxAs/Cubin.pm > $(KERNEL_RECEIPT)

$(OUT_DIR):
	mkdir -p $@

$(OUT_DIR)/model_cli_test: $(SOURCE_DIR)/model_cli_test.c | $(OUT_DIR)
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror -o $@ $<

$(OUT_DIR)/cuinit_probe.so: $(SOURCE_DIR)/cuinit_probe.c | $(OUT_DIR)
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror -fPIC -shared -o $@ $<

$(OUT_DIR)/engine_check: $(SOURCE_DIR)/engine_check.c | $(OUT_DIR)
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror -o $@ $< -lcrypto

engine-build: $(OUT_DIR)/engine_check
	$(MAKE) -B -C $(ROOT)/kern FLASH_REV=$(ENGINE_REVISION) libengine.so flashcore

engine-check: $(OUT_DIR)/engine_check
	$(OUT_DIR)/engine_check --self-test
	sha256sum $(OUT_DIR)/engine_check

$(OUT_DIR)/model_oracle: $(ORACLE_SOURCES) $(ORACLE_HEADERS) | $(OUT_DIR)
	$(CC) $(ORACLE_CFLAGS) $(ORACLE_ARCH_FLAGS) -DMODEL_ORACLE_STANDALONE -o $@ $(ORACLE_SOURCES) -lcrypto -lm

$(OUT_DIR)/oracle_test: $(SOURCE_DIR)/oracle_test.c $(ORACLE_SOURCES) $(ORACLE_HEADERS) | $(OUT_DIR)
	$(CC) $(ORACLE_CFLAGS) $(ORACLE_ARCH_FLAGS) -o $@ $< $(ORACLE_SOURCES) -lcrypto -lm

oracle-header:
	$(MAKE) -B -C $(ROOT)/kern dieops.h

oracle-build: oracle-header $(OUT_DIR)/model_oracle $(OUT_DIR)/oracle_test

oracle-check: oracle-build
	$(OUT_DIR)/oracle_test $(OUT_DIR)/model_oracle
	sha256sum $(OUT_DIR)/model_oracle $(OUT_DIR)/oracle_test

oracle-source-identity: | $(OUT_DIR)
	env --chdir=$(ROOT) sha256sum $(ORACLE_SOURCE_FILES) > $(ORACLE_SOURCE_MANIFEST)
	sha256sum $(ORACLE_SOURCE_MANIFEST)

$(ORACLE_IDS):
	mkdir -p $(ORACLE_DATA)
	gawk -v count=$(TOKENS) 'BEGIN { for (i = 0; i < count; ++i) print (10 + i) % 248320 }' > $@

# CPU execution uses persistent storage outside GPUQ; every invocation commits
# at most 32 positions, and resumes with the same content/geometry seal.
oracle-chunk: oracle-build oracle-source-identity $(ORACLE_IDS)
	ldd $(OUT_DIR)/model_oracle
	sha256sum $(OUT_DIR)/model_oracle $(ORACLE_IDS)
	env SPLIT_SHARES=$(SHARES) ATTN_MAP=$(ATTENTION_MAP) $(OUT_DIR)/model_oracle $(ROOT) $(ORACLE_SOURCE_MANIFEST) $(ORACLE_REVISION) $(MODEL_CACHE) $(CACHE_ROOT)/manifest.new $(CACHE_ROOT)/tensors.sha256 $(CACHE_ROOT)/checkpoints.sha256 $(GGUF) $(PLE_GGUF) 192 $(ORACLE_IDS) $(BLOCKS) $(ORACLE_DIES) $(ORACLE_DATA)/records.bin $(ORACLE_DATA)/checkpoint.bin $(ORACLE_DATA)/receipt.bin $(ORACLE_CHUNK)

# Eight serialized, individually checkpointed chunks per routed CPU job.
# The deep reference can resume this bounded batch after an interruption.
ORACLE_BATCH_STEPS := 1 2 3 4 5 6 7 8
define ORACLE_BATCH_STEP
.PHONY: oracle-step-$(1)
oracle-step-$(1):
	$$(MAKE) -f $$(SOURCE_DIR)/Makefile oracle-chunk
endef
$(foreach step,$(ORACLE_BATCH_STEPS),$(eval $(call ORACLE_BATCH_STEP,$(step))))
oracle-batch: $(addprefix oracle-step-,$(ORACLE_BATCH_STEPS))

# Parse the complete produced stream on CPU before submitting GPU work.
oracle-validate: oracle-build oracle-source-identity
	$(OUT_DIR)/model_oracle $(ROOT) $(ORACLE_SOURCE_MANIFEST) $(ORACLE_REVISION) $(MODEL_CACHE) $(CACHE_ROOT)/manifest.new $(CACHE_ROOT)/tensors.sha256 $(CACHE_ROOT)/checkpoints.sha256 $(GGUF) $(PLE_GGUF) 192 $(ORACLE_IDS) $(BLOCKS) $(ORACLE_DIES) $(ORACLE_DATA)/records.bin $(ORACLE_DATA)/checkpoint.bin $(ORACLE_DATA)/receipt.bin verify

# Compare complete scalar outputs after independently running both producers.
oracle-parity:
	$(if $(ORACLE_OTHER_DATA),,$(error Supply ORACLE_OTHER_DATA for the independent reference))
	cmp $(ORACLE_DATA)/records.bin $(ORACLE_OTHER_DATA)/records.bin
	cmp $(ORACLE_DATA)/checkpoint.bin $(ORACLE_OTHER_DATA)/checkpoint.bin
	sha256sum $(ORACLE_DATA)/records.bin $(ORACLE_OTHER_DATA)/records.bin
	sha256sum $(ORACLE_DATA)/checkpoint.bin $(ORACLE_OTHER_DATA)/checkpoint.bin

cpu-check: $(OUT_DIR)/model_cli_test $(OUT_DIR)/cuinit_probe.so
	$(MAKE) -C $(ROOT)/kern model_test
	$(OUT_DIR)/model_cli_test $(ROOT)/kern/model_test $(OUT_DIR)/cuinit_probe.so
	sha256sum $(ROOT)/kern/model_test $(OUT_DIR)/model_cli_test $(OUT_DIR)/cuinit_probe.so

# CPU-only export. Run through dsh build; the destination must be new.
cache-export:
	$(MAKE) -B -C $(ROOT)/kern hold_cache
	mkdir -p $(MODEL_CACHE)
	$(ROOT)/kern/hold_cache $(GGUF) 0 0 48 $(CACHE_ROOT)/manifest token_embd output cache=$(MODEL_CACHE)

# File-based CPU tiles can span two owned filesystems. Each exporter still
# requires a fresh destination; no existing holder/cache files are overwritten.
cache-export-two-volumes:
	$(MAKE) -B -C $(ROOT)/kern hold_cache
	mkdir -p $(MODEL_CACHE) $(CACHE_SECOND_ROOT)/tensors
	$(ROOT)/kern/hold_cache $(GGUF) 0 0 36 $(CACHE_ROOT)/part0 token_embd output cache=$(MODEL_CACHE)
	$(ROOT)/kern/hold_cache $(GGUF) 0 36 48 $(CACHE_SECOND_ROOT)/part1 cache=$(CACHE_SECOND_ROOT)/tensors
	ln -s -t $(MODEL_CACHE) $(CACHE_SECOND_ROOT)/tensors/*
	gawk 'FNR == 1 { if (NR == 1) print; next } { print }' $(CACHE_ROOT)/part0.new $(CACHE_SECOND_ROOT)/part1.new > $(CACHE_ROOT)/manifest.new

cache-identity:
	sha256sum $(MODEL_CACHE)/* > $(CACHE_ROOT)/tensors.sha256
	sha256sum $(GGUF) $(PLE_GGUF) > $(CACHE_ROOT)/checkpoints.sha256
	sha256sum $(CACHE_ROOT)/manifest.new $(CACHE_ROOT)/tensors.sha256

# These targets compile nothing. Invoke model only through Archy's GPU queue.
identity: assembler-check
	$(if $(GPUQ_JOB_ID),,$(error Invoke model through gpuq))
	env --chdir=$(ROOT) sha256sum -c $(KERNEL_RECEIPT)
	git -C $(ROOT) rev-parse HEAD
	git -C $(ROOT) status --short
	cc --version
	ldd $(ROOT)/kern/model_test
	sha256sum /usr/lib/libcuda.so.470.256.02
	sha256sum $(ROOT)/kern/model_test $(ROOT)/kern/die.cubin $(ROOT)/kern/libengine.so
	sha256sum $(CACHE_ROOT)/manifest.new $(CACHE_ROOT)/tensors.sha256 $(MANIFESTS)
	cat $(CACHE_ROOT)/checkpoints.sha256
	nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.free --format=csv

model: identity
	gawk -v count=$(TOKENS) 'BEGIN { for (i = 0; i < count; ++i) print (10 + i) % 248320 }' > $(GPUQ_RUN_DIR)/ids.txt
	sha256sum $(GPUQ_RUN_DIR)/ids.txt
	env --chdir=$(ROOT)/kern -u MODEL_NOTWIN -u MODEL_ORACLE_RECORDS -u MODEL_NPH -u MODEL_INJECT -u MODEL_DUMP_S -u MODEL_DUMP_DIR -u MODEL_DUMP_TWIN MODEL_ATTN=qsa MODEL_IDS=$(GPUQ_RUN_DIR)/ids.txt MODEL_PLE=$(PLE_GGUF) MODEL_PLE_OFF=192 MODEL_CAPTURE_QSA=$(CAPTURE) FLASH_NBLK=$(BLOCKS) FLASH_KV_HOST=$(HOST_KV) SPLIT_SHARES=$(SHARES) ATTN_MAP=$(ATTENTION_MAP) CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 $(if $(INJECT),MODEL_INJECT=$(INJECT)) ./model_test $(MODEL_CACHE) 0 48 $(TOKENS) $(FLAGS) $(MANIFESTS)

model-oracle: identity oracle-source-identity
	sha256sum $(ORACLE_IDS) $(ORACLE_DATA)/records.bin $(ORACLE_DATA)/receipt.bin
	env --chdir=$(ROOT)/kern -u MODEL_NOTWIN -u MODEL_NPH -u MODEL_INJECT -u MODEL_DUMP_S -u MODEL_DUMP_DIR -u MODEL_DUMP_TWIN MODEL_ATTN=qsa MODEL_IDS=$(ORACLE_IDS) MODEL_PLE=$(PLE_GGUF) MODEL_PLE_OFF=192 MODEL_CAPTURE_QSA=$(CAPTURE) FLASH_NBLK=$(BLOCKS) FLASH_KV_HOST=$(HOST_KV) SPLIT_SHARES=$(SHARES) ATTN_MAP=$(ATTENTION_MAP) CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 MODEL_ORACLE_RECORDS=$(ORACLE_DATA)/records.bin MODEL_ORACLE_RECEIPT=$(ORACLE_DATA)/receipt.bin MODEL_ORACLE_SOURCE_ROOT=$(ROOT) MODEL_ORACLE_SOURCE_MANIFEST=$(ORACLE_SOURCE_MANIFEST) MODEL_ORACLE_SOURCE_REVISION=$(ORACLE_REVISION) MODEL_ORACLE_TILE_MANIFEST=$(CACHE_ROOT)/manifest.new MODEL_ORACLE_TILE_HASHES=$(CACHE_ROOT)/tensors.sha256 MODEL_ORACLE_CHECKPOINT_HASHES=$(CACHE_ROOT)/checkpoints.sha256 MODEL_ORACLE_PRIMARY=$(GGUF) $(if $(INJECT),MODEL_INJECT=$(INJECT)) ./model_test $(MODEL_CACHE) 0 48 $(TOKENS) $(FLAGS) $(MANIFESTS)

engine-matrix: identity
	$(if $(filter 6,$(words $(HOLDER_PIDS))),,$(error Supply the six verified live holder PIDs in logical-die order))
	$(OUT_DIR)/engine_check --repo $(ROOT) --out $(GPUQ_RUN_DIR)/engine-matrix --ple $(PLE_GGUF) --revision $(ENGINE_REVISION) $(foreach path,$(MANIFESTS),--manifest $(path)) $(foreach pid,$(HOLDER_PIDS),--holder-pid $(pid)) --cuda-visible 0,1,2,3,4,5 --split-shares $(SHARES) --attn-map $(ATTENTION_MAP) --tokens $(TOKENS) --snapshots $(SNAPSHOTS) --nblk $(BLOCKS) --lock-fd $(GPUQ_LOCK_FD) --timeout $(ENGINE_TIMEOUT) --weight-manifest $(CACHE_ROOT)/checkpoints.sha256

direct-identity: assembler-check
	$(if $(GPUQ_JOB_ID),,$(error Invoke direct-matrix through gpuq))
	env --chdir=$(ROOT) sha256sum -c $(KERNEL_RECEIPT)
	git -C $(ROOT) rev-parse HEAD
	git -C $(ROOT) status --short
	sha256sum $(ROOT)/kern/qsa_direct_test $(ROOT)/kern/die.cubin /usr/lib/libcuda.so.470.256.02
	sha256sum $(MAXAS_ROOT)/lib/MaxAs/MaxAsGrammar.pm $(MAXAS_ROOT)/lib/MaxAs/MaxAs.pm $(MAXAS_ROOT)/lib/MaxAs/Cubin.pm
	sha256sum $(ROOT)/kern/qsa_direct_test.c $(ROOT)/kern/qsa_test.c $(ROOT)/kern/prog.c $(ROOT)/kern/prog.h $(ROOT)/kern/die.pl $(ROOT)/kern/trace.pl $(ROOT)/kern/*.sass $(ROOT)/ref/qsa.c $(ROOT)/ref/attn.c
	nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.free --format=csv

# Four consecutive positions cover every modulo-four tail. The 2048 case
# includes 2051, and 2052 starts at production-budget selection exclusion.
DIRECT_LENGTHS := 1 2048 2052 3100 8000 65000 262000
define DIRECT_CASE
.PHONY: direct-$(1)-$(2)
direct-$(1)-$(2): direct-identity
	env --chdir=$(ROOT)/kern -u QSA_DIRECT_INJECT -u QSA_REPLAY_PREFIX -u QSA_PROJECTED CUDA_VISIBLE_DEVICES=6 QSA_HOSTKV=$(1) QSA_DIRECT_SEED=4242 QSA_PATTERN=hash ./qsa_direct_test $(2) 512 4
endef
$(foreach placement,0 1,$(foreach length,$(DIRECT_LENGTHS),$(eval $(call DIRECT_CASE,$(placement),$(length)))))
direct-matrix: $(foreach placement,0 1,$(foreach length,$(DIRECT_LENGTHS),direct-$(placement)-$(length)))
