SOURCE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ROOT := $(abspath $(SOURCE_DIR)/../..)
OUT_DIR ?= $(SOURCE_DIR)/bin
CC ?= cc
CACHE_ROOT ?= /mnt/sentry-nfs/flash-next-q2/issue9-sol-cache
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
.NOTPARALLEL:

.PHONY: all cpu-check cache-export cache-identity identity model direct-identity direct-matrix
all:
	$(MAKE) -C $(ROOT)/kern model_test hold_cache libengine.so flashcore

$(OUT_DIR):
	mkdir -p $@

$(OUT_DIR)/model_cli_test: $(SOURCE_DIR)/model_cli_test.c | $(OUT_DIR)
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror -o $@ $<

$(OUT_DIR)/cuinit_probe.so: $(SOURCE_DIR)/cuinit_probe.c | $(OUT_DIR)
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror -fPIC -shared -o $@ $<

cpu-check: $(OUT_DIR)/model_cli_test $(OUT_DIR)/cuinit_probe.so
	$(MAKE) -C $(ROOT)/kern model_test
	$(OUT_DIR)/model_cli_test $(ROOT)/kern/model_test $(OUT_DIR)/cuinit_probe.so
	sha256sum $(ROOT)/kern/model_test $(OUT_DIR)/model_cli_test $(OUT_DIR)/cuinit_probe.so

# CPU-only export. Run through dsh build; the destination must be new.
cache-export:
	$(MAKE) -B -C $(ROOT)/kern hold_cache
	mkdir -p $(MODEL_CACHE)
	$(ROOT)/kern/hold_cache $(GGUF) 0 0 48 $(CACHE_ROOT)/manifest token_embd output cache=$(MODEL_CACHE)

cache-identity:
	sha256sum $(MODEL_CACHE)/* > $(CACHE_ROOT)/tensors.sha256
	sha256sum $(GGUF) $(PLE_GGUF) > $(CACHE_ROOT)/checkpoints.sha256
	sha256sum $(CACHE_ROOT)/manifest.new $(CACHE_ROOT)/tensors.sha256

# These targets compile nothing. Invoke model only through Archy's GPU queue.
identity:
	$(if $(GPUQ_JOB_ID),,$(error Invoke model through gpuq))
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
	gawk -v count=$(TOKENS) 'BEGIN { for (i = 0; i < count; ++i) print 10 + i }' > $(GPUQ_RUN_DIR)/ids.txt
	sha256sum $(GPUQ_RUN_DIR)/ids.txt
	env --chdir=$(ROOT)/kern -u MODEL_NOTWIN -u MODEL_NPH -u MODEL_INJECT -u MODEL_DUMP_S -u MODEL_DUMP_DIR -u MODEL_DUMP_TWIN MODEL_ATTN=qsa MODEL_IDS=$(GPUQ_RUN_DIR)/ids.txt MODEL_PLE=$(PLE_GGUF) MODEL_PLE_OFF=192 MODEL_CAPTURE_QSA=$(CAPTURE) FLASH_NBLK=$(BLOCKS) FLASH_KV_HOST=$(HOST_KV) SPLIT_SHARES=$(SHARES) ATTN_MAP=$(ATTENTION_MAP) CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 $(if $(INJECT),MODEL_INJECT=$(INJECT)) ./model_test $(MODEL_CACHE) 0 48 $(TOKENS) $(FLAGS) $(MANIFESTS)

direct-identity:
	$(if $(GPUQ_JOB_ID),,$(error Invoke direct-matrix through gpuq))
	git -C $(ROOT) rev-parse HEAD
	git -C $(ROOT) status --short
	sha256sum $(ROOT)/kern/qsa_direct_test $(ROOT)/kern/die.cubin /usr/lib/libcuda.so.470.256.02
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
