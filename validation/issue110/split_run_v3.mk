ROOT := /dev/shm/fs110-issue110-split-v3-7e163718
ENGINE_CHECK := $(ROOT)/validation/issue9/bin/engine_check
MANIFESTS := /tmp/hold-h0.manifest /tmp/hold-h1.manifest /tmp/hold-h2.manifest /tmp/hold-h3.manifest /tmp/hold-h4.manifest /tmp/hold-h5.manifest
GGUF := /mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
PLE ?= /home/nate/codex/flash-astra-perf-20261018/ple-direct.gguf
WEIGHT_MANIFEST ?= $(ROOT)/ple-direct.sha256
TOKENS ?= 800
SHARES := 112,97,81,116,97,97
ATTN_MAP := 1,1,1,1,4,4,4,4,5,5,5,5
REVISION := 7e16371866d6bd183a23eee78736016326067f83
SNAPSHOTS ?= 512,515
ENGINE_TIMEOUT ?= 3600
HOLDER_PIDS := $(shell ps -C hold -o pid=,args= | gawk -v gguf='$(GGUF)' -f $(ROOT)/validation/issue110/holders.awk)
LOCK_FD := $(if $(GPUQ_LOCK_FD),$(GPUQ_LOCK_FD),9)

.PHONY: issue110-single
issue110-single:
	$(if $(filter 6,$(words $(HOLDER_PIDS))),,$(error expected six validated holder PIDs after queue admission))
	$(ENGINE_CHECK) --repo $(ROOT) --out $(GPUQ_RUN_DIR)/engine-matrix --ple $(PLE) --revision $(REVISION) $(foreach path,$(MANIFESTS),--manifest $(path)) $(foreach pid,$(HOLDER_PIDS),--holder-pid $(pid)) --cuda-visible 0,1,2,3,4,5 --split-shares $(SHARES) --attn-map $(ATTN_MAP) --tokens $(TOKENS) --snapshots $(SNAPSHOTS) --nblk 512 --lock-fd $(LOCK_FD) --timeout $(ENGINE_TIMEOUT) --weight-manifest $(WEIGHT_MANIFEST) --first-only > $(GPUQ_RUN_DIR)/engine-check.stdout 2> $(GPUQ_RUN_DIR)/engine-check.stderr
