SOURCE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ROOT := $(abspath $(SOURCE_DIR)/../..)
OUT_DIR ?= $(CURDIR)/vision-artifacts
CUDA ?= /opt/cuda
LLAMA_CPP ?= $(HOME)/claude/llama.cpp-cuda
LLAMA_BUILD ?= $(LLAMA_CPP)/build
CC ?= cc
CFLAGS := -std=gnu11 -O2 -Wall -Wextra
INCLUDES := -I$(CUDA)/include -I$(LLAMA_CPP)/include -I$(LLAMA_CPP)/ggml/include -I$(LLAMA_CPP)/tools/mtmd
LIBS := -L$(LLAMA_BUILD)/bin -Wl,-rpath,$(LLAMA_BUILD)/bin
MMPROJ ?= /home/nate/models-hdd-backup/Qwen3.8-Flash-Next-GGUF/mmproj-BF16.gguf
TEXT_GGUF ?= /mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
IMAGE ?= $(ROOT)/issue96.png
ORIGINAL_RUN ?= /home/nate/.local/state/gpuq/staging/b1c3a80a-0caa-482c-8ce1-4466da720162
ORACLE_RUN ?= /home/nate/.local/state/gpuq/staging/5f77f1c1-a8e5-4177-8ed9-4dc9347cae74

.PHONY: all encoder cpu-check gpu-check full identity retained-check
all: encoder $(OUT_DIR)/vision_check $(OUT_DIR)/supervisor_fixture $(OUT_DIR)/supervisor_runner $(OUT_DIR)/fa_kernel_test $(OUT_DIR)/fa_native_test $(OUT_DIR)/gpu_reference $(OUT_DIR)/flashvis_fault.so

$(OUT_DIR):
	mkdir -p $@

encoder:
	$(MAKE) -C $(ROOT)/kern flashvis

$(OUT_DIR)/vision_check: $(SOURCE_DIR)/vision_check.c | $(OUT_DIR)
	$(CC) $(CFLAGS) -Werror -o $@ $< -lcrypto -lm

$(OUT_DIR)/supervisor_fixture: $(SOURCE_DIR)/supervisor_test.c $(ROOT)/kern/vis_supervisor.c $(ROOT)/kern/vis_supervisor.h | $(OUT_DIR)
	$(CC) $(CFLAGS) -Werror -I$(ROOT)/kern -o $@ $(SOURCE_DIR)/supervisor_test.c $(ROOT)/kern/vis_supervisor.c

$(OUT_DIR)/supervisor_runner: $(SOURCE_DIR)/supervisor_runner.c | $(OUT_DIR)
	$(CC) $(CFLAGS) -Werror -o $@ $<

$(OUT_DIR)/fa_kernel_test: $(SOURCE_DIR)/fa_kernel_test.c $(ROOT)/kern/vis_fa.c $(ROOT)/kern/vis_fa.h | $(OUT_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -DVIS_FA_NO_GGML -DVIS_FA_PTX_PATH='"$(ROOT)/kern/vis_fa.cubin"' -o $@ $(SOURCE_DIR)/fa_kernel_test.c $(ROOT)/kern/vis_fa.c $(LIBS) -lggml-base -lcuda -lm

$(OUT_DIR)/fa_native_test: $(SOURCE_DIR)/fa_native_test.c | $(OUT_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $< $(LIBS) -lggml -lggml-base -lm

$(OUT_DIR)/gpu_reference: $(SOURCE_DIR)/gpu_reference.c | $(OUT_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -Wl,--export-dynamic -o $@ $< $(LIBS) -lmtmd -lllama -lggml -lggml-base -lm

$(OUT_DIR)/flashvis_fault.so: $(SOURCE_DIR)/fault.c | $(OUT_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -shared -fPIC -o $@ $< -ldl

cpu-check: $(OUT_DIR)/supervisor_fixture $(OUT_DIR)/supervisor_runner $(OUT_DIR)/vision_check
	$(OUT_DIR)/supervisor_runner $(OUT_DIR)/supervisor_fixture $(OUT_DIR)/vision_check
	$(OUT_DIR)/vision_check compare $(ORACLE_RUN)/reference.bin $(ORIGINAL_RUN)/embeddings-0.bin
	$(OUT_DIR)/vision_check retained $(ORIGINAL_RUN)
	$(OUT_DIR)/vision_check fixture 2048x2048
	$(OUT_DIR)/vision_check fixture 4096x4096

# Build first via dsh build. These targets perform no compilation.
gpu-check: identity
	$(OUT_DIR)/vision_check recover $(ROOT)/kern/flashvis $(MMPROJ) $(TEXT_GGUF) $(IMAGE) $(OUT_DIR)/flashvis_fault.so
	$(OUT_DIR)/gpu_reference $(MMPROJ) $(TEXT_GGUF) $(IMAGE) $(GPUQ_RUN_DIR)/reference.bin $(LLAMA_BUILD)/bin
	$(OUT_DIR)/vision_check encode $(ROOT)/kern/flashvis $(MMPROJ) $(TEXT_GGUF) $(IMAGE) 256x256
	$(OUT_DIR)/vision_check compare $(GPUQ_RUN_DIR)/reference.bin $(GPUQ_RUN_DIR)/embeddings-0.bin

full: identity
	$(OUT_DIR)/vision_check encode $(ROOT)/kern/flashvis $(MMPROJ) $(TEXT_GGUF) $(IMAGE) 2048x2048 4096x4096

retained-check:
	$(OUT_DIR)/vision_check retained $(ORIGINAL_RUN)

identity:
	git -C $(ROOT) rev-parse HEAD
	git -C $(ROOT) status --short
	$(CC) --version
	/opt/cuda-11.4/bin/ptxas --version
	sha256sum $(ROOT)/kern/flashvis $(ROOT)/kern/vis_fa.cubin $(ROOT)/kern/vis_fa.ptx
	sha256sum $(OUT_DIR)/vision_check $(OUT_DIR)/gpu_reference $(OUT_DIR)/flashvis_fault.so
	sha256sum $(LLAMA_BUILD)/bin/libmtmd.so $(LLAMA_BUILD)/bin/libllama.so $(LLAMA_BUILD)/bin/libggml-cuda.so
