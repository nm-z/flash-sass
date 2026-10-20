ISSUE109_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
include $(ISSUE109_DIR)/../issue9/Makefile
.DEFAULT_GOAL := pair-build
PAIR_OUT := $(ISSUE109_DIR)/bin
PAIR_MAXAS := env PATH=/opt/cuda/bin:/usr/bin:/bin perl -I$(MAXAS_ROOT)/lib $(MAXAS_ROOT)/bin/maxas.pl

ifneq ($(GPUQ_JOB_ID),)
ifneq ($(filter pair-build pair-check,$(MAKECMDGOALS)),)
$(error Run pair diagnostic builds/checks through dsh build outside gpuq)
endif
endif

.PHONY: pair-build pair-check pair-matrix
$(PAIR_OUT):
	mkdir -p $@

$(PAIR_OUT)/pair_barrier_patch: $(ISSUE109_DIR)/pair_barrier_patch.c | $(PAIR_OUT)
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror -o $@ $<

$(PAIR_OUT)/pair_barrier_test: $(ISSUE109_DIR)/pair_barrier_test.c | $(PAIR_OUT)
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror -o $@ $<

# Fresh DSH workspaces retain both the pinned baseline and the transformed SASS.
# The inherited identity target checks this final receipt before any GPU run.
pair-build: kernel-build $(PAIR_OUT)/pair_barrier_patch
	env --chdir=$(ROOT) sha256sum -c validation/issue109/baseline.sha256
	cp $(ROOT)/kern/die.cubin $(PAIR_OUT)/baseline.cubin
	$(PAIR_OUT)/pair_barrier_patch $(ROOT)/kern/die.sass $(PAIR_OUT)/die-pair-grid.sass
	cp $(ROOT)/kern/skel_die.cubin $(ROOT)/kern/die.cubin
	$(PAIR_MAXAS) -i $(PAIR_OUT)/die-pair-grid.sass $(ROOT)/kern/die.cubin
	env --chdir=$(ROOT) sha256sum kern/die.cubin kern/die.sass validation/issue109/pair_barrier_patch.c validation/issue109/bin/die-pair-grid.sass $(patsubst $(ROOT)/%,%,$(KERNEL_INPUTS)) $(MAXAS_ROOT)/lib/MaxAs/MaxAsGrammar.pm $(MAXAS_ROOT)/lib/MaxAs/MaxAs.pm $(MAXAS_ROOT)/lib/MaxAs/Cubin.pm > $(KERNEL_RECEIPT)
	$(MAKE) -f $(ISSUE109_DIR)/../issue9/Makefile ENGINE_REVISION=$(ENGINE_REVISION) engine-build
	$(MAKE) -C $(ROOT)/kern model_test
	sha256sum $(ROOT)/kern/die.cubin $(ROOT)/kern/libengine.so $(ROOT)/kern/flashcore $(OUT_DIR)/engine_check $(PAIR_OUT)/pair_barrier_patch

pair-check: $(PAIR_OUT)/pair_barrier_patch $(PAIR_OUT)/pair_barrier_test
	$(MAKE) -B -C $(ROOT)/kern die.sass
	$(PAIR_OUT)/pair_barrier_test $(PAIR_OUT)/pair_barrier_patch $(ROOT)/kern/die.sass

pair-matrix: engine-matrix
