die_exact.sass exact_ops.h &: die.sass dieops.h exactbatch/link.lua exactbatch/frame.sass exactbatch/operators.sass
	$(LUA) exactbatch/link.lua die.sass dieops.h exactbatch die_exact.sass exact_ops.h

die_exact.cubin: die_exact.sass die_exact.ptx
	$(PTXAS11) -arch=sm_52 -o $@ die_exact.ptx
	$(MAXAS) -i die_exact.sass $@

EXACT_TEST_SRC = exactbatch/prog_group.c prog.c ../ref/quant.c ../ref/gguf.c ../ref/split.c
libengine_exact.so: $(ENGINE_SRC) exactbatch/prog_group.c exactbatch/prog_group.h exact_ops.h prog.h engine.h core.h prefill.h trace_host.inc timings.h qsa_state.h ple_read_pool.h ple_read_reload.h
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -fPIC -shared -pthread -DFLASH_EXACT_BATCH -DFLASH_REV='"$(FLASH_REV)-exact"' -o $@.new $(ENGINE_SRC) exactbatch/prog_group.c -I. -I../ref -I$(CUDA)/include -lcuda -lm && mv -f $@.new $@

exact_bf16_test: exactbatch/bf16_test.c exact_ops.h exactbatch/prog_group.h $(EXACT_TEST_SRC)
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ exactbatch/bf16_test.c $(EXACT_TEST_SRC) -I. -Iexactbatch -I../ref -I$(CUDA)/include -lcuda -lm
exact_q2_test: exactbatch/q2_test.c exact_ops.h exactbatch/prog_group.h $(EXACT_TEST_SRC)
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ exactbatch/q2_test.c $(EXACT_TEST_SRC) -I. -Iexactbatch -I../ref -I$(CUDA)/include -lcuda -lm

exact_state_test: pair_state_test.c exact_ops.h exactbatch/prog_group.h $(EXACT_TEST_SRC)
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -DFLASH_EXACT_STATE -o $@ pair_state_test.c $(EXACT_TEST_SRC) -I. -Iexactbatch -I../ref -I$(CUDA)/include -lcuda -lm

die_verify.sass: die_exact.sass exactbatch/checkpoints.lua exactbatch/copy.sass
	$(LUA) exactbatch/checkpoints.lua die_exact.sass $@ exactbatch/copy.sass
die_verify.cubin: die_verify.sass die_verify.ptx
	$(PTXAS11) -arch=sm_52 -o $@ die_verify.ptx
	$(MAXAS) -i die_verify.sass $@
verify_state_test: pair_state_test.c exact_ops.h exactbatch/prog_group.h $(EXACT_TEST_SRC)
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -DFLASH_EXACT_STATE -DFLASH_VERIFY_STATE -o $@ pair_state_test.c $(EXACT_TEST_SRC) -I. -Iexactbatch -I../ref -I$(CUDA)/include -lcuda -lm
libengine_verify.so: $(ENGINE_SRC) exactbatch/prog_group.c exactbatch/prog_group.h exactbatch/engine_verify.c exact_ops.h prog.h engine.h core.h prefill.h trace_host.inc timings.h qsa_state.h ple_read_pool.h ple_read_reload.h
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -fPIC -shared -pthread -DFLASH_EXACT_BATCH -DFLASH_VERIFY_NATIVE -DFLASH_REV='"$(FLASH_REV)-verify"' -o $@.new $(ENGINE_SRC) exactbatch/prog_group.c -I. -I../ref -I$(CUDA)/include -lcuda -lm && mv -f $@.new $@
