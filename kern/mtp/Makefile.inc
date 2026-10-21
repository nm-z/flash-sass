MTP_WIDTHS = 320 640 2560 5120 6144 10240
MTP_SASS = $(addprefix mtp/build/q8_,$(addsuffix .sass,$(MTP_WIDTHS))) $(addprefix mtp/build/,swiglu.sass weighted_sum.sass eh_concat.sass)
mtp/build:
	mkdir -p $@
mtp/build/q8_%.sass: mtp_q8_gen | mtp/build
	./mtp_q8_gen $* --phase > $@
mtp/build/%.sass: mtp/helpers.lua | mtp/build
	$(LUA) mtp/helpers.lua $* > $@
die_mtp.sass mtp_ops.h &: die.sass dieops.h mtp/link.lua mtp/schedule.lua mtp/prefetch.lua $(MTP_SASS)
	$(LUA) mtp/link.lua die.sass dieops.h die_mtp.sass mtp_ops.h mtp/build
die_mtp.cubin: die_mtp.sass die_mtp.ptx
	$(PTXAS11) -arch=sm_52 -o $@ die_mtp.ptx
	$(MAXAS) -i die_mtp.sass $@
mtp_phase_test: mtp_q8_test.c mtp/prog.c mtp/prog.h mtp_ops.h prog.c ../ref/quant.c ../ref/gguf.c ../ref/split.c ../ref/fmath.c
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -DMTP_PHASE_TEST -o $@ mtp_q8_test.c mtp/prog.c prog.c ../ref/quant.c ../ref/gguf.c ../ref/split.c ../ref/fmath.c -I. -I../ref -I$(CUDA)/include -lcuda -lm
mtp/model.o: mtp/model.c mtp/model.h mtp/prog.h mtp_ops.h prog.h
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -Wextra -Werror -fPIC -c -o $@ mtp/model.c -I. -I../ref -I$(CUDA)/include
mtp/load.o: mtp/load.c mtp/load.h mtp/model.h mtp_ops.h prog.h ../ref/quant.h ../ref/gguf.h
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -Wextra -Werror -fPIC -c -o $@ mtp/load.c -I. -I../ref -I$(CUDA)/include

MTP_ENGINE_SRC = mtp/prog.c mtp/model.c mtp/load.c
mtp_load_test: mtp/load_test.c mtp/load.c mtp/load.h mtp/model.h mtp_ops.h prog.c ../ref/quant.c ../ref/gguf.c ../ref/split.c
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ mtp/load_test.c mtp/load.c prog.c ../ref/quant.c ../ref/gguf.c ../ref/split.c -I. -I../ref -I$(CUDA)/include -lm

libengine_mtp.so: $(ENGINE_SRC) $(MTP_ENGINE_SRC) mtp/engine_mtp.c mtp/model.h mtp/load.h mtp/prog.h mtp_ops.h exact_ops.h exactbatch/prog_group.c exactbatch/prog_group.h prog.h engine.h core.h prefill.h trace_host.inc timings.h qsa_state.h ple_read_pool.h ple_read_reload.h
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -fPIC -shared -pthread -DFLASH_EXACT_BATCH -DFLASH_MTP_NATIVE -DFLASH_REV='"$(FLASH_REV)-mtp"' -o $@.new $(ENGINE_SRC) $(MTP_ENGINE_SRC) exactbatch/prog_group.c -I. -I../ref -I$(CUDA)/include -lcuda -lm && mv -f $@.new $@
