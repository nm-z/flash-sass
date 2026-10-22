die_exact.sass exact_ops.h &: die.sass dieops.h exactbatch/link.lua exactbatch/frame.sass exactbatch/operators.sass
	$(LUA) exactbatch/link.lua die.sass dieops.h exactbatch die_exact.sass exact_ops.h

die_exact.cubin: die_exact.sass die_exact.ptx
	$(PTXAS11) -arch=sm_52 -o $@ die_exact.ptx
	$(MAXAS) -i die_exact.sass $@

die_gr.sass: die.sass die_exact.sass gr/link.lua
	$(LUA) gr/link.lua die.sass die_exact.sass $@
die_gr.cubin: die_gr.sass die_gr.ptx
	$(PTXAS11) -arch=sm_52 -o $@ die_gr.ptx
	$(MAXAS) -i die_gr.sass $@

die_phase.sass: die.sass die_gr.sass phase/gr_cache.lua phase/quant.lua phase/local.lua phase/publish.lua phase/gdn.lua phase/moe.lua phase/router.lua phase/chain.lua phase/attention.lua phase/quant_emit.lua phase/select.lua phase/shared.lua phase/score_select.lua phase/part_merge.lua phase/ple.lua phase/ple_rows.lua phase/edges.lua phase/waits.lua phase/kv_tail.lua phase/frame_dependencies.lua phase/predicate_latency.lua die_fused.sass
	$(LUA) phase/gr_cache.lua die.sass die_gr.sass $@.cached
	$(LUA) phase/quant.lua die.sass $@.cached $@.quant
	$(LUA) phase/local.lua die.sass $@.quant $@.local
	$(LUA) phase/gdn.lua die.sass $@.local $@.gdn
	$(LUA) phase/publish.lua $@.gdn $@.publish
	$(LUA) phase/moe.lua die_fused.sass $@.publish $@.moe
	$(LUA) phase/router.lua die.sass $@.moe $@.router
	$(LUA) phase/attention.lua die.sass $@.router $@.attention
	$(LUA) phase/select.lua die.sass $@.attention $@.select
	$(LUA) phase/shared.lua $@.select $@.shared
	$(LUA) phase/score_select.lua die.sass $@.shared $@.score
	$(LUA) phase/part_merge.lua die.sass $@.score $@.part
	$(LUA) phase/ple.lua die.sass $@.part $@.ple
	$(LUA) phase/ple_rows.lua die.sass $@.ple $@.plerows
	$(LUA) phase/edges.lua $@.plerows $@.edges
	$(LUA) phase/chain.lua $@.edges $@.chain
	$(LUA) phase/waits.lua $@.chain $@.waits
	$(LUA) phase/kv_tail.lua $@.waits $@.tail
	$(LUA) phase/frame_dependencies.lua $@.tail $@.frame
	$(LUA) phase/mutable_loads.lua $@.frame $@.coherent $@.loads.csv
	$(LUA) phase/serial_dependencies.lua $@.coherent $@.plecomplete all fixedbody plegatherq,plenormgate,pleconv selected
	$(LUA) phase/serial_dependencies.lua $@.plecomplete $@.grcomplete all none bf16hcrwx,grreadsum,hcwsum selected
	$(LUA) phase/serial_dependencies.lua $@.grcomplete $@.gdncomplete all none gdn selected
	$(LUA) phase/fixed_latency.lua $@.gdncomplete $@.fixed $@.fixed.csv
	$(LUA) phase/pipeline_completions.lua $@.fixed $@.pipeline
	$(LUA) phase/predicate_latency.lua $@.pipeline $@ $@.predicate.csv

die_deterministic.sass: die.sass phase/kv_tail.lua phase/frame_dependencies.lua phase/predicate_latency.lua
	$(LUA) phase/kv_tail.lua $< $@.tail
	$(LUA) phase/frame_dependencies.lua $@.tail $@.frame
	$(LUA) phase/mutable_loads.lua $@.frame $@.coherent $@.loads.csv
	$(LUA) phase/serial_dependencies.lua $@.coherent $@.plecomplete all fixedbody plerows,plegate,pleconv,rms2560,actq256 selected
	$(LUA) phase/serial_dependencies.lua $@.plecomplete $@.grcomplete all none bf16hcrx,hcwsum selected
	$(LUA) phase/serial_dependencies.lua $@.grcomplete $@.gdncomplete all none conv1d,dnstep,dnnormx selected
	$(LUA) phase/fixed_latency.lua $@.gdncomplete $@.fixed $@.fixed.csv
	$(LUA) phase/pipeline_completions.lua $@.fixed $@.pipeline
	$(LUA) phase/predicate_latency.lua $@.pipeline $@ $@.predicate.csv
die_deterministic.cubin: die_deterministic.sass skel_die.cubin
	cp skel_die.cubin $@
	$(MAXAS) -i die_deterministic.sass $@

die_phase.sass die_deterministic.sass: phase/mutable_loads.lua phase/serial_dependencies.lua phase/fixed_latency.lua phase/pipeline_completions.lua exactbatch/Makefile.inc

die_phase_xtrace.sass: die_phase.sass phase/exchange_trace.lua
	$(LUA) phase/exchange_trace.lua $< $@
die_deterministic_xtrace.sass: die_deterministic.sass phase/exchange_trace.lua
	$(LUA) phase/exchange_trace.lua $< $@
die_phase_xtrace.ptx: die_phase.ptx phase/xtrace_marker.ptxinc
	cat $^ > $@
die_deterministic_xtrace.ptx: die.ptx phase/xtrace_marker.ptxinc
	cat $^ > $@
die_phase_xtrace.cubin: die_phase_xtrace.sass die_phase_xtrace.ptx
	$(PTXAS11) -arch=sm_52 -o $@ die_phase_xtrace.ptx
	$(MAXAS) -i die_phase_xtrace.sass $@
die_deterministic_xtrace.cubin: die_deterministic_xtrace.sass die_deterministic_xtrace.ptx
	$(PTXAS11) -arch=sm_52 -o $@ die_deterministic_xtrace.ptx
	$(MAXAS) -i die_deterministic_xtrace.sass $@

die_phase_replica.sass: die_phase.sass phase/gr_local.lua phase/predicate_latency.lua
	$(LUA) phase/gr_local.lua $< $@.local
	$(LUA) phase/predicate_latency.lua $@.local $@ $@.predicate.csv
die_phase_replica.ptx: die_phase.ptx phase/gr_local_marker.ptxinc
	cat $^ > $@
die_phase_replica.cubin: die_phase_replica.sass die_phase_replica.ptx
	$(PTXAS11) -arch=sm_52 -o $@ die_phase_replica.ptx
	$(MAXAS) -i die_phase_replica.sass $@
die_phase_replica_xtrace.sass: die_phase_replica.sass phase/exchange_trace.lua
	$(LUA) phase/exchange_trace.lua $< $@
die_phase_replica_xtrace.ptx: die_phase_replica.ptx phase/xtrace_marker.ptxinc
	cat $^ > $@
die_phase_replica_xtrace.cubin: die_phase_replica_xtrace.sass die_phase_replica_xtrace.ptx
	$(PTXAS11) -arch=sm_52 -o $@ die_phase_replica_xtrace.ptx
	$(MAXAS) -i die_phase_replica_xtrace.sass $@
die_phase.cubin: die_phase.sass die_phase.ptx
	$(PTXAS11) -arch=sm_52 -o $@ die_phase.ptx
	$(MAXAS) -i die_phase.sass $@

die_balanced.sass: die_phase.sass balance/embedding.lua balance/ple.lua balance/moe.lua balance/prepare.lua balance/attention.lua balance/attention_heads.lua balance/head.lua balance/head_producer.lua balance/publish.lua balance/quant_publish.lua balance/schedule.lua balance/composite.lua phase/quant_emit.lua phase/serial_dependencies.lua phase/predicate_latency.lua
	$(LUA) balance/embedding.lua $< $@.embedding
	$(LUA) balance/ple.lua $@.embedding $@.ple
	$(LUA) balance/moe.lua $@.ple $@.moe
	$(LUA) balance/attention.lua $@.moe $@.attention
	$(LUA) balance/head.lua $@.attention $@.head
	$(LUA) balance/publish.lua $@.head $@.published
	$(LUA) balance/quant_publish.lua $@.published $@.quantpublished
	$(LUA) phase/serial_dependencies.lua $@.quantpublished $@.complete all none attnpartn,kvgather,attnqkq,attnquant selected
	$(LUA) balance/schedule.lua $@.complete $@.scheduled
	$(LUA) balance/composite.lua $@.scheduled $@.composite
	$(LUA) balance/load_bursts.lua $@.composite $@.bursts
	$(LUA) phase/predicate_latency.lua $@.bursts $@ $@.predicate.csv
die_balanced.ptx: die_phase.ptx balance/marker.ptxinc
	cat $^ > $@
die_balanced.sass: exactbatch/Makefile.inc
die_balanced.sass: balance/load_bursts.lua
die_balanced.cubin: die_balanced.sass die_balanced.ptx
	$(PTXAS11) -arch=sm_52 -o $@ die_balanced.ptx
	$(MAXAS) -i die_balanced.sass $@

# Complete production-image experiment: replicated local GR-up. It has a
# distinct marker so an ordinary image cannot silently execute its descriptors.
die_balanced_localup.sass: die_balanced.sass balance/local_up.lua phase/predicate_latency.lua
	$(LUA) balance/local_up.lua $< $@.raw
	$(LUA) phase/predicate_latency.lua $@.raw $@ $@.predicate.csv
die_balanced_localup.ptx: die_balanced.ptx balance/local_up_marker.ptxinc
	cat $^ > $@
die_balanced_localup.cubin: die_balanced_localup.sass die_balanced_localup.ptx
	$(PTXAS11) -arch=sm_52 -o $@ die_balanced_localup.ptx
	$(MAXAS) -i die_balanced_localup.sass $@

die_balanced_body.sass: die_balanced.sass balance/body_trace.lua phase/exchange_trace.lua phase/predicate_latency.lua
	$(LUA) balance/body_trace.lua $< $@.body
	$(LUA) phase/exchange_trace.lua $@.body $@.exchange
	$(LUA) phase/predicate_latency.lua $@.exchange $@ $@.predicate.csv
die_balanced_body.ptx: die_balanced.ptx phase/xtrace_marker.ptxinc balance/body_marker.ptxinc
	cat $^ > $@
die_balanced_body.cubin: die_balanced_body.sass die_balanced_body.ptx
	$(PTXAS11) -arch=sm_52 -o $@ die_balanced_body.ptx
	$(MAXAS) -i die_balanced_body.sass $@

die_balanced_exchange.sass: die_balanced.sass phase/exchange_trace.lua phase/predicate_latency.lua
	$(LUA) phase/exchange_trace.lua $< $@.exchange
	$(LUA) phase/predicate_latency.lua $@.exchange $@ $@.predicate.csv
die_balanced_exchange.ptx: die_balanced.ptx phase/xtrace_marker.ptxinc
	cat $^ > $@
die_balanced_exchange.cubin: die_balanced_exchange.sass die_balanced_exchange.ptx
	$(PTXAS11) -arch=sm_52 -o $@ die_balanced_exchange.ptx
	$(MAXAS) -i die_balanced_exchange.sass $@

die_phase_grhash.sass: die_phase.sass phase/gr_write_trace.lua
	$(LUA) phase/gr_write_trace.lua $< $@
die_phase_grhash.ptx: die_phase.ptx phase/xtrace_marker.ptxinc phase/gr_write_marker.ptxinc
	cat $^ > $@
die_phase_grhash.cubin: die_phase_grhash.sass die_phase_grhash.ptx
	$(PTXAS11) -arch=sm_52 -o $@ die_phase_grhash.ptx
	$(MAXAS) -i die_phase_grhash.sass $@

die_phase_serial.sass: die_phase.sass phase/serial_dependencies.lua
	$(LUA) phase/serial_dependencies.lua $< $@ $(or $(SERIAL_SCOPE),body) $(SERIAL_FIXED) $(SERIAL_FIXED_OPS)
die_phase_serial.cubin: die_phase_serial.sass die_phase.ptx
	$(PTXAS11) -arch=sm_52 -o $@ die_phase.ptx
	$(MAXAS) -i die_phase_serial.sass $@

libengine.so libengine_exact.so libengine_mtp.so libengine_mtp_chunk.so libengine_dflash.so libengine_verify.so libengine_chunk.so: phase/plan.h phase/ownership.h phase/gr_replica.h phase/exchange_host.inc balance/layout.h balance/program.h balance/attention_program.h balance/attention_plan.h balance/body_host.inc balance/qsa_audit.inc balance/head_audit.inc balance/work_dump.inc balance/ple_dma.h

die_gr_fp8.sass: die_gr.sass fp8/link.lua
	$(LUA) fp8/link.lua $< $@
die_chunk_fp8.sass: die_chunk.sass fp8/link.lua
	$(LUA) fp8/link.lua $< $@
die_gr_fp8.cubin: die_gr_fp8.sass die_gr_fp8.ptx
	$(PTXAS11) -arch=sm_52 -o $@ die_gr_fp8.ptx
	$(MAXAS) -i die_gr_fp8.sass $@
die_chunk_fp8.cubin: die_chunk_fp8.sass die_chunk_fp8.ptx
	$(PTXAS11) -arch=sm_52 -o $@ die_chunk_fp8.ptx
	$(MAXAS) -i die_chunk_fp8.sass $@

libengine.so libengine_exact.so libengine_mtp.so libengine_mtp_chunk.so libengine_dflash.so libengine_verify.so libengine_chunk.so: fp8/storage.h

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

# Descriptor compilation only: deliberately no CUDA library and no model math.
phase_schedule: phase/schedule.c phase/schedule_symbols.inc engine.c phase/plan.h phase/ownership.h balance/layout.h balance/program.h balance/attention_program.h balance/attention_plan.h prog.c prog.h holdc.c hold.h ../ref/split.c ../ref/quant.c ../ref/gguf.c
	$(CC) -std=gnu99 -O2 -ffunction-sections -fdata-sections -Wl,--gc-sections -I$(CUDA)/include -I. -I../ref -o $@ phase/schedule.c prog.c holdc.c ../ref/split.c ../ref/quant.c ../ref/gguf.c -lm
