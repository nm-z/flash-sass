LUA ?= lua5.4
CHUNK_GEN = chunk/build
CHUNK_GEMM = bf16 q2 q3 q4k q5k q6k iq4 q40 q50 iq4nl q80
CHUNK_SLAB = silu mix write rms conv bias router tr group swiglu saxpy zero rs sum pgate pconv xs xh uh
CHUNK_EB = gu gu2 dn
CHUNK_CA = kv idx sel app attn
CHUNK_SASS = $(addprefix $(CHUNK_GEN)/gemm_,$(addsuffix .sass,$(CHUNK_GEMM))) $(addprefix $(CHUNK_GEN)/slab_,$(addsuffix .sass,$(CHUNK_SLAB))) $(addprefix $(CHUNK_GEN)/ebatch_,$(addsuffix .sass,$(CHUNK_EB))) $(addprefix $(CHUNK_GEN)/cattn_,$(addsuffix .sass,$(CHUNK_CA))) $(CHUNK_GEN)/dnstepc.sass $(CHUNK_GEN)/qact_s.sass $(CHUNK_GEN)/qact_p.sass

$(CHUNK_GEN):
	mkdir -p $@
$(CHUNK_GEN)/gemm_%.sass: chunk/gen_gemm.lua chunk/sassfence.lua | $(CHUNK_GEN)
	GEMM_OP=1 GEMM_FMT=$* $(LUA) $< > $@.tmp
	mv $@.tmp $@
$(CHUNK_GEN)/slab_%.sass: chunk/slab.lua | $(CHUNK_GEN)
	SLAB_OP=$* $(LUA) $< > $@.tmp
	mv $@.tmp $@
$(CHUNK_GEN)/ebatch_%.sass: chunk/ebatch.lua | $(CHUNK_GEN)
	EB_OP=$* EB_WHOLE=1 $(LUA) $< > $@.tmp
	mv $@.tmp $@
$(CHUNK_GEN)/cattn_%.sass: chunk/cattn.lua | $(CHUNK_GEN)
	CA_OP=$* $(LUA) $< > $@.tmp
	mv $@.tmp $@
$(CHUNK_GEN)/dnstepc.sass: chunk/dnstepc.sass | $(CHUNK_GEN)
	cp $< $@
$(CHUNK_GEN)/qact_%.sass: chunk/qact.lua | $(CHUNK_GEN)
	QACT_LAYOUT=$* $(LUA) $< > $@.tmp
	mv $@.tmp $@

die_chunk.sass chunk_ops.h &: die.sass dieops.h chunk/link.lua $(CHUNK_SASS)
	$(LUA) chunk/link.lua die.sass dieops.h die_chunk.sass chunk_ops.h $(CHUNK_GEN)

die_chunk.cubin: die_chunk.sass die_chunk.ptx
	$(PTXAS11) -arch=sm_52 -o $@ die_chunk.ptx
	$(MAXAS) -i die_chunk.sass $@

libengine_chunk.so: $(ENGINE_SRC) chunk/engine_chunk.c chunk/prog_chunk.c chunk/prog_chunk.h chunk_ops.h prog.h
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -fPIC -shared -pthread -DFLASH_CHUNK_NATIVE -DFLASH_REV='"$(FLASH_REV)-chunk"' -o $@.new $(ENGINE_SRC) chunk/prog_chunk.c -I. -I../ref -I$(CUDA)/include -lcuda -lm
	mv $@.new $@

chunk_ops_test: chunk_ops_test.c chunk/prog_chunk.c chunk_ops.h prog.c prog.h ../ref/quant.c ../ref/gguf.c ../ref/split.c
	$(CC) -std=gnu99 -O2 -ffp-contract=off -Wall -o $@ chunk_ops_test.c chunk/prog_chunk.c prog.c ../ref/quant.c ../ref/gguf.c ../ref/split.c -I. -I../ref -I$(CUDA)/include -lcuda -lm
