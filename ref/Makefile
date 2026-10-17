CC ?= gcc
CFLAGS ?= -std=gnu99 -O2 -Wall -Wextra
SHARD1 ?= /mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf

all: ref_test ggufhdr

ref_test: ref_test.c gguf.c quant.c gguf.h quant.h
	$(CC) $(CFLAGS) -o $@ ref_test.c gguf.c quant.c -lm

ggufhdr: ggufhdr.c
	$(CC) $(CFLAGS) -o $@ $<

test: ref_test
	./ref_test $(SHARD1)

clean:
	rm -f ref_test ggufhdr attn_contract_test

attn_contract_test: attn_contract_test.c attn.c attn.h checked.c checked.h qsa.c qsa.h fmath.c fmath.h
	$(CC) $(CFLAGS) -fno-fast-math -ffp-contract=off -o $@ attn_contract_test.c checked.c qsa.c attn.c fmath.c -lm

test-attn-contract: attn_contract_test
	./attn_contract_test

.PHONY: all test test-attn-contract clean
