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
	rm -f ref_test ggufhdr

.PHONY: all test clean
