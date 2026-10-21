CC ?= cc
CFLAGS ?= -std=c11 -O2 -Wall -Wextra

.PHONY: all clean

all: fake_core run test_fixture fake_core_probe

fake_core: fake_core.c
	$(CC) $(CFLAGS) -o $@ $<

run: run.c
	$(CC) $(CFLAGS) -o $@ $<

test_fixture: test_fixture.c test_fixture.h
	$(CC) $(CFLAGS) -o $@ test_fixture.c

fake_core_probe: fake_core_probe.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f fake_core run test_fixture fake_core_probe
