CC ?= cc
TEST_BINARY ?= /tmp/fs101-gpuq-empty-artifact-test
TEST_SCRATCH ?= /tmp
TEST_SOURCE := $(dir $(lastword $(MAKEFILE_LIST)))test-empty-artifact.c

.PHONY: test reproduce clean

test: $(TEST_BINARY)
	$(TEST_BINARY) --test "$(CURDIR)" "$(TEST_SCRATCH)"

reproduce: $(TEST_BINARY)
	$(TEST_BINARY) --reproduce "$(CURDIR)" "$(TEST_SCRATCH)"

$(TEST_BINARY): $(TEST_SOURCE)
	$(CC) -std=c11 -Wall -Wextra -Werror -O2 -o "$@" "$<"

clean:
	$(RM) "$(TEST_BINARY)"
