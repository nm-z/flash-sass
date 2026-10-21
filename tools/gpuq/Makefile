CC ?= cc
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Werror
LDLIBS ?= -lsqlite3

.PHONY: test-sqlite-transactions
test-sqlite-transactions:
	@set -eu; \
	test_bin=$$(mktemp /tmp/gpuq-sqlite-transactions.XXXXXX); \
	trap 'rm -f "$$test_bin"' EXIT; \
	$(CC) $(CPPFLAGS) $(CFLAGS) -o "$$test_bin" sqlite_transaction_test.c $(LDFLAGS) $(LDLIBS); \
	"$$test_bin"
