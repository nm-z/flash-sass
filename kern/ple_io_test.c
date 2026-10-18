#define _GNU_SOURCE
#define FLASH_PLE_IO_TEST 1
#include "ple_io.h"
#include <assert.h>
#include <fcntl.h>
#include <stdio.h>

int main(void) {
    char name[] = "/tmp/flash-ple-io-XXXXXX";
    int fd = mkstemp(name);
    assert(fd >= 0 && !unlink(name));
    unsigned char expected[90], actual[90];
    for (int i = 0; i < 90; i++) expected[i] = (unsigned char)(3*i + 1);
    assert(write(fd, expected, 90) == 90);
    ple_io_error e = {0};
    const char *ok[] = {"none", "partial", "eintr"};
    for (int k = 0; k < 3; k++) {
        assert(!ple_io_test_set(ok[k], 7, 3, 17));
        ple_io_test_reset_stats();
        memset(actual, 0, 90);
        assert(!ple_read_exact(fd, actual, 90, 0, 7, 3, &e));
        assert(!memcmp(actual, expected, 90));
        if (k == 0) assert(ple_io_stats.read_calls == 1 && !ple_io_stats.fault_matches);
        if (k == 1) assert(ple_io_stats.partial_reads > 2 && ple_io_stats.read_calls > 2);
        if (k == 2) assert(ple_io_stats.eintr_retries == 1 && ple_io_stats.read_calls == 3);
        printf("PASS %s: reads=%llu partial=%llu EINTR=%llu\n", ok[k],
               ple_io_stats.read_calls, ple_io_stats.partial_reads, ple_io_stats.eintr_retries);
    }
    const char *bad[] = {"eof", "eio"};
    for (int k = 0; k < 2; k++) {
        assert(!ple_io_test_set(bad[k], 7, 3, 17));
        ple_io_test_reset_stats();
        memset(actual, 0xaa, 90);
        assert(ple_read_exact(fd, actual, 90, 0, 7, 3, &e) == -1);
        assert(e.position == 7 && e.head == 3 && e.offset == 0 && e.bytes == 17 && e.expected == 90);
        assert(e.eof == (k == 0) && e.errnum == (k == 0 ? 0 : EIO));
        assert(ple_io_stats.fault_reads == 1 && ple_io_stats.read_calls == 2);
        assert(!memcmp(actual, expected, 17) && actual[17] == 0xaa);
        assert(!ple_read_exact(fd, actual, 90, 0, 8, 3, &e)); /* unmatched row */
        assert(!memcmp(actual, expected, 90));
        printf("PASS %s: exact metadata, unread tail retained, unmatched row succeeds\n", bad[k]);
    }
    assert(ple_io_test_set("unknown", 0, 0, 0) == -1);
    assert(ple_io_test_set("eio", 0, 16, 0) == -1);
    assert(ple_io_test_set("eio", 0, 0, 90) == -1);
    assert(!ple_io_test_set("none", 0, 0, 0));
    assert(!ftruncate(fd, 17));
    assert(ple_read_exact(fd, actual, 90, 10, 2, 4, &e) == -1);
    assert(e.eof && e.bytes == 7 && e.offset == 10);
    assert(ple_read_exact(-1, actual, 90, 0, 2, 4, &e) == -1);
    assert(!e.eof && e.errnum == EBADF && e.bytes == 0);
    assert(!close(fd));
    puts("PASS real truncated-file EOF, invalid-fd error, and plan validation");
    return 0;
}
