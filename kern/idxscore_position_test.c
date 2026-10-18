/* CPU-only regression for the GPU harness's generated token positions. */
#include "qsa_test_bounds.h"
#include <limits.h>
#include <stdio.h>

static unsigned failures, checks;

static void accepted(unsigned blocks, unsigned tail, unsigned capacity,
                     unsigned expected) {
    unsigned position = 77;
    int rc = qsa_idxscore_fixture_position(blocks, tail, capacity, &position);
    checks++;
    if (rc || position != expected || position >= capacity ||
        (position + 1u) / 4u != blocks) {
        fprintf(stderr, "FAIL blocks=%u tail=%u capacity=%u: rc=%d position=%u expected=%u\n",
                blocks, tail, capacity, rc, position, expected);
        failures++;
    }
}

static void rejected(unsigned blocks, unsigned tail, unsigned capacity) {
    unsigned position = 77;
    int rc = qsa_idxscore_fixture_position(blocks, tail, capacity, &position);
    checks++;
    if (rc != 1 || position != 77) {
        fprintf(stderr, "FAIL rejection blocks=%u tail=%u capacity=%u: rc=%d position=%u\n",
                blocks, tail, capacity, rc, position);
        failures++;
    }
}

int main(void) {
    for (unsigned tail = 0; tail < 4; tail++) {
        accepted(0, tail, 1, 0);
        accepted(0, tail, QSA_TEST_CAPACITY, 0);
        accepted(1, tail, 5, tail ? 4 : 3);
        accepted(65535, tail, QSA_TEST_CAPACITY, 262139u + tail);
        accepted(65536, tail, QSA_TEST_CAPACITY, 262143);
        accepted(UINT_MAX / 4u, tail, UINT_MAX,
                 UINT_MAX - 4u + tail);
        rejected(65537, tail, QSA_TEST_CAPACITY);
    }
    rejected(0, 0, 0);
    rejected(0, 4, QSA_TEST_CAPACITY);
    rejected(1, 4, QSA_TEST_CAPACITY);
    rejected(1, 0, 3);
    rejected(UINT_MAX / 4u + 1u, 0, UINT_MAX);
    rejected(UINT_MAX, 0, UINT_MAX);
    checks++;
    if (qsa_idxscore_fixture_position(1, 0, 4, NULL) != 1) {
        fputs("FAIL null output accepted\n", stderr);
        failures++;
    }
    printf("%s: %u position checks; max blocks=65536 tails=0..3 position=262143 capacity=262144\n",
           failures ? "FAIL" : "PASS", checks);
    puts("UINT_MAX cases check integer arithmetic only, not GPU allocation support.");
    return failures ? 1 : 0;
}
