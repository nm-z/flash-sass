/* CPU-only fixture dimensions for the QSA GPU harness. */
#ifndef FLASH_QSA_TEST_BOUNDS_H
#define FLASH_QSA_TEST_BOUNDS_H

#define QSA_TEST_CAPACITY 262144u

/* tail is a generated choice in 0..3, not an extra user-supplied position.
 * Keep every requested complete block and fit its tail within capacity.
 * Zero complete blocks still have the appended current token at position 0.
 * Return 1 without changing *position when the dimensions are invalid. */
static inline int qsa_idxscore_fixture_position(unsigned blocks, unsigned tail,
                                               unsigned capacity,
                                               unsigned *position) {
    if (!position || !capacity || tail > 3u || blocks > capacity / 4u)
        return 1;
    if (!blocks) {
        *position = 0;
        return 0;
    }
    unsigned visible = 4u * blocks;
    unsigned remaining = capacity - visible;
    visible += tail < remaining ? tail : remaining;
    *position = visible - 1u;
    return 0;
}

#endif
