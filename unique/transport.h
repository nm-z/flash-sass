#ifndef FLASH_PAIR_TRANSPORT_H
#define FLASH_PAIR_TRANSPORT_H
/* Per-position, per-die tile epochs. All six dies use the same 40 output
 * tiles; the owner board for tile t is t % 3. Payloads remain FP32. */
#define PAIR_TILES 40u
#define PAIR_PARTIAL_READY 0u
#define PAIR_REDUCE_READY 40u
#define PAIR_FINAL_READY 160u
#define PAIR_FLAG_WORDS 200u
#define PAIR_TABLE_POINTERS 22u
/* table[0..3]: destination/source of odd -> even local partial and its flag.
 * [4..7]: this die's reduction input/flags and final output/flags.
 * [8..9]: even -> odd final output/flag destinations.
 * [10..12], [13..15]: reduction data/flag destinations by owner board.
 * [16..18], [19..21]: final broadcast data/flag destinations by board. */
#endif
