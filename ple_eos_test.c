/* The published PLE metadata resets at endoftext, not the chat turn terminator. */
#include "ple.h"
#include <stdio.h>
#include <string.h>
int main(void) {
    const uint32_t eos = PLE_EOS_TOKEN, ids[] = {42, 248046, 198};
    uint64_t got[16], reset[16], pad[16];
    ple_rows(ids, 2, eos, &eos, 1, got);
    /* Independent unsigned-64 XOR/modulo witness for [198, 248046, 42]. */
    if (got[0] != 510813 || got[8] != 165588995) {
        fprintf(stderr, "chat-turn history lost: rows %llu/%llu, expected 510813/165588995\n",
                (unsigned long long)got[0], (unsigned long long)got[8]); return 1;
    }
    const uint32_t at_end[] = {42, 248044, 198}, absent[] = {248044, 248044, 198};
    ple_rows(at_end, 2, eos, &eos, 1, reset);
    ple_rows(absent, 2, eos, &eos, 1, pad);
    if (memcmp(reset, pad, sizeof reset)) { fputs("PLE EOS did not reset the window\n", stderr); return 1; }
    if (!memcmp(got, reset, sizeof got)) { fputs("turn and PLE EOS were conflated\n", stderr); return 1; }
    puts("PASS: im_end preserves history; PLE endoftext resets all 16 heads");
    return 0;
}
