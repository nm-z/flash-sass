/* CPU-only decoder contract. Literal tags are the on-wire GPU ABI; no CUDA
 * initialization or GPU allocation is performed by this executable. */
#include "phase_probe.h"
#include <stdio.h>

static unsigned checks, failures;
static void check(int condition, const char *name) {
    checks++;
    if (!condition) { failures++; fprintf(stderr, "FAIL %s\n", name); }
}

int main(void) {
    phase_probe_descriptor scalar[] = {{OP_XCOPY, 10, 0}, {OP_XCOPY, 11, 0}};
    phase_probe_descriptor generic[] = {{OP_XCOPY, 10, 8}, {OP_XCOPY, 11, 0}};
    phase_probe_descriptor fused[] = {{OP_BF16K2560P, 10, 24}, {OP_BF16K2560, 11, 0}};
    phase_probe_record record;
    int rc = phase_probe_decode(UINT64_C(0x0000000700000020), scalar, 2, 7, 9, &record);
    check(!rc && record.token == 7 && record.base_token == 7 &&
          record.primary_index == 0 && record.physical_index == 0 &&
          record.mode == 0 && record.descriptor.units == 10, "scalar phase zero");
    rc = phase_probe_decode(UINT64_C(0x0000000800000040), scalar, 2, 7, 9, &record);
    check(!rc && record.token == 8 && record.physical_index == 1 &&
          record.descriptor.units == 11, "scalar final phase and token");
    rc = phase_probe_decode(UINT64_C(0x0000000700000028), generic, 2, 7, 9, &record);
    check(!rc && record.token == 7 && record.physical_index == 0 && record.mode == 8,
          "generic first body");
    rc = phase_probe_decode(UINT64_C(0x0000000700000020), generic, 2, 7, 9, &record);
    check(!rc && record.token == 8 && record.base_token == 7 && record.physical_index == 1 &&
          record.descriptor.units == 11, "generic second body");
    rc = phase_probe_decode(UINT64_C(0x0000000700000038), fused, 2, 7, 9, &record);
    check(!rc && record.token == 7 && record.physical_index == 0 && record.mode == 24 &&
          record.descriptor.opcode == OP_BF16K2560P, "fused two-position body");
    check(phase_probe_decode(0, scalar, 2, 7, 9, &record) < 0, "unwritten tag rejected");
    check(phase_probe_decode(UINT64_C(0x0000000700000000), scalar, 2, 7, 9, &record) < 0,
          "zero encoded phase rejected");
    check(phase_probe_decode(UINT64_C(0x0000000700000060), scalar, 2, 7, 9, &record) < 0,
          "phase outside uploaded program rejected");
    check(phase_probe_decode(UINT64_C(0x0000000700000030), generic, 2, 7, 9, &record) < 0,
          "reserved mode rejected");
    check(phase_probe_decode(UINT64_C(0x0000000700000028), scalar, 2, 7, 9, &record) < 0,
          "unexpected generic mode rejected");
    check(phase_probe_decode(UINT64_C(0x0000000700000020), fused, 2, 7, 9, &record) < 0,
          "unexpected scalar mode in fused body rejected");
    check(phase_probe_decode(UINT64_C(0x0000000800000020), generic, 2, 7, 9, &record) < 0,
          "second body outside segment rejected");
    check(phase_probe_decode(UINT64_C(0x0000000700000028), generic, 2, 7, 8, &record) < 0,
          "truncated generic pair rejected");
    check(phase_probe_decode(UINT64_C(0x0000000700000038), fused, 2, 7, 8, &record) < 0,
          "truncated fused pair rejected");
    check(phase_probe_decode(UINT64_C(0x0000000700000040), generic, 2, 7, 9, &record) < 0,
          "secondary descriptor used as primary rejected");
    check(phase_probe_decode(UINT64_C(0x0000000600000020), scalar, 2, 7, 9, &record) < 0,
          "token before segment rejected");
    check(phase_probe_decode(UINT64_C(0x0000000900000020), scalar, 2, 7, 9, &record) < 0,
          "token at segment end rejected");
    check(phase_probe_decode(UINT64_C(0x0000000700000020), scalar, 2, 9, 7, &record) < 0,
          "invalid segment rejected");
    check(phase_probe_decode(UINT64_C(0x0000000700000020), NULL, 2, 7, 9, &record) < 0,
          "missing program rejected");
    check(phase_probe_decode(UINT64_C(0x0000000700000020), scalar, 2, 7, 9, NULL) < 0,
          "missing output rejected");
    check(!phase_probe_enabled(), "decoder test performs no CUDA setup");
    printf("phase_probe_test: checks=%u failures=%u cuda_initialization=not-run\n", checks, failures);
    return failures ? 1 : 0;
}
