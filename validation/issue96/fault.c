/* Test-only libmtmd interposer. Never load in the service. */
#define _GNU_SOURCE
#include "mtmd.h"
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int32_t mtmd_encode_chunk(mtmd_context *ctx, const mtmd_input_chunk *chunk) {
    static int calls;
    const char *mode = getenv("FLASHVIS_TEST_FAULT");
    if (calls++ == 0 && mode && !strcmp(mode, "error")) {
        fprintf(stderr, "issue96-test: injecting one reported GPU encoding failure\n");
        return 1;
    }
    int32_t (*real_encode)(mtmd_context *, const mtmd_input_chunk *) = dlsym(RTLD_NEXT, "mtmd_encode_chunk");
    if (!real_encode) abort();
    return real_encode(ctx, chunk);
}

float *mtmd_get_output_embd(mtmd_context *ctx) {
    static int calls;
    float *(*real_output)(mtmd_context *) = dlsym(RTLD_NEXT, "mtmd_get_output_embd");
    if (!real_output) abort();
    float *output = real_output(ctx);
    const char *mode = getenv("FLASHVIS_TEST_FAULT");
    if (calls++ == 0 && output && mode && !strcmp(mode, "nan")) {
        fprintf(stderr, "issue96-test: injecting one non-finite GPU embedding\n");
        output[0] = NAN;
    }
    return output;
}
