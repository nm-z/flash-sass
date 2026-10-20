/* Test-only libmtmd interposer. Never load in the service. */
#define _GNU_SOURCE
#include "mtmd.h"
#include <dlfcn.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <unistd.h>

static int claim_fault(void) {
    const char *marker = getenv("FLASHVIS_TEST_FAULT_ONCE");
    if (!marker) return 1;
    int fd = open(marker, O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) return 0;
    close(fd);
    return 1;
}

/* The executable's native-aligned dispatch uses RTLD_NEXT. Abort at that
 * boundary once across worker restarts, like a native CUDA_CHECK failure. */
void native_abort(void *opaque, struct ggml_tensor *dst)
    __asm__("_Z24ggml_cuda_flash_attn_extR25ggml_backend_cuda_contextP11ggml_tensor");
void native_abort(void *opaque, struct ggml_tensor *dst) {
    const char *mode = getenv("FLASHVIS_TEST_FAULT");
    if (mode && !strcmp(mode, "native-abort") && claim_fault()) {
            (void)setrlimit(RLIMIT_CORE, &(struct rlimit){0, 0});
            fprintf(stderr, "issue96-test: injecting native GPU FA dispatch abort\n");
            abort();
    }
    void (*real)(void *, struct ggml_tensor *) = dlsym(RTLD_NEXT,
        "_Z24ggml_cuda_flash_attn_extR25ggml_backend_cuda_contextP11ggml_tensor");
    if (!real) abort();
    real(opaque, dst);
}

int32_t mtmd_encode_chunk(mtmd_context *ctx, const mtmd_input_chunk *chunk) {
    static int calls;
    const char *mode = getenv("FLASHVIS_TEST_FAULT");
    if (calls++ == 0 && mode && !strcmp(mode, "error") && claim_fault()) {
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
    if (calls++ == 0 && output && mode && !strcmp(mode, "nan") && claim_fault()) {
        fprintf(stderr, "issue96-test: injecting one non-finite GPU embedding\n");
        output[0] = NAN;
    }
    return output;
}
