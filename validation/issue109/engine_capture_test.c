#define _GNU_SOURCE
#include "prog.h"
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <setjmp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* Host memory replaces CUDA calls in this publication/lifecycle fixture.
 * The separate queued xcopy_trace_gpu test validates actual GPU addressing. */
#define MAXD 2
#define FLASH_REV "capture-fixture"
static int nd = MAXD, in_init, in_run, kv_host = 1, two_position;
static CUcontext ctx[MAXD];
static jmp_buf init_jb, run_jb;
static uint64_t trace_request = 7, cubin_fnv = 123;
static struct { void *pointer; size_t bytes; } allocations[32];
static size_t allocation_count;
#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition); exit(1); \
} } while (0)
#define CK(call) CHECK((call) == CUDA_SUCCESS)

static void *host_range(CUdeviceptr pointer, size_t bytes) {
    uintptr_t address = (uintptr_t)pointer;
    for (size_t i = 0; i < allocation_count; i++) {
        uintptr_t base = (uintptr_t)allocations[i].pointer;
        if (address >= base && address - base <= allocations[i].bytes &&
            bytes <= allocations[i].bytes - (address - base))
            return (void *)address;
    }
    return NULL;
}
CUresult CUDAAPI cuCtxSetCurrent(CUcontext context) { (void)context; return CUDA_SUCCESS; }
CUresult CUDAAPI cuMemsetD8(CUdeviceptr pointer, unsigned char value, size_t bytes) {
    void *host = host_range(pointer, bytes);
    if (!host) return CUDA_ERROR_INVALID_VALUE;
    memset(host, value, bytes); return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyDtoH(void *destination, CUdeviceptr source, size_t bytes) {
    void *host = host_range(source, bytes);
    if (!host) return CUDA_ERROR_INVALID_VALUE;
    memcpy(destination, host, bytes); return CUDA_SUCCESS;
}
static CUdeviceptr dalloc(size_t bytes) {
    CHECK(allocation_count < sizeof allocations / sizeof allocations[0]);
    void *pointer = calloc(1, bytes); CHECK(pointer);
    allocations[allocation_count].pointer = pointer;
    allocations[allocation_count++].bytes = bytes;
    return (CUdeviceptr)(uintptr_t)pointer;
}

static int fail_directory_sync;
static int capture_test_fsync(int descriptor) {
    struct stat metadata;
    if (fail_directory_sync && !fstat(descriptor, &metadata) && S_ISDIR(metadata.st_mode)) {
        fail_directory_sync = 0; errno = EIO; return -1;
    }
    return fsync(descriptor);
}
#define fsync capture_test_fsync
#include "engine_capture.h"
#undef fsync

static void fill_capture(void) {
    for (int die = 0; die < nd; die++) for (int stage = 0; stage < CAP_STAGES; stage++) {
        float *values = host_range(capture_device[die][stage], (size_t)CAP_POSITIONS * CAP_BYTES);
        CHECK(values);
        for (int position = capture_resume; position < capture_total; position++)
            for (int element = 0; element < CAP_BYTES / 4; element++)
                values[(size_t)position * (CAP_BYTES / 4) + element] =
                    (float)(die * 100 + stage * 10 + position) + (float)(element % 16) / 16.0f;
    }
}

static char *read_file(const char *directory, const char *name, size_t *length) {
    char path[1024]; CHECK(snprintf(path, sizeof path, "%s/%s", directory, name) > 0);
    FILE *in = fopen(path, "rb"); CHECK(in);
    CHECK(!fseek(in, 0, SEEK_END)); long size = ftell(in); CHECK(size >= 0);
    CHECK(!fseek(in, 0, SEEK_SET));
    char *bytes = malloc((size_t)size + 1); CHECK(bytes);
    CHECK(fread(bytes, 1, (size_t)size, in) == (size_t)size);
    CHECK(!fclose(in)); bytes[size] = 0; *length = (size_t)size; return bytes;
}

int main(void) {
    unsetenv("FLASH_ENGINE_CAPTURE"); capture_init(); CHECK(capture_dir == -1);
    CHECK(!setenv("FLASH_ENGINE_CAPTURE", "bad", 1));
    in_init = 1;
    if (!setjmp(init_jb)) { capture_init(); CHECK(0); }
    capture_fini();
    CHECK(!setenv("FLASH_ENGINE_CAPTURE", "1", 1));
    unsetenv("GPUQ_RUN_DIR"); unsetenv("GPUQ_JOB_ID");
    if (!setjmp(init_jb)) { capture_init(); CHECK(0); }
    capture_fini();

    char directory[] = "/tmp/fs109-capture-test.XXXXXX"; CHECK(mkdtemp(directory));
    CHECK(!setenv("GPUQ_RUN_DIR", directory, 1));
    CHECK(!setenv("GPUQ_JOB_ID", "cpu-fixture-not-a-GPU-job", 1));
    capture_init(); in_init = 0; CHECK(capture_dir >= 0 && capture_host);
    CHECK(capture_begin(0, 0) == -1);
    CHECK(capture_begin(CAP_POSITIONS + 1, 0) == -1);
    CHECK(capture_begin(3, -1) == -1);
    CHECK(capture_begin(3, 3) == -1);
    CHECK(!capture_begin(3, 1));
    CHECK(capture_finish(2) == -1);

    prog_t program; prog_init(&program); program.loop = 1; program.token_offset = 1;
    capture_tap(&program, 0, CAP_EMBED, capture_device[0][0]);
    CHECK(program.n == 1 && program.p[0].w[0] == OP_XCOPY && program.p[0].w[1] == 10);
    CHECK(program.p[0].w[6] == 1 && program.p[0].w[31] == (1u << 6));
    prog_free(&program);

    fill_capture();
    CHECK(!capture_finish(3));
    char final[128], temporary[160];
    snprintf(final, sizeof final, "engine-capture.%ld.7.tsv", (long)getpid());
    snprintf(temporary, sizeof temporary, "%s.tmp", final);
    CHECK(faccessat(capture_dir, temporary, F_OK, 0) == -1 && errno == ENOENT);
    size_t before_length; char *before = read_file(directory, final, &before_length);
    CHECK(strstr(before, "resumed=1 total=3"));
    unsigned seen[MAXD][CAP_STAGES][3] = {{{0}}}, rows = 0;
    char *copy = strdup(before), *save; CHECK(copy);
    for (char *line = strtok_r(copy, "\n", &save); line; line = strtok_r(NULL, "\n", &save)) {
        if (*line == '#' || !strncmp(line, "position\t", 9)) continue;
        int position, die, used = 0; char stage_name[64], hash[17]; unsigned nonfinite;
        CHECK(sscanf(line, "%d\t%63[^\t]\t%d\t%16[0-9a-f]\t%u%n",
                     &position, stage_name, &die, hash, &nonfinite, &used) == 5);
        CHECK(line[used] == 0 && strlen(hash) == 16 && nonfinite == 0);
        CHECK(position >= 1 && position < 3 && die >= 0 && die < nd);
        int stage = 0; while (stage < CAP_STAGES && strcmp(stage_name, capture_names[stage])) stage++;
        CHECK(stage < CAP_STAGES && !seen[die][stage][position]++); rows++;
    }
    free(copy); CHECK(rows == MAXD * CAP_STAGES * 2);

    CHECK(capture_finish(3) == -1);
    size_t after_length; char *after = read_file(directory, final, &after_length);
    CHECK(before_length == after_length && !memcmp(before, after, before_length));
    free(before); free(after);
    CHECK(faccessat(capture_dir, temporary, F_OK, 0) == -1 && errno == ENOENT);
    trace_request = 8; CHECK(!capture_begin(3, 0));
    CHECK(capture_finish(3) == -1);
    snprintf(temporary, sizeof temporary, "engine-capture.%ld.8.tsv", (long)getpid());
    CHECK(faccessat(capture_dir, temporary, F_OK, 0) == -1 && errno == ENOENT);
    snprintf(temporary, sizeof temporary, "engine-capture.%ld.8.tsv.tmp", (long)getpid());
    CHECK(faccessat(capture_dir, temporary, F_OK, 0) == -1 && errno == ENOENT);
    trace_request = 9; fill_capture(); fail_directory_sync = 1;
    CHECK(capture_finish(3) == -1 && !fail_directory_sync);
    snprintf(temporary, sizeof temporary, "engine-capture.%ld.9.tsv", (long)getpid());
    CHECK(faccessat(capture_dir, temporary, F_OK, 0) == -1 && errno == ENOENT);
    snprintf(temporary, sizeof temporary, "engine-capture.%ld.9.tsv.tmp", (long)getpid());
    CHECK(faccessat(capture_dir, temporary, F_OK, 0) == -1 && errno == ENOENT);
    CHECK(!unlinkat(capture_dir, final, 0));
    capture_fini(); CHECK(capture_dir == -1 && !capture_host);
    for (size_t i = 0; i < allocation_count; i++) free(allocations[i].pointer);
    CHECK(!rmdir(directory));
    puts("engine_capture_test: bounds, resume coverage, publication, collision preservation, missing-write rejection, and post-link fsync rollback PASS");
    return 0;
}
