/* Isolated issue109 diagnostic. XCOPY taps add synchronization and can mask a
 * race; a passing instrumented request does not clear the production failure.
 * Include from engine.c after its device/context and request identity globals. */
#ifndef ISSUE109_ENGINE_CAPTURE_H
#define ISSUE109_ENGINE_CAPTURE_H

enum {
    CAP_EMBED, CAP_BLOCK0, CAP_PLE_KN, CAP_PLE_QN,
    CAP_PLE_G, CAP_PLE_X, CAP_PLE_STREAM, CAP_BLOCK1, CAP_STAGES
};
enum { CAP_POSITIONS = 16, CAP_BYTES = 40960 };
static const char *const capture_names[CAP_STAGES] = {
    "embed.S", "b0.S", "ple.kn", "ple.qn", "ple.pG", "ple.px", "ple.S", "b1.S"
};
static CUdeviceptr capture_device[MAXD][CAP_STAGES];
static unsigned char *capture_host;
static int capture_dir = -1, capture_resume, capture_total;

static void capture_error(const char *message) {
    fprintf(stderr, "engine capture: %s: %s\n", message, strerror(errno));
    if (in_init) longjmp(init_jb, 1);
    if (in_run) longjmp(run_jb, 1);
    exit(1);
}

static void capture_fini(void) {
    if (capture_dir >= 0) close(capture_dir);
    capture_dir = -1;
    free(capture_host); capture_host = NULL;
    memset(capture_device, 0, sizeof capture_device);
    capture_resume = capture_total = 0;
}

static void capture_init(void) {
    const char *option = getenv("FLASH_ENGINE_CAPTURE");
    if (!option || !strcmp(option, "0")) return;
    if (strcmp(option, "1")) {
        errno = EINVAL; capture_error("FLASH_ENGINE_CAPTURE must be 0 or 1");
    }
    const char *directory = getenv("GPUQ_RUN_DIR");
    const char *job = getenv("GPUQ_JOB_ID");
    if (!directory || !*directory || !job || !*job) {
        errno = EINVAL; capture_error("capture requires GPUQ_RUN_DIR and GPUQ_JOB_ID");
    }
    capture_dir = open(directory, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (capture_dir < 0) capture_error("open GPUQ capture directory");
    capture_host = malloc((size_t)CAP_POSITIONS * CAP_BYTES);
    if (!capture_host) capture_error("allocate host capture buffer");
    for (int d = 0; d < nd; d++) {
        CK(cuCtxSetCurrent(ctx[d]));
        for (int stage = 0; stage < CAP_STAGES; stage++)
            capture_device[d][stage] = dalloc((size_t)CAP_POSITIONS * CAP_BYTES);
    }
    CK(cuCtxSetCurrent(ctx[0]));
    fprintf(stderr, "engine capture: eight stage taps, at most %d positions; instrumented scheduling\n", CAP_POSITIONS);
}

static int capture_begin(int total, int resume) {
    if (capture_dir < 0) return 0;
    if (total < 1 || total > CAP_POSITIONS || resume < 0 || resume >= total) {
        fprintf(stderr, "engine capture: request exceeds the 1..%d position diagnostic scope\n", CAP_POSITIONS);
        return -1;
    }
    capture_total = total; capture_resume = resume;
    for (int d = 0; d < nd; d++) {
        CK(cuCtxSetCurrent(ctx[d]));
        for (int stage = 0; stage < CAP_STAGES; stage++)
            CK(cuMemsetD8(capture_device[d][stage], 0xa5,
                          (size_t)CAP_POSITIONS * CAP_BYTES));
    }
    CK(cuCtxSetCurrent(ctx[0]));
    return 0;
}

static void capture_tap(prog_t *program, int die, int stage, CUdeviceptr source) {
    if (capture_dir < 0) return;
    phase_t *phase = prog_add(program, OP_XCOPY, 10);
    ph_ptr(phase, 0x140, source);
    ph_ptr(phase, 0x148, capture_device[die][stage]);
    /* The isolated XCOPY variant adds 40960 * position to its destination.
     * Ordinary embedding-copy descriptors leave this parameter zero. */
    ph_tokv(program, phase, 0x150, 0, 0);
}

static uint64_t capture_hash(const unsigned char *bytes) {
    uint64_t hash = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < CAP_BYTES; i++) {
        hash ^= bytes[i]; hash *= 0x100000001b3ull;
    }
    return hash;
}

static int capture_finish_error(const char *message) {
    fprintf(stderr, "engine capture: %s: %s\n", message, strerror(errno));
    return -1;
}

static int capture_finish(int completed) {
    if (capture_dir < 0) return 0;
    if (completed != capture_total) {
        errno = EINVAL; return capture_finish_error("incomplete request has no accepted stage capture");
    }
    char temporary[160], final[128];
    snprintf(final, sizeof final, "engine-capture.%ld.%llu.tsv", (long)getpid(),
             (unsigned long long)trace_request);
    snprintf(temporary, sizeof temporary, "%s.tmp", final);
    int fd = openat(capture_dir, temporary, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) return capture_finish_error("create stage capture");
    FILE *out = fdopen(fd, "w");
    if (!out) { int saved = errno; close(fd); unlinkat(capture_dir, temporary, 0); errno = saved; return capture_finish_error("open stage capture stream"); }
    int failed = fprintf(out,
        "# issue109 stage fingerprints v1; additional XCOPY synchronization; not a production pass\n"
        "# pid=%ld request=%llu revision=%s cubin_fnv=%016llx kv_host=%d paired=%d resumed=%d total=%d\n"
        "position\tstage\tdie\tfnv64\tnonfinite\n",
        (long)getpid(), (unsigned long long)trace_request, FLASH_REV,
        (unsigned long long)cubin_fnv, kv_host, two_position,
        capture_resume, capture_total) < 0;
    for (int d = 0; d < nd && !failed; d++) {
        CUresult context_result = cuCtxSetCurrent(ctx[d]);
        if (context_result != CUDA_SUCCESS) {
            fclose(out); unlinkat(capture_dir, temporary, 0); CK(context_result);
        }
        for (int stage = 0; stage < CAP_STAGES; stage++) {
            CUresult result = cuMemcpyDtoH(capture_host, capture_device[d][stage],
                                          (size_t)capture_total * CAP_BYTES);
            if (result != CUDA_SUCCESS) {
                fclose(out); unlinkat(capture_dir, temporary, 0); CK(result);
            }
            for (int position = capture_resume; position < capture_total; position++) {
                const unsigned char *bytes = capture_host + (size_t)position * CAP_BYTES;
                unsigned nonfinite = 0, changed = 0;
                for (size_t i = 0; i < CAP_BYTES; i++) changed |= bytes[i] ^ 0xa5u;
                if (!changed) {
                    fprintf(stderr, "engine capture: unwritten stage die=%d position=%d stage=%s\n",
                            d, position, capture_names[stage]);
                    failed = 1;
                }
                for (size_t i = 0; i < CAP_BYTES; i += sizeof(float)) {
                    float value; memcpy(&value, bytes + i, sizeof value);
                    nonfinite += !isfinite(value);
                }
                if (fprintf(out, "%d\t%s\t%d\t%016llx\t%u\n", position,
                            capture_names[stage], d,
                            (unsigned long long)capture_hash(bytes), nonfinite) < 0)
                    failed = 1;
            }
        }
    }
    CUresult context_result = cuCtxSetCurrent(ctx[0]);
    if (context_result != CUDA_SUCCESS) {
        fclose(out); unlinkat(capture_dir, temporary, 0); CK(context_result);
    }
    if (fflush(out) || fsync(fileno(out))) failed = 1;
    if (fclose(out)) failed = 1;
    int published = 0;
    if (!failed) {
        if (linkat(capture_dir, temporary, capture_dir, final, 0)) failed = 1;
        else {
            published = 1;
            if (unlinkat(capture_dir, temporary, 0) || fsync(capture_dir)) failed = 1;
        }
    }
    if (failed) {
        int saved = errno ? errno : EIO;
        unlinkat(capture_dir, temporary, 0);
        if (published) { unlinkat(capture_dir, final, 0); fsync(capture_dir); }
        errno = saved;
        return capture_finish_error("publish complete stage capture");
    }
    return 0;
}
#endif
