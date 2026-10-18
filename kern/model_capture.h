#ifndef FLASH_MODEL_CAPTURE_H
#define FLASH_MODEL_CAPTURE_H
#include <errno.h>
#include <float.h>
#include <sys/stat.h>
#include <unistd.h>
#ifndef FLASH_REV
#define FLASH_REV "unknown"
#endif

static FILE *model_capture_logits;
static unsigned model_capture_next;
static int model_capture_enabled(void) {
    const char *p = getenv("MODEL_CAPTURE");
    return p && *p;
}
static void model_capture_fail(const char *message) {
    fprintf(stderr, "MODEL_CAPTURE: %s\n", message);
    exit(2);
}
static void model_capture_check(int b0, int b1, int embed, int head, int loop) {
    if (!model_capture_enabled()) return;
    if (!model_qsa || b0 != 0 || b1 != 48 || !embed || !head || loop ||
        !getenv("MODEL_IDS") || !getenv("MODEL_PLE") ||
        getenv("MODEL_NPH") || getenv("MODEL_INJECT"))
        model_capture_fail("needs full blocks 0..48, embed+head, per-token QSA, "
                           "complete MODEL_IDS, real MODEL_PLE, and no truncation/injection");
}
static void model_capture_path(char *path, size_t size, const char *leaf) {
    int n = snprintf(path, size, "%s/%s", getenv("MODEL_CAPTURE"), leaf);
    if (n < 0 || (size_t)n >= size) model_capture_fail("path too long");
}
static FILE *model_capture_open(const char *leaf) {
    char path[1200];
    model_capture_path(path, sizeof(path), leaf);
    FILE *f = fopen(path, "wx");
    if (!f) { perror(path); model_capture_fail("cannot create capture file"); }
    return f;
}
static void model_capture_close(FILE *f) {
    /* Always close even if a preceding flush reports a failure. */
    int failed = fflush(f);
    if (fclose(f)) failed = 1;
    if (failed) model_capture_fail("cannot close capture file");
}
static void model_capture_complete(unsigned T) {
    FILE *f = model_capture_open("capture_info.pending.json");
    if (fprintf(f,
        "{\n  \"schema\":\"flash-sass.raw-model-capture.v2\",\n"
        "  \"complete\":true,\n  \"dtype\":\"<f4\",\n"
        "  \"teacher_forced\":true,\n  \"positions\":%u,\n"
        "  \"vocab_size\":248320,\n  \"byte_order\":\"little\",\n"
        "  \"blocks\":[0,48],\n  \"attention_mode\":\"qsa\",\n"
        "  \"FLASH_NBLK\":%u,\n  \"FLASH_KV_HOST\":%d,\n  \"dies\":%d,\n"
        "  \"build_revision\":\"%s\",\n"
        "  \"logits_source\":\"actual GPU per-die head row slices\",\n"
        "  \"checkpoint_provenance\":null,\n  \"reference_provenance\":null,\n"
        "  \"quality_validation\":false\n}\n",
        T, model_budget, model_host, nd, FLASH_REV) < 0)
        model_capture_fail("metadata write failed");
    model_capture_close(f);
    char pending[1200], complete[1200];
    model_capture_path(pending, sizeof(pending), "capture_info.pending.json");
    model_capture_path(complete, sizeof(complete), "capture_info.json");
    if (link(pending, complete) || unlink(pending))
        model_capture_fail("cannot publish completion marker");
}
static void model_capture_record(unsigned t, unsigned T, const uint32_t *ids,
                                  const dn_bufs_t *buf, split_t sp[MAXD][MAXB]) {
    if (!model_capture_enabled()) return;
    if (!T || t >= T || t != model_capture_next)
        model_capture_fail("capture positions must be contiguous in [0,T)");
    if (!t) {
        uint32_t endian = 1;
        if (*(const unsigned char *)&endian != 1 || sizeof(float) != 4 ||
            FLT_RADIX != 2 || FLT_MANT_DIG != 24)
            model_capture_fail("needs little-endian IEEE FP32 host");
        unsigned char *covered = calloc(248320, 1);
        if (!covered) model_capture_fail("coverage allocation failed");
        for (int d = 0; d < nd; d++) {
            int start = sp[d][0].o0, count = sp[d][0].on;
            if (start < 0 || count < 0 || start > 248320 || count > 248320 - start)
                model_capture_fail("head slice out of bounds");
            for (int j = start; j < start + count; j++)
                if (covered[j]++) model_capture_fail("head slices overlap");
        }
        for (int j = 0; j < 248320; j++)
            if (!covered[j]) model_capture_fail("head slices miss a vocabulary ID");
        free(covered);
        if (mkdir(getenv("MODEL_CAPTURE"), 0700)) {
            perror(getenv("MODEL_CAPTURE"));
            model_capture_fail("output must be a new directory");
        }
        FILE *f = model_capture_open("ids.u32");
        if (fwrite(ids, 4, T, f) != T) model_capture_fail("IDs write failed");
        model_capture_close(f);
        model_capture_logits = model_capture_open("logits.f32");
    }
    float *row = malloc(4 * 248320);
    if (!row) model_capture_fail("logit allocation failed");
    for (int d = 0; d < nd; d++) {
        CK(cuCtxSetCurrent(ctx[d]));
        CK(cuMemcpyDtoH(row + sp[d][0].o0, buf[d].hlo, 4u * (unsigned)sp[d][0].on));
    }
    if (!model_capture_logits ||
        fwrite(row, 4, 248320, model_capture_logits) != 248320 ||
        fflush(model_capture_logits))
        model_capture_fail("logit write failed");
    free(row);
    model_capture_next++;
    if (t + 1 == T) {
        model_capture_close(model_capture_logits);
        model_capture_logits = NULL;
        /* An interrupted run never acquires this completion marker. */
        model_capture_complete(T);
        printf("captured %u teacher-forced GPU logit rows; "
               "checkpoint/reference provenance still required\n", T);
    }
}
#endif
