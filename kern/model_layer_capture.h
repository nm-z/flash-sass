/* Optional post-projection QSA fixtures from model_test's completed GPU token.
 * No kernel instructions or numerical inputs are changed. A complete.json marker
 * is written only after all prefix streams have been flushed and closed.
 * These are candidate fixtures, not independent checkpoint evidence. */
#ifndef FLASH_MODEL_LAYER_CAPTURE_H
#define FLASH_MODEL_LAYER_CAPTURE_H
#include <errno.h>
#include <limits.h>
#include <sys/stat.h>
#include <unistd.h>

static int mlc_layer = -1;
static unsigned mlc_lengths[32], mlc_nlengths;
static FILE *mlc_k, *mlc_v, *mlc_ik;

static int model_layer_capture_enabled(void) {
    const char *p = getenv("MODEL_LAYER_CAPTURE");
    return p && *p;
}
static void mlc_fail(const char *why) {
    fprintf(stderr, "MODEL_LAYER_CAPTURE: %s\n", why);
    exit(2);
}
static unsigned mlc_uint(const char *s, char **end) {
    if (!s || *s < '0' || *s > '9') mlc_fail("expected an unsigned integer");
    errno = 0;
    unsigned long n = strtoul(s, end, 10);
    if (errno || n > UINT_MAX) mlc_fail("integer out of range");
    return (unsigned)n;
}
static void mlc_add_length(unsigned n, unsigned T) {
    if (!n || n > T) mlc_fail("capture length must be in [1,T]");
    for (unsigned i = 0; i < mlc_nlengths; i++)
        if (mlc_lengths[i] == n) mlc_fail("duplicate capture length");
    if (mlc_nlengths == sizeof mlc_lengths / sizeof *mlc_lengths)
        mlc_fail("too many capture lengths");
    mlc_lengths[mlc_nlengths++] = n;
}
static void model_layer_capture_check(int b0, int b1, int embed, int head,
                                      int loop, unsigned T) {
    if (!model_layer_capture_enabled()) return;
    if (!model_qsa || b0 != 0 || b1 != 48 || !embed || !head || loop ||
        !getenv("MODEL_IDS") || !getenv("MODEL_PLE") ||
        getenv("MODEL_NPH") || getenv("MODEL_INJECT"))
        mlc_fail("requires full 48-layer teacher-forced per-token QSA with real PLE and no truncation/injection");
    const char *s = getenv("MODEL_LAYER_INDEX");
    if (!s) mlc_fail("MODEL_LAYER_INDEX must name one attention layer");
    char *end;
    unsigned layer = mlc_uint(s, &end);
    if (*end || layer >= 48 || !BLK_IS_ATTN((int)layer))
        mlc_fail("MODEL_LAYER_INDEX must name one attention layer");
    mlc_layer = (int)layer;
    s = getenv("MODEL_LAYER_LENGTHS");
    if (!s || !*s) mlc_fail("MODEL_LAYER_LENGTHS must list the predeclared snapshot lengths");
    while (*s) {
        unsigned n = mlc_uint(s, &end);
        mlc_add_length(n, T);
        if (!*end) break;
        if (*end != ',' || !end[1]) mlc_fail("invalid capture length list");
        s = end + 1;
    }
    unsigned endian = 1;
    if (*(const unsigned char *)&endian != 1 || sizeof(float) != 4)
        mlc_fail("requires little-endian FP32 host representation");
}
static void mlc_path(char *path, size_t cap, const char *leaf) {
    int n = snprintf(path, cap, "%s/%s", getenv("MODEL_LAYER_CAPTURE"), leaf);
    if (n < 0 || (size_t)n >= cap) mlc_fail("path too long");
}
static FILE *mlc_open(const char *leaf) {
    char path[1600]; mlc_path(path, sizeof path, leaf);
    FILE *f = fopen(path, "wx");
    if (!f) { perror(path); mlc_fail("cannot create fixture file"); }
    return f;
}
static void mlc_close(FILE *f) {
    if (fflush(f) || fclose(f)) mlc_fail("cannot close fixture file");
}
static void mlc_write(FILE *f, const void *p, size_t bytes) {
    if (bytes && fwrite(p, 1, bytes, f) != bytes) mlc_fail("fixture write failed");
}
static void mlc_host_file(const char *leaf, const void *p, size_t bytes) {
    FILE *f = mlc_open(leaf); mlc_write(f, p, bytes); mlc_close(f);
}
static void mlc_device_to_file(FILE *f, CUdeviceptr ptr, size_t bytes) {
    unsigned char data[65536];
    while (bytes) {
        size_t take = bytes < sizeof data ? bytes : sizeof data;
        CK(cuMemcpyDtoH(data, ptr, take));
        mlc_write(f, data, take);
        ptr += take; bytes -= take;
    }
}
static void mlc_device_file(const char *leaf, CUdeviceptr ptr, size_t bytes) {
    FILE *f = mlc_open(leaf); mlc_device_to_file(f, ptr, bytes); mlc_close(f);
}
static void mlc_snapshot_file(unsigned N, const char *name, CUdeviceptr ptr,
                              size_t bytes) {
    char leaf[128];
    int n = snprintf(leaf, sizeof leaf, "n%u/%s", N, name);
    if (n < 0 || (size_t)n >= sizeof leaf) mlc_fail("snapshot name too long");
    mlc_device_file(leaf, ptr, bytes);
}
static void model_layer_capture_record(unsigned t, unsigned T, const uint32_t *ids,
                                       const dn_bufs_t *db,
                                       dn_weights_t dw[MAXD][MAXB]) {
    if (!model_layer_capture_enabled()) return;
    int die = attn_die_of(mlc_layer);
    if (die < 0 || die >= nd) mlc_fail("attention owner outside model dies");
    const dn_bufs_t *q = &model_qdev[die][mlc_layer];
    const dn_weights_t *w = &dw[die][mlc_layer];
    CK(cuCtxSetCurrent(ctx[die]));
    if (!t) {
        if (mkdir(getenv("MODEL_LAYER_CAPTURE"), 0700)) {
            perror(getenv("MODEL_LAYER_CAPTURE")); mlc_fail("output must be a new directory");
        }
        mlc_host_file("ids.u32", ids, (size_t)T * 4);
        mlc_device_file("qn.f32", w->attn_q_norm, 256 * 4);
        mlc_device_file("kn.f32", w->attn_k_norm, 256 * 4);
        mlc_device_file("iqn.f32", w->indexer_qn, 128 * 4);
        mlc_device_file("ikn.f32", w->indexer_kn, 128 * 4);
        mlc_device_file("rope_cs.f32", db[die].rope_cs, (size_t)T * 32 * 4);
        mlc_device_file("rope_sn.f32", db[die].rope_sn, (size_t)T * 32 * 4);
        mlc_k = mlc_open("raw_k.f32");
        mlc_v = mlc_open("raw_v.f32");
        mlc_ik = mlc_open("raw_ik.f32");
    }
    mlc_device_to_file(mlc_k, q->k, 512 * 4);
    mlc_device_to_file(mlc_v, q->v, 512 * 4);
    mlc_device_to_file(mlc_ik, q->iqk + 512 * 4, 128 * 4);
    unsigned N = t + 1, selected = 0;
    for (unsigned i = 0; i < mlc_nlengths; i++)
        if (mlc_lengths[i] == N) selected = 1;
    if (selected) {
        char leaf[128], path[1600];
        snprintf(leaf, sizeof leaf, "n%u", N);
        mlc_path(path, sizeof path, leaf);
        if (mkdir(path, 0700)) mlc_fail("cannot create snapshot directory");
        unsigned nv = N / 4, ns = nv < model_budget ? nv : model_budget;
        unsigned nkeys = ns * 4 + N % 4;
        mlc_snapshot_file(N, "qg.f32", q->qg, 12288 * 4);
        mlc_snapshot_file(N, "iqk.f32", q->iqk, 640 * 4);
        mlc_snapshot_file(N, "iq.f32", q->iq, 512 * 4);
        mlc_snapshot_file(N, "qo.f32", q->qo, 6144 * 4);
        mlc_snapshot_file(N, "aout.f32", q->aout, 6144 * 4);
        mlc_snapshot_file(N, "layer_output.f32", model_trace[die][mlc_layer], 10240 * 4);
        /* Store the full valid candidate INT8 prefix for isolated dequantization. */
        mlc_snapshot_file(N, "kq.i8", q->kq, (size_t)N * 512);
        mlc_snapshot_file(N, "vq.i8", q->vq, (size_t)N * 512);
        mlc_snapshot_file(N, "ks.f16", q->ks, (size_t)N * 16 * 2);
        mlc_snapshot_file(N, "vs.f16", q->vs, (size_t)N * 16 * 2);
        size_t tiled = ((size_t)nv + 31) / 32 * 4096;
        uint16_t *pkt = malloc((tiled ? tiled : 1) * 2);
        uint16_t *pool = malloc(((size_t)nv * 128 + 1) * 2);
        if (!pkt || !pool) mlc_fail("pool allocation failed");
        if (tiled) CK(cuMemcpyDtoH(pkt, q->pkt, tiled * 2));
        for (unsigned b = 0; b < nv; b++) for (unsigned d = 0; d < 128; d++) {
            size_t ix = (size_t)(b >> 5) * 4096 + (d >> 2) * 128 +
                        (b & 31) * 4 + (d & 3);
            pool[(size_t)b * 128 + d] = pkt[ix];
        }
        snprintf(leaf, sizeof leaf, "n%u/pool.f16", N);
        mlc_host_file(leaf, pool, (size_t)nv * 128 * 2);
        free(pkt); free(pool);
        if (nv > model_budget) {
            mlc_snapshot_file(N, "scores.f32", q->sc, (size_t)nv * 4);
            mlc_snapshot_file(N, "selected_ids.u32", q->sel, (size_t)ns * 4);
        } else {
            uint32_t *all = malloc(((size_t)ns + 1) * 4);
            if (!all) mlc_fail("selection allocation failed");
            for (unsigned i = 0; i < ns; i++) all[i] = i;
            snprintf(leaf, sizeof leaf, "n%u/selected_ids.u32", N);
            mlc_host_file(leaf, all, (size_t)ns * 4); free(all);
        }
        snprintf(leaf, sizeof leaf, "n%u/snapshot.json", N);
        FILE *f = mlc_open(leaf);
        if (fprintf(f, "{\"schema\":\"flash-sass.qsa-layer-snapshot.v1\","
                       "\"length\":%u,\"position\":%u,\"layer\":%d,\"die\":%d,"
                       "\"budget\":%u,\"complete_blocks\":%u,\"tail\":%u,"
                       "\"attended_keys\":%u,\"sparse_exclusion\":%s,"
                       "\"scores_source\":\"%s\",\"selected_ids_source\":\"%s\","
                       "\"quality_validation\":false}\n",
                    N, t, mlc_layer, die, model_budget, nv, N % 4, nkeys,
                    nv > model_budget ? "true" : "false",
                    nv > model_budget ? "gpu" : "not_dispatched",
                    nv > model_budget ? "gpu" : "implicit_all_blocks") < 0)
            mlc_fail("snapshot metadata write failed");
        mlc_close(f);
    }
    if (N == T) {
        mlc_close(mlc_k); mlc_close(mlc_v); mlc_close(mlc_ik);
        mlc_k = mlc_v = mlc_ik = NULL;
        FILE *f = mlc_open("complete.json.tmp");
        if (fprintf(f, "{\"schema\":\"flash-sass.qsa-layer-capture.v1\","
                       "\"positions\":%u,\"layer\":%d,\"die\":%d,\"budget\":%u,"
                       "\"epsilon\":%.17g,\"rope_base\":10000000,\"FLASH_KV_HOST\":%d,"
                       "\"byte_order\":\"little\",\"teacher_forced\":true,"
                       "\"build_revision\":\"%s\","
                       "\"input_boundary\":\"post_projection_before_norm_rope_and_kv_quantization\","
                       "\"gamma\":\"supplied_multiplicative_f32\","
                       "\"pool_dtype\":\"fp16\",\"rope_tables\":\"supplied_f32\","
                       "\"upstream_activation_quantization_isolated\":false,"
                       "\"independent_checkpoint_evidence\":false,\"quality_validation\":false,"
                       "\"snapshot_lengths\":[", T, mlc_layer, die, model_budget,
                    (double)1e-6f, model_host, FLASH_REV) < 0)
            mlc_fail("completion metadata write failed");
        for (unsigned i = 0; i < mlc_nlengths; i++)
            if (fprintf(f, "%s%u", i ? "," : "", mlc_lengths[i]) < 0)
                mlc_fail("completion metadata write failed");
        if (fputs("]}\n", f) == EOF) mlc_fail("completion metadata write failed");
        mlc_close(f);
        char pending[1600], complete[1600];
        mlc_path(pending, sizeof pending, "complete.json.tmp");
        mlc_path(complete, sizeof complete, "complete.json");
        if (link(pending, complete) || unlink(pending))
            mlc_fail("cannot publish completion marker");
        printf("captured QSA layer %d projection prefixes and %u snapshots; no independent checkpoint attestation\n",
               mlc_layer, mlc_nlengths);
    }
}
#endif
