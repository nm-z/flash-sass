/* Production QSA instrumentation for model_test. The instruction-order twin
 * and discrete representations are compared bit for bit (-ffp-contract=off).
 * Per-token launches check every token/layer; loop launches retain the last
 * token's layer stages and compare every prediction. */
#ifndef MODEL_QSA_TEST_H
#define MODEL_QSA_TEST_H
#include "qsa.h"
#include "qsa_state.h"
static int model_qsa, model_host, model_embed_die = -1;
static unsigned model_budget, model_maxp;
static size_t model_kcf, model_vcf;
static dn_bufs_t model_qdev[MAXD][MAXB];
static model_qsa_t *model_qhost[MAXB];
static CUdeviceptr model_trace[MAXD][MAXB];
static float *model_href[MAXB];
static unsigned model_capture_last;
static int model_capture_full_scope;
static int model_qsa_capture_mode = -1;
static int model_qsa_capture_enabled(void) {
    if (model_qsa_capture_mode < 0) {
        const char *v = getenv("MODEL_CAPTURE_QSA");
        if (!v || !*v || !strcmp(v, "0")) model_qsa_capture_mode = 0;
        else if (!strcmp(v, "1")) model_qsa_capture_mode = 1;
        else { fprintf(stderr, "MODEL_CAPTURE_QSA must be 0 or 1\n"); exit(2); }
    }
    return model_qsa_capture_mode;
}
static void model_qsa_capture_emit(int die, int layer, unsigned t, CUdeviceptr device_ids) {
    int logical_die = attn_die_of(layer);
    if (logical_die != die || logical_die < 0 || logical_die >= nd) {
        fprintf(stderr, "MODEL_CAPTURE_QSA: invalid logical die for block %d\n", layer); exit(1);
    }
    size_t bytes = (size_t)model_budget * sizeof(uint32_t);
    uint32_t *ids = malloc(bytes);
    if (!ids) { fprintf(stderr, "MODEL_CAPTURE_QSA: selected-ID readback allocation failed\n"); exit(1); }
    CUresult r = cuCtxSetCurrent(ctx[logical_die]);
    if (r == CUDA_SUCCESS) r = cuMemcpyDtoH(ids, device_ids, bytes);
    if (r != CUDA_SUCCESS) {
        const char *s = "?"; cuGetErrorString(r, &s);
        fprintf(stderr, "MODEL_CAPTURE_QSA: GPU selected-ID readback failed: %s\n", s);
        free(ids); exit(1);
    }
    if (printf("{\"schema\":\"flash-sass.qsa-selected-blocks.v1\",\"source\":\"actual_model_gpu_readback\",\"visible_length\":%u,\"layer\":%d,\"logical_die\":%d,\"physical_die\":%d,\"FLASH_NBLK\":%u,\"selected_block_ids\":[",
               t + 1, layer, logical_die, ar[logical_die].die, model_budget) < 0) {
        fprintf(stderr, "MODEL_CAPTURE_QSA: JSON output failed\n"); free(ids); exit(1);
    }
    for (unsigned i = 0; i < model_budget; i++) {
        if (printf("%s%u", i ? "," : "", (unsigned)ids[i]) < 0) {
            fprintf(stderr, "MODEL_CAPTURE_QSA: JSON output failed\n"); free(ids); exit(1);
        }
    }
    free(ids);
    if (fputs("],\"full_model_scope\":true,\"selected_ids_match_scalar\":true}\n", stdout) == EOF || fflush(stdout) == EOF) {
        fprintf(stderr, "MODEL_CAPTURE_QSA: JSON output failed\n"); exit(1);
    }
}
static CUdeviceptr model_kv_alloc(size_t n) {
    if (!model_host) return dalloc(n);
    void *p; CUdeviceptr dev;
    CK(cuMemHostAlloc(&p, n, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP));
    memset(p, 0, n); CK(cuMemHostGetDevicePointer(&dev, p, 0)); return dev;
}
static void model_qsa_host_alloc(int layer) {
    model_qsa_t *q = calloc(1, sizeof *q); model_qhost[layer] = q;
    q->nblk = model_budget;
    q->kq = calloc((size_t)model_maxp * 512, 1); q->vq = calloc((size_t)model_maxp * 512, 1);
    q->ks = calloc((size_t)model_maxp * 16, 2); q->vs = calloc((size_t)model_maxp * 16, 2);
    q->pool = calloc((size_t)model_maxp / 4 * 128, 2); q->pkt = calloc((size_t)model_maxp / 128 * 4096, 2);
    q->score = calloc((size_t)model_maxp / 4 + 64, 4); q->sel = calloc(model_budget + 8, 4);
    q->kc = calloc(model_kcf, 4); q->vc = calloc(model_vcf, 4);
    if (!q->kq || !q->vq || !q->ks || !q->vs || !q->pool || !q->pkt || !q->score || !q->sel || !q->kc || !q->vc) { fprintf(stderr, "scalar QSA allocation failed\n"); exit(1); }
}
static void model_qsa_dev_alloc(int die, int layer) {
    dn_bufs_t *q = &model_qdev[die][layer];
    q->qsa = 1; q->nblk = model_budget;
    q->kq = model_kv_alloc((size_t)model_maxp * 512); q->vq = model_kv_alloc((size_t)model_maxp * 512);
    q->ks = model_kv_alloc((size_t)model_maxp * 32); q->vs = model_kv_alloc((size_t)model_maxp * 32);
    q->pkt = dalloc((size_t)model_maxp / 128 * 8192); q->ring = dalloc(2048);
    q->iqk = dalloc(4 * 640); q->iq = dalloc(4 * 512); q->sc = dalloc(4 * ((size_t)model_maxp / 4 + 64));
    q->hst = dalloc(4 * 1280); q->sel = dalloc(4 * (model_budget + 8));
    q->kc = dalloc(4 * model_kcf); q->vc = dalloc(4 * model_vcf);
    q->qg = dalloc(4 * 12288); q->k = dalloc(4 * 512); q->v = dalloc(4 * 512); q->qo = dalloc(4 * 6144); q->aout = dalloc(4 * 6144);
    q->pm = dalloc(4 * 24 * 8); q->pd = dalloc(4 * 24 * 8); q->po = dalloc(4 * 24 * 8 * 256);
}
static void model_qsa_bind(dn_bufs_t *b, int die, int layer) {
    dn_bufs_t *q = &model_qdev[die][layer];
    qsa_bind_layer(b, model_budget, q->kq, q->ks, q->vq, q->vs, q->pkt, q->ring);
    if (!q->kq) return;
    b->iqk=q->iqk; b->iq=q->iq; b->sc=q->sc; b->hst=q->hst; b->sel=q->sel;
    b->kc=q->kc; b->vc=q->vc; b->qg=q->qg; b->k=q->k; b->v=q->v; b->qo=q->qo; b->aout=q->aout;
    b->pm=q->pm; b->pd=q->pd; b->po=q->po;
}
static void model_layer_trace(prog_t *P, int die, int layer, CUdeviceptr S) {
    phase_t *p = prog_add(P, OP_XCOPY, 10); ph_ptr(p, 0x140, S); ph_ptr(p, 0x148, model_trace[die][layer]);
}
static unsigned model_param(const phase_t *p, int addr) { return p->w[2 + (addr - 0x140) / 4]; }
static void model_assert_program(const prog_t *P, int die, int b0, int b1) {
    unsigned count[DIE_NOPS] = {0}, layers = 0; static int printed[MAXD];
    for (int b = b0; b < b1; b++) if (BLK_IS_ATTN(b) && die == attn_die_of(b)) layers++;
    for (int i = 0; i < P->n; i++) {
        const phase_t *p = &P->p[i]; unsigned op = p->w[0]; if (op < DIE_NOPS) count[op]++;
        int addr = op == OP_IDXSCORE ? 0x164 : op == OP_IDXSEL ? 0x15c : op == OP_KVGATHER ? 0x17c : op == OP_ATTNPARTN ? 0x174 : 0;
        if (addr && model_param(p, addr) != model_budget) { fprintf(stderr, "QSA budget mismatch in %s\n", die_opname[op]); exit(3); }
        if (op == OP_KVGATHER && p->w[1] != model_budget + 1) { fprintf(stderr, "QSA gather grid/budget mismatch\n"); exit(3); }
    }
    if (count[OP_ATTNQK] || count[OP_ATTNPART] || count[OP_ATTNQKQ] != layers || count[OP_IDXQK] != layers || count[OP_IDXSCORE] != layers || count[OP_IDXSEL] != 5 * layers || count[OP_KVGATHER] != layers || count[OP_ATTNPARTN] != layers || count[OP_ATTNMERGE] != layers) {
        fprintf(stderr, "die %d: incomplete production QSA program (%u attention layers)\n", die, layers); exit(3);
    }
    if (!printed[die]++) printf("  die %d asserted QSA opcodes and FLASH_NBLK=%u for %u layer(s)\n", die, model_budget, layers);
}
static int model_qsa_compare(int die, int layer, unsigned t) {
    model_qsa_t *h = model_qhost[layer]; dn_bufs_t *q = &model_qdev[die][layer]; int bad = 0;
    unsigned nv = (t + 1) / 4, n = qsa_nkeys(t, model_budget);
    int capture = model_qsa_capture_enabled();
#define MC(name, ref, dev, bytes, unit) do { char label[80]; snprintf(label, sizeof label, "b%d.%s", layer, name); bad += cmpbuf(label, ref, dev, bytes, unit); } while (0)
    MC("kq", h->kq + (size_t)t * 512, q->kq + (size_t)t * 512, 512, 1);
    MC("vq", h->vq + (size_t)t * 512, q->vq + (size_t)t * 512, 512, 1);
    MC("ks", h->ks + (size_t)t * 16, q->ks + (size_t)t * 32, 32, 2);
    MC("vs", h->vs + (size_t)t * 16, q->vs + (size_t)t * 32, 32, 2);
    MC("iqk", h->iqk, q->iqk, sizeof h->iqk, 4); MC("iq", h->iq, q->iq, sizeof h->iq, 4);
    MC("ring", h->ring, q->ring, 4 * 128 * ((t & 3) + 1), 4);
    if ((t & 3) == 3) {
        unsigned b = t / 4; uint16_t got[128];
        for (int d = 0; d < 128; d++) {
            size_t ix = (size_t)(b >> 5) * 4096 + (size_t)(d >> 2) * 128 + (b & 31) * 4 + (d & 3);
            CK(cuMemcpyDtoH(got + d, q->pkt + ix * 2, 2));
        }
        if (memcmp(got, h->pool + (size_t)b * 128, sizeof got)) { fprintf(stderr, "b%d pool MISMATCH\n", layer); bad++; }
    }
    if (nv > model_budget) {
        MC("scores", h->score, q->sc, 4 * (size_t)nv, 4);
        char label[80]; snprintf(label, sizeof label, "b%d.selected_ids", layer);
        int selection_bad = cmpbuf_control(label, h->sel, q->sel, 4 * model_budget, 4, "selection");
        bad += selection_bad;
    }
    float *kg = malloc(4 * model_kcf), *vg = malloc(4 * (size_t)n * 512);
    if (!kg || !vg) { fprintf(stderr, "gather comparison allocation failed\n"); exit(1); }
    CK(cuMemcpyDtoH(kg, q->kc, 4 * model_kcf)); CK(cuMemcpyDtoH(vg, q->vc, 4 * (size_t)n * 512));
    int gather_bad = 0;
    for (unsigned slot = 0; slot < n; slot++) for (int kv = 0; kv < 2; kv++) for (int d = 0; d < 256; d++) {
        size_t ki = attn_kidx(slot, kv, d), vi = (size_t)slot * 512 + kv * 256 + d;
        if (memcmp(kg + ki, h->kc + ki, 4) || memcmp(vg + vi, h->vc + vi, 4)) gather_bad++;
    }
    free(kg); free(vg); if (gather_bad) { fprintf(stderr, "b%d gathered K/V MISMATCH (%d)\n", layer, gather_bad); bad++; }
    MC("qo", h->qo, q->qo, sizeof h->qo, 4); MC("pm", h->pm, q->pm, sizeof h->pm, 4); MC("pd", h->pd, q->pd, sizeof h->pd, 4);
    MC("po", h->po, q->po, sizeof h->po, 4); MC("aout", h->aout, q->aout, sizeof h->aout, 4);
#undef MC
    if (capture && model_capture_full_scope && nv > model_budget && t == model_capture_last && bad == 0)
        model_qsa_capture_emit(die, layer, t, q->sel);
    return bad;
}
static int model_compare_recurrent(int die, int layer, const split_t *s,
                                   const hstate_t *h, CUdeviceptr dn,
                                   CUdeviceptr conv, CUdeviceptr ple) {
    int bad = 0;
    char label[64];
    if (!BLK_IS_ATTN(layer)) {
        snprintf(label, sizeof label, "b%d.dn_state.d%d", layer, die);
        bad += cmpbuf_control(label, h->dn_state + (size_t)s->h0 * 16384,
                              dn, (size_t)s->hn * 16384 * sizeof(float), 4, "dn");
        unsigned nloc = SPLIT_NLOC(s);
        float *expected = malloc((size_t)4 * nloc * sizeof(float));
        if (!expected) { fprintf(stderr, "convolution comparison allocation failed\n"); exit(1); }
        /* Local slots contain wrapped key-head Q, then K, then contiguous V.
         * The scalar convolution ring contains all 10240 channels. */
        for (unsigned slot = 0; slot < 4; slot++) {
            for (int j = 0; j < 2 * s->nk + s->hn; j++) {
                unsigned src;
                if (j < s->nk) src = 128u * (unsigned)((s->klo + j) % 16);
                else if (j < 2 * s->nk) src = 2048u + 128u * (unsigned)((s->klo + j - s->nk) % 16);
                else src = 4096u + 128u * (unsigned)(s->h0 + j - 2 * s->nk);
                memcpy(expected + (size_t)slot * nloc + 128u * (unsigned)j,
                       h->conv_state + (size_t)slot * 10240 + src, 128 * sizeof(float));
            }
        }
        snprintf(label, sizeof label, "b%d.conv_state.d%d", layer, die);
        bad += cmpbuf_control(label, expected, conv, (size_t)4 * nloc * sizeof(float), 4, "conv");
        free(expected);
    }
    if (layer == 1) {
        snprintf(label, sizeof label, "b1.ple_state.d%d", die);
        bad += cmpbuf_control(label, h->ple_state, ple, 10 * 10240 * sizeof(float), 4, "ple");
    }
    return bad;
}
static int model_compare_layers(int b0, int b1, unsigned t, int head,
                                const dn_hbufs_t *hb, const dn_bufs_t *db,
                                split_t sp[MAXD][MAXB], const hstate_t hs[MAXB],
                                CUdeviceptr dn[MAXD][MAXB], CUdeviceptr conv[MAXD][MAXB],
                                const CUdeviceptr ple[MAXD]) {
    int bad = 0;
    for (int d = 0; d < nd; d++) {
        CK(cuCtxSetCurrent(ctx[d]));
        for (int b = b0; b < b1; b++) {
            char label[64]; snprintf(label, sizeof label, "b%d.S.d%d", b, d);
            bad += cmpbuf_control(label, model_href[b], model_trace[d][b], 4 * 10240, 4, "layer");
            if (BLK_IS_ATTN(b) && d == attn_die_of(b)) bad += model_qsa_compare(d, b, t);
            bad += model_compare_recurrent(d, b, &sp[d][b], &hs[b], dn[d][b], conv[d][b], ple[d]);
        }
        if (head) {
            char label[32]; snprintf(label, sizeof label, "logits.d%d", d);
            bad += cmpbuf_control(label, hb->logits + sp[d][b0].o0, db[d].hlo, 4u * (unsigned)sp[d][b0].on, 4, "logits");
        }
    }
    return bad;
}
#endif
