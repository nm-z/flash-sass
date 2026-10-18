/* Native numerical oracle. Reference reductions below are independent scalar
 * binary64 arithmetic; pinned Transformers outputs are checked separately.
 * This does not execute Transformers and is not checkpoint-quality evidence. */
#include "attn.h"
#include "fmath.h"
#include "qsa.h"
#include <errno.h>
#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#ifdef QSA_ORACLE_TARGET
#include "qsa_oracle_gpu.h"
#endif
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#ifdef __FAST_MATH__
#error strict floating point is required
#endif
int qsaoracle_environment(void);
int qsaoracle_set_rounding(int);
int qsaoracle_index(const float *, unsigned, const float *, const float *,
                    const float *, const float *, const float *, float *,
                    uint16_t *, float *);
int qsaoracle_quant(const float *, unsigned, int8_t *, uint16_t *, float *);
int qsaoracle_attention(const float *, const float *, const float *, unsigned,
                        unsigned, const float *, float *, float *, float *,
                        float *);
/* The candidate backend never supplies the independent reference arithmetic. */
static int gpu_mode;
static const char *capture_dir;
static unsigned long gpu_launches_(void) {
#ifdef QSA_ORACLE_TARGET
  if (gpu_mode)
    return qsaoracle_gpu_launch_count();
#endif
  return 0;
}
static int candidate_index(const float *raw, unsigned n, const float *query,
                           const float *qn, const float *kn, const float *cs,
                           const float *sn, float *iq, uint16_t *pool,
                           float *score) {
#ifdef QSA_ORACLE_TARGET
  if (gpu_mode)
    return qsaoracle_gpu_index(raw, n, query, qn, kn, cs, sn, iq, pool, score);
#endif
  return qsaoracle_index(raw, n, query, qn, kn, cs, sn, iq, pool, score);
}
static unsigned candidate_select(const float *scores, unsigned nv,
                                 unsigned nblk, uint32_t *ids) {
#ifdef QSA_ORACLE_TARGET
  if (gpu_mode)
    return qsaoracle_gpu_select(scores, nv, nblk, ids);
#endif
  return idx_select_gpu(scores, nv, nblk, ids);
}
static unsigned candidate_scores_select(const float *iq, const uint16_t *pool,
                                        unsigned nv, unsigned nblk,
                                        float *scores, uint32_t *ids) {
#ifdef QSA_ORACLE_TARGET
  if (gpu_mode)
    return qsaoracle_gpu_scores_select(iq, pool, nv, nblk, scores, ids);
#endif
  idx_scores_gpu(iq, pool, nv, scores);
  return idx_select_gpu(scores, nv, nblk, ids);
}
static int candidate_attention(const float *q, const float *k, const float *v,
                               unsigned n, unsigned cap, const float *g,
                               float *out, float *pm, float *pd, float *po) {
#ifdef QSA_ORACLE_TARGET
  if (gpu_mode)
    return qsaoracle_gpu_attention(q, k, v, n, cap, g, out, pm, pd, po);
#endif
  return qsaoracle_attention(q, k, v, n, cap, g, out, pm, pd, po);
}
#define qsaoracle_index candidate_index
#define idx_select_gpu candidate_select
#define qsaoracle_attention candidate_attention

static unsigned failures, metric_count;
static void die(const char *s) {
  fprintf(stderr, "qsa_oracle: %s\n", s);
  exit(2);
}
static void check(int ok, const char *s) {
  if (!ok) {
    fprintf(stderr, "FAIL: %s\n", s);
    failures++;
  }
}
static void *alloc_(size_t n, size_t s) {
  if (s && n > SIZE_MAX / s)
    die("allocation overflow");
  void *p = calloc(n ? n : 1, s);
  if (!p)
    die("allocation");
  return p;
}
static unsigned count_(const char *s) {
  char *e;
  errno = 0;
  unsigned long n = strtoul(s, &e, 10);
  if (errno || !*s || *e || n < 1 || n > 262003)
    die("length must be 1..262003");
  return (unsigned)n;
}
static void path_(char *p, size_t cap, const char *dir, const char *name) {
  int n = snprintf(p, cap, "%s/%s", dir, name);
  if (n < 0 || (size_t)n >= cap)
    die("path too long");
}
static void *read_(const char *dir, const char *name, size_t n, size_t s) {
  char p[4096];
  path_(p, sizeof p, dir, name);
  FILE *f = fopen(p, "rb");
  if (!f) {
    fprintf(stderr, "%s: ", p);
    die("open input");
  }
  void *x = alloc_(n, s);
  if (fread(x, s, n, f) != n || fgetc(f) != EOF || ferror(f))
    die("input shape/byte mismatch");
  if (fclose(f))
    die("close input");
  return x;
}
static void save_(const char *dir, const char *name, const void *x, size_t n,
                  size_t s) {
  char p[4096];
  path_(p, sizeof p, dir, name);
  FILE *f = fopen(p, "wbx");
  if (!f)
    die("output must be new");
  if (fwrite(x, s, n, f) != n || fclose(f))
    die("write output");
}
static uint32_t bits(float x) {
  uint32_t u;
  memcpy(&u, &x, 4);
  return u;
}
static float unbits(uint32_t u) {
  float x;
  memcpy(&x, &u, 4);
  return x;
}
static float bf16(float x) {
  uint32_t u = bits(x);
  if ((u & 0x7fffffff) > 0x7f800000)
    return unbits(u | 0x00400000);
  u += 0x7fffu + ((u >> 16) & 1u);
  return unbits(u & 0xffff0000u);
}
/* Independent arithmetic FP16 oracle; no candidate conversion routines used. */
static uint16_t half_(float x) {
  unsigned sign = signbit(x) ? 0x8000u : 0;
  double a = fabs((double)x);
  if (isnan(x))
    return sign | 0x7e00u;
  if (isinf(x) || a >= 65520.)
    return sign | 0x7c00u;
  if (a < 0x1p-14)
    return (uint16_t)(sign | (unsigned)nearbyint(ldexp(a, 24)));
  int exp;
  frexp(a, &exp);
  int e = exp - 1;
  unsigned m = (unsigned)nearbyint((ldexp(a, -e) - 1.) * 1024.);
  if (m == 1024) {
    m = 0;
    e++;
  }
  return (uint16_t)(sign | ((unsigned)(e + 15) << 10) | m);
}
static float unhalf_(uint16_t h) {
  double a;
  unsigned e = (h >> 10) & 31, m = h & 1023;
  if (e == 31)
    a = m ? NAN : INFINITY;
  else
    a = e ? ldexp(1. + m / 1024., (int)e - 15) : ldexp((double)m, -24);
  return (float)((h & 0x8000) ? -a : a);
}
typedef struct {
  double max_abs, max_relative, rms, nrms, max_ref;
} Metric;
static Metric measure(const void *rp, int rd, const void *cp, int cd,
                      size_t n) {
  if (!n)
    die("empty metric");
  Metric m = {0};
  double se = 0, sr = 0;
  for (size_t i = 0; i < n; i++) {
    double r = rd ? ((const double *)rp)[i] : ((const float *)rp)[i],
           c = cd ? ((const double *)cp)[i] : ((const float *)cp)[i];
    if (!isfinite(r) || !isfinite(c))
      die("nonfinite comparison");
    double d = c - r, a = fabs(d);
    se += d * d;
    sr += r * r;
    m.max_abs = fmax(m.max_abs, a);
    m.max_relative = fmax(m.max_relative, a / fmax(fabs(r), 1e-8));
    m.max_ref = fmax(m.max_ref, fabs(r));
  }
  m.rms = sqrt(se / n);
  m.nrms = m.rms / fmax(sqrt(sr / n), 1e-8);
  return m;
}
static int score_ok(Metric m) { return isfinite(m.nrms) && m.nrms <= 0.005; }
static void record(const char *name, Metric m, double abs_limit,
                   double nrms_limit, int enforce) {
  int ok = (!enforce) || (m.max_abs <= abs_limit && m.nrms <= nrms_limit);
  if (enforce)
    check(ok, name);
  printf("%s{\"name\":\"%s\",\"max_abs\":%.17g,\"max_relative\":%.17g,\"rms\":%"
         ".17g,\"normalized_rms\":%.17g,\"enforced\":%s,\"pass\":%s}",
         metric_count++ ? "," : "", name, m.max_abs, m.max_relative, m.rms,
         m.nrms, enforce ? "true" : "false", ok ? "true" : "false");
}
static void begin_(const char *type, unsigned n) {
  failures = 0;
  metric_count = 0;
  printf("{\"type\":\"%s\",\"length\":%u,\"metrics\":[", type, n);
}
static int end_(void) {
  printf("],\"failures\":%u,\"status\":\"%s\",\"candidate_backend\":\"%s\","
         "\"gpu_launches\":%lu,\"gpu_executed\":%s,\"checkpoint_conformance\":"
         "false}\n",
         failures, failures ? "FAIL_NATIVE" : "PASS_NATIVE",
         gpu_mode ? "SM52" : "CPU_SCALAR", gpu_launches_(),
         gpu_launches_() ? "true" : "false");
  return failures ? 3 : 0;
}
static int fcmp(const void *a, const void *b) {
  float x = *(const float *)a, y = *(const float *)b;
  return (x > y) - (x < y);
}
static double margin_(const float *s, unsigned n) {
  if (n <= 512)
    return 0;
  float *x = alloc_(n, 4);
  memcpy(x, s, n * 4);
  qsort(x, n, 4, fcmp);
  double d = (double)x[n - 512] - x[n - 513];
  free(x);
  return d;
}
static int admissible(const float *s, unsigned n, const uint32_t *ids,
                      unsigned k) {
  if (k > n)
    return 0;
  if (!k)
    return 1;
  float *x = alloc_(n, 4);
  unsigned char *seen = alloc_(n, 1);
  memcpy(x, s, n * 4);
  qsort(x, n, 4, fcmp);
  float cut = x[n - k];
  int ok = 1;
  for (unsigned j = 0; j < k; j++) {
    unsigned id = ids[j];
    if (id >= n || seen[id] || s[id] < cut) {
      ok = 0;
      break;
    }
    seen[id] = 1;
  }
  for (unsigned i = 0; i < n; i++)
    if (s[i] > cut && !seen[i])
      ok = 0;
  free(x);
  free(seen);
  return ok;
}
static int lower_policy(const float *s, unsigned n, const uint32_t *ids,
                        unsigned k) {
  if (!admissible(s, n, ids, k))
    return 0;
  for (unsigned j = 0; j < k; j++) {
    unsigned id = ids[j], rank = 0;
    if (j && ids[j - 1] >= id)
      return 0;
    for (unsigned i = 0; i < n; i++)
      rank += s[i] > s[id] || (s[i] == s[id] && i < id);
    if (rank >= k)
      return 0;
  }
  return 1;
}
static unsigned overlap_(const uint32_t *a, const uint32_t *b, unsigned k) {
  unsigned m = 0;
  for (unsigned i = 0; i < k; i++)
    for (unsigned j = 0; j < k; j++)
      m += a[i] == b[j];
  return m;
}
/* Mathematical reductions in binary64, independent of candidate fmaf chains,
 * warp butterflies, exp polynomial, tiled softmax and FP16 conversion. */
static void attention_ref(const float *q, const float *k, const float *v,
                          unsigned n, const float *g, double *out) {
  double *s = alloc_(n, sizeof(double));
  for (unsigned h = 0; h < 24; h++) {
    unsigned kv = h / 12;
    double max = -INFINITY, den = 0;
    for (unsigned j = 0; j < n; j++) {
      double d = 0;
      for (unsigned z = 0; z < 256; z++)
        d += (double)q[h * 256 + z] * k[(size_t)j * 512 + kv * 256 + z];
      s[j] = d / 16.;
      max = fmax(max, s[j]);
    }
    for (unsigned j = 0; j < n; j++) {
      s[j] = exp(s[j] - max);
      den += s[j];
    }
    for (unsigned z = 0; z < 256; z++) {
      double num = 0;
      for (unsigned j = 0; j < n; j++)
        num += s[j] * (double)v[(size_t)j * 512 + kv * 256 + z];
      out[h * 256 + z] = (num / den) / (1. + exp(-(double)g[h * 256 + z]));
    }
  }
  free(s);
}
static void quant_(const float *x, size_t n, float *out, int8_t *q,
                   uint16_t *s) {
  if (n % 32 || n / 32 > UINT32_MAX ||
      qsaoracle_quant(x, (unsigned)(n / 32), q, s, out))
    die("quantization domain");
}
static void local_attn(const float *q, const float *k, const float *v,
                       unsigned n, const float *g, float *out, float *pm,
                       float *pd, float *po) {
  unsigned cap = ((n + 8191) / 8192) * 8192;
  float *kp = alloc_((size_t)cap * 512, 4), *vp = alloc_((size_t)cap * 512, 4);
  memcpy(kp, k, (size_t)n * 512 * 4);
  memcpy(vp, v, (size_t)n * 512 * 4);
  if (qsaoracle_attention(q, kp, vp, n, cap, g, out, pm, pd, po))
    die("attention domain");
  if (n < cap) {
    vp[(size_t)n * 512] = NAN;
    check(qsaoracle_attention(q, kp, vp, n, cap, g, out, pm, pd, po) != 0,
          "reject nonfinite V padding");
    vp[(size_t)n * 512] = 1.f;
    check(qsaoracle_attention(q, kp, vp, n, cap, g, out, pm, pd, po) != 0,
          "reject nonzero V padding");
    vp[(size_t)n * 512] = 0;
    kp[(size_t)n * 512] = NAN;
    check(qsaoracle_attention(q, kp, vp, n, cap, g, out, pm, pd, po) != 0,
          "reject nonfinite K padding");
  }
  free(kp);
  free(vp);
}
static uint64_t rng_state;
static uint32_t random_(void) {
  rng_state ^= rng_state >> 12;
  rng_state ^= rng_state << 25;
  rng_state ^= rng_state >> 27;
  return (uint32_t)((rng_state * UINT64_C(2685821657736338717)) >> 40);
}
static float uniform_(float scale) {
  return ((float)random_() * 0x1p-23f - 1.f) * scale;
}
static void rope_(unsigned n, float *cs, float *sn) {
  for (unsigned t = 0; t < n; t++)
    for (unsigned d = 0; d < 32; d++) {
      double a = t * pow(1e7, -2. * d / 64.);
      cs[(size_t)t * 32 + d] = (float)cos(a);
      sn[(size_t)t * 32 + d] = (float)sin(a);
    }
}
/* Pinned HF boundaries: FP32 mean -> raw dtype -> RMS in FP32 -> (1+offset)
 * -> raw dtype -> 64-dim RoPE. Binary64 RMS reduction is the mathematical
 * oracle, deliberately independent of both backend reduction trees. */
static void norm_ref(const float *x, const float *gamma, const float *cs,
                     const float *sn, int bf, float *y) {
  double ss = 0;
  for (unsigned d = 0; d < 128; d++)
    ss += (double)x[d] * x[d];
  float r = (float)(1. / sqrt(ss / 128. + (double)1e-6f));
  for (unsigned d = 0; d < 128; d++) {
    float z = (x[d] * r) * gamma[d];
    y[d] = bf ? bf16(z) : z;
  }
  for (unsigned d = 0; d < 32; d++) {
    float a = y[d], b = y[d + 32];
    float ac = a * cs[d], bs = b * sn[d], bc = b * cs[d], as = a * sn[d];
    if (bf) {
      ac = bf16(ac);
      bs = bf16(bs);
      bc = bf16(bc);
      as = bf16(as);
    }
    y[d] = bf ? bf16(ac - bs) : ac - bs;
    y[d + 32] = bf ? bf16(bc + as) : bc + as;
  }
}
static void scores_ref(const float *iq, const float *pool, unsigned nv,
                       float *score) {
  for (unsigned b = 0; b < nv; b++) {
    float s = 0;
    for (unsigned h = 0; h < 4; h++) {
      double dot = 0;
      for (unsigned d = 0; d < 128; d++)
        dot += (double)iq[h * 128 + d] * pool[(size_t)b * 128 + d];
      float a = (float)dot;
      if (a < 0)
        a = 0;
      s += a;
    }
    score[b] = s / 11.313708305358887f;
  }
}
typedef struct {
  float score;
  uint32_t id;
} Ranked;
static int rankcmp(const void *a, const void *b) {
  const Ranked *x = a, *y = b;
  if (x->score != y->score)
    return x->score > y->score ? -1 : 1;
  return (x->id > y->id) - (x->id < y->id);
}
static int ucmp(const void *a, const void *b) {
  uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
  return (x > y) - (x < y);
}
static void select_ref(const float *s, unsigned n, uint32_t *ids) {
  unsigned k = n < 512 ? n : 512;
  Ranked *x = alloc_(n, sizeof *x);
  for (unsigned i = 0; i < n; i++) {
    x[i].score = s[i];
    x[i].id = i;
  }
  qsort(x, n, sizeof *x, rankcmp);
  for (unsigned j = 0; j < k; j++)
    ids[j] = x[j].id;
  qsort(ids, k, 4, ucmp);
  free(x);
}
static void index_eval(unsigned nv, const float *iq, const uint16_t *pool,
                       const float *riq, const float *rpool,
                       const float *reference, const uint32_t *reference_ids,
                       const float *frozen_raw, const char *outdir,
                       uint32_t *ids) {
  if (!outdir)
    outdir = capture_dir;
  unsigned k = nv < 512 ? nv : 512;
  float *raw = alloc_(nv, 4), *scaled = alloc_(nv, 4), *ideal = alloc_(nv, 4),
        *hp = alloc_((size_t)nv * 128, 4), *pool_only = alloc_(nv, 4);
  unsigned selected = candidate_scores_select(iq, pool, nv, 512, raw, ids);
  if (selected != k)
    die("candidate score or selection execution failed");
  for (unsigned b = 0; b < nv; b++)
    scaled[b] = raw[b] / 11.313708305358887f;
  if (frozen_raw)
    check(!memcmp(raw, frozen_raw, (size_t)nv * 4),
          "current scores reproduce archived candidate bits");
  check(selected == k, "candidate selection count");
  check(lower_policy(raw, nv, ids, k),
        "candidate deterministic lower-ID selection");
  check(admissible(reference, nv, reference_ids, k),
        "reference cutoff admissibility");
  if (nv) {
    scores_ref(riq, rpool, nv, ideal);
    if (frozen_raw)
      record("independent_math_vs_frozen_HF_scores",
             measure(reference, 0, ideal, 0, nv), 3e-5, 3e-5, 1);
    Metric m = measure(reference, 0, scaled, 0, nv);
    record("scaled_score_error", m, DBL_MAX, .005, 1);
    check(score_ok(m), "score normalized RMS <= 0.005");
    record("query_normalization", measure(riq, 0, iq, 0, 512), 1e-5, DBL_MAX,
           1);
    for (size_t i = 0; i < (size_t)nv * 128; i++)
      hp[i] = unhalf_(half_(rpool[i]));
    scores_ref(riq, hp, nv, pool_only);
    record("pooled_key_quantization_only",
           measure(reference, 0, pool_only, 0, nv), DBL_MAX, DBL_MAX, 0);
    record("after_same_pool_quantization", measure(pool_only, 0, scaled, 0, nv),
           DBL_MAX, DBL_MAX, 0);
    if (nv > 512) {
      float *sorted = alloc_(nv, 4);
      memcpy(sorted, reference, (size_t)nv * 4);
      qsort(sorted, nv, 4, fcmp);
      float cut = sorted[nv - 512];
      unsigned char *selected = alloc_(nv, 1);
      for (unsigned j = 0; j < k; j++) {
        selected[ids[j]] = 1;
        check(reference[ids[j]] >= cut - 2 * m.max_abs,
              "selected ID lies within measured cutoff interval");
      }
      for (unsigned b = 0; b < nv; b++)
        if (reference[b] > cut + 2 * m.max_abs)
          check(selected[b], "mandatory separated ID selected");
      if (margin_(reference, nv) > 2 * m.max_abs)
        check(overlap_(ids, reference_ids, k) == k,
              "stable-cutoff selection set equality");
      free(sorted);
      free(selected);
    }
  }
  if (outdir) {
    save_(outdir, "raw_score.f32", raw, nv, 4);
    save_(outdir, "scaled_score.f32", scaled, nv, 4);
    save_(outdir, "reference_score.f32", reference, nv, 4);
    save_(outdir, "math_score.f32", ideal, nv, 4);
    save_(outdir, "pool_only_score.f32", pool_only, nv, 4);
    save_(outdir, "local_ids.u32", ids, k, 4);
    save_(outdir, "reference_ids.u32", reference_ids, k, 4);
    char p[4096];
    path_(p, sizeof p, outdir, "selection.json");
    FILE *f = fopen(p, "wx");
    if (!f)
      die("new selection metadata");
    fprintf(f,
            "{\"blocks\":%u,\"selected\":%u,\"overlap\":%u,\"raw_margin\":%."
            "17g,\"reference_margin\":%.17g}\n",
            nv, k, overlap_(ids, reference_ids, k), margin_(raw, nv),
            margin_(reference, nv));
    if (fclose(f))
      die("selection metadata write");
  }
  free(raw);
  free(scaled);
  free(ideal);
  free(hp);
  free(pool_only);
}
static void attention_eval(const float *q, const float *k, const float *v,
                           unsigned n, const float *g, const double *gold,
                           const double *qgold, const char *outdir) {
  if (!outdir)
    outdir = capture_dir;
  size_t nk = (size_t)n * 512;
  double *ref = alloc_(6144, 8), *qref = alloc_(6144, 8);
  float *out = alloc_(6144, 4), *qo = alloc_(6144, 4), *kd = alloc_(nk, 4),
        *vd = alloc_(nk, 4), *pm = alloc_(192, 4), *pd = alloc_(192, 4),
        *po = alloc_(49152, 4);
  int8_t *kq = alloc_(nk, 1), *vq = alloc_(nk, 1);
  uint16_t *ks = alloc_(nk / 32, 2), *vs = alloc_(nk / 32, 2);
  attention_ref(q, k, v, n, g, ref);
  if (gold)
    record("independent_attention_vs_frozen_external",
           measure(gold, 1, ref, 1, 6144), 1e-12, 1e-12, 1);
  local_attn(q, k, v, n, g, out, pm, pd, po);
  record("same_ID_attention_reduction", measure(ref, 1, out, 0, 6144), 3e-5,
         3e-5, 1);
  if (outdir) {
    save_(outdir, "pm.f32", pm, 192, 4);
    save_(outdir, "pd.f32", pd, 192, 4);
    save_(outdir, "po.f32", po, 49152, 4);
  }
  quant_(k, nk, kd, kq, ks);
  quant_(v, nk, vd, vq, vs);
  attention_ref(q, kd, vd, n, g, qref);
  if (qgold)
    record("independent_quant_attention_vs_frozen_external",
           measure(qgold, 1, qref, 1, 6144), 1e-12, 1e-12, 1);
  local_attn(q, kd, vd, n, g, qo, pm, pd, po);
  record("same_ID_KV_quantization", measure(ref, 1, qref, 1, 6144), .025, .025,
         1);
  record("quantized_attention_reduction", measure(qref, 1, qo, 0, 6144), 3e-5,
         3e-5, 1);
  if (outdir) {
    save_(outdir, "query.f32", q, 6144, 4);
    save_(outdir, "key.f32", k, nk, 4);
    save_(outdir, "value.f32", v, nk, 4);
    save_(outdir, "gate.f32", g, 6144, 4);
    save_(outdir, "reference.f64", ref, 6144, 8);
    save_(outdir, "candidate.f32", out, 6144, 4);
    save_(outdir, "quant_reference.f64", qref, 6144, 8);
    save_(outdir, "quant_candidate.f32", qo, 6144, 4);
    save_(outdir, "kq.i8", kq, nk, 1);
    save_(outdir, "vq.i8", vq, nk, 1);
    save_(outdir, "ks.u16", ks, nk / 32, 2);
    save_(outdir, "vs.u16", vs, nk / 32, 2);
    save_(outdir, "quant_pm.f32", pm, 192, 4);
    save_(outdir, "quant_pd.f32", pd, 192, 4);
    save_(outdir, "quant_po.f32", po, 49152, 4);
  }
  free(ref);
  free(qref);
  free(out);
  free(qo);
  free(kd);
  free(vd);
  free(pm);
  free(pd);
  free(po);
  free(kq);
  free(vq);
  free(ks);
  free(vs);
}
static unsigned token_ids_(unsigned n, const uint32_t *blocks,
                           uint32_t *tokens) {
  unsigned nv = n / 4, k = nv < 512 ? nv : 512, z = 0;
  for (unsigned j = 0; j < k; j++)
    for (unsigned t = 0; t < 4; t++)
      tokens[z++] = blocks[j] * 4 + t;
  for (unsigned t = nv * 4; t < n; t++)
    tokens[z++] = t;
  return z;
}
static void indexed_kv_(const uint32_t *ids, unsigned n, float *k, float *v) {
  for (unsigned j = 0; j < n; j++)
    for (unsigned d = 0; d < 512; d++) {
      k[(size_t)j * 512 + d] = (float)(sin(ids[j] * .013 + d * .071) * .3);
      v[(size_t)j * 512 + d] = (float)(cos(ids[j] * .021 + d * .047) * .5);
    }
}
static void sparse_eval(unsigned n, const uint32_t *local,
                        const uint32_t *remote, const float *q, const float *g,
                        const double *gold, const char *outdir) {
  if (!outdir)
    outdir = capture_dir;
  uint32_t lt[2051], rt[2051];
  unsigned nk = token_ids_(n, local, lt);
  check(token_ids_(n, remote, rt) == nk, "same complete-block and tail counts");
  size_t elems = (size_t)nk * 512;
  float *lk = alloc_(elems, 4), *lv = alloc_(elems, 4), *rk = alloc_(elems, 4),
        *rv = alloc_(elems, 4), *kd = alloc_(elems, 4), *vd = alloc_(elems, 4),
        *out = alloc_(6144, 4), *pm = alloc_(192, 4), *pd = alloc_(192, 4),
        *po = alloc_(49152, 4);
  int8_t *codes = alloc_(elems, 1);
  uint16_t *scales = alloc_(elems / 32, 2);
  double *re = alloc_(6144, 8), *lo = alloc_(6144, 8), *qu = alloc_(6144, 8);
  indexed_kv_(lt, nk, lk, lv);
  indexed_kv_(rt, nk, rk, rv);
  attention_ref(q, rk, rv, nk, g, re);
  attention_ref(q, lk, lv, nk, g, lo);
  if (gold)
    record("sparse_reference_vs_frozen_external", measure(gold, 1, re, 1, 6144),
           1e-12, 1e-12, 1);
  quant_(lk, elems, kd, codes, scales);
  quant_(lv, elems, vd, codes, scales);
  attention_ref(q, kd, vd, nk, g, qu);
  local_attn(q, kd, vd, nk, g, out, pm, pd, po);
  record("selection_only", measure(re, 1, lo, 1, 6144), DBL_MAX, DBL_MAX, 0);
  record("same_selection_KV", measure(lo, 1, qu, 1, 6144), .025, .025, 1);
  record("sparse_reduction", measure(qu, 1, out, 0, 6144), 3e-5, 3e-5, 1);
  if (outdir) {
    save_(outdir, "sparse_local_token_ids.u32", lt, nk, 4);
    save_(outdir, "sparse_reference_token_ids.u32", rt, nk, 4);
    save_(outdir, "sparse_query.f32", q, 6144, 4);
    save_(outdir, "sparse_gate.f32", g, 6144, 4);
    save_(outdir, "sparse_reference_selection.f64", re, 6144, 8);
    save_(outdir, "sparse_local_selection.f64", lo, 6144, 8);
    save_(outdir, "sparse_quant_reference.f64", qu, 6144, 8);
    save_(outdir, "sparse_candidate.f32", out, 6144, 4);
    save_(outdir, "sparse_pm.f32", pm, 192, 4);
    save_(outdir, "sparse_pd.f32", pd, 192, 4);
    save_(outdir, "sparse_po.f32", po, 49152, 4);
  }
  free(lk);
  free(lv);
  free(rk);
  free(rv);
  free(kd);
  free(vd);
  free(out);
  free(pm);
  free(pd);
  free(po);
  free(codes);
  free(scales);
  free(re);
  free(lo);
  free(qu);
}
static int generated_index(const char *dir, unsigned n) {
  if (mkdir(dir, 0755))
    die("new generated-index directory required");
  begin_("generated_index_and_sparse_attention", n);
  unsigned nv = n / 4;
  size_t rawlen = (size_t)n * 128;
  float *raw = alloc_(rawlen, 4), *cs = alloc_((size_t)n * 32, 4),
        *sn = alloc_((size_t)n * 32, 4), iq[512], riq[512], query[512], qn[128],
        kn[128], oq[128], ok[128];
  uint16_t *pool = alloc_((size_t)nv * 128, 2);
  float *rpool = alloc_((size_t)nv * 128, 4), *scores = alloc_(nv, 4),
        *unused = alloc_(nv, 4);
  uint32_t ids[512], rids[512];
  rng_state = 100000 + n;
  for (size_t i = 0; i < rawlen; i++)
    raw[i] = uniform_(.5f);
  for (unsigned i = 0; i < 512; i++)
    query[i] = uniform_(.5f);
  for (unsigned i = 0; i < 128; i++) {
    oq[i] = bf16(uniform_(.1f));
    ok[i] = bf16(uniform_(.1f));
    qn[i] = 1.f + oq[i];
    kn[i] = 1.f + ok[i];
  }
  rope_(n, cs, sn);
  if (qsaoracle_index(raw, n, query, qn, kn, cs, sn, iq, pool, unused))
    die("generated index domain");
  for (unsigned h = 0; h < 4; h++)
    norm_ref(query + h * 128, qn, cs + (size_t)(n - 1) * 32,
             sn + (size_t)(n - 1) * 32, 0, riq + h * 128);
  for (unsigned b = 0; b < nv; b++) {
    float mean[128];
    for (unsigned d = 0; d < 128; d++) {
      double sum = 0;
      for (unsigned t = 0; t < 4; t++)
        sum += raw[((size_t)b * 4 + t) * 128 + d];
      mean[d] = (float)(sum * .25);
    }
    norm_ref(mean, kn, cs + (size_t)b * 128, sn + (size_t)b * 128, 0,
             rpool + (size_t)b * 128);
  }
  scores_ref(riq, rpool, nv, scores);
  select_ref(scores, nv, rids);
  index_eval(nv, iq, pool, riq, rpool, scores, rids, NULL, dir, ids);
  save_(dir, "raw_keys.f32", raw, rawlen, 4);
  save_(dir, "index_query.f32", query, 512, 4);
  save_(dir, "q_gamma.f32", qn, 128, 4);
  save_(dir, "k_gamma.f32", kn, 128, 4);
  save_(dir, "stored_q_offset.f32", oq, 128, 4);
  save_(dir, "stored_k_offset.f32", ok, 128, 4);
  save_(dir, "local_iq.f32", iq, 512, 4);
  save_(dir, "reference_iq.f32", riq, 512, 4);
  save_(dir, "local_pool.u16", pool, (size_t)nv * 128, 2);
  save_(dir, "reference_pool.f32", rpool, (size_t)nv * 128, 4);
  float *q = alloc_(6144, 4), *g = alloc_(6144, 4);
  rng_state = 200000 + n;
  for (unsigned i = 0; i < 6144; i++) {
    q[i] = uniform_(.3f);
    g[i] = uniform_(1.f);
  }
  sparse_eval(n, ids, rids, q, g, NULL, dir);
  free(q);
  free(g);
  free(raw);
  free(cs);
  free(sn);
  free(pool);
  free(rpool);
  free(scores);
  free(unused);
  return end_();
}
static int generated_attention(const char *dir, unsigned n) {
  if (mkdir(dir, 0755))
    die("new generated-attention directory required");
  begin_("generated_attention", n);
  size_t nk = (size_t)n * 512;
  float *q = alloc_(6144, 4), *g = alloc_(6144, 4), *k = alloc_(nk, 4),
        *v = alloc_(nk, 4);
  rng_state = 300000 + n;
  for (unsigned i = 0; i < 6144; i++) {
    q[i] = uniform_(.3f);
    g[i] = uniform_(2.f);
  }
  for (size_t i = 0; i < nk; i++) {
    k[i] = uniform_(.3f);
    v[i] = uniform_(.5f);
  }
  for (unsigned j = 0; j < n; j += 1024) {
    k[(size_t)j * 512] += 4.f * (j / 1024 + 1);
    k[(size_t)j * 512 + 256] += 4.f * (j / 1024 + 1);
  }
  attention_eval(q, k, v, n, g, NULL, NULL, dir);
  free(q);
  free(g);
  free(k);
  free(v);
  return end_();
}
static int self_test(int negative) {
  begin_(negative ? "negative_score_control" : "domain_and_arithmetic", 4);
  double r[4] = {1, 0, 0, 0}, bad[4] = {1, .004, .004, .004};
  Metric m = measure(r, 1, bad, 1, 4);
  if (negative) {
    record("score_gate_must_reject", m, DBL_MAX, .005, 1);
    return end_();
  }
  check(m.max_abs <= .005 * m.max_ref,
        "negative control passes obsolete maximum-score gate");
  check(!score_ok(m), "true score NRMS rejects negative control");
  record("old_gate_blind_spot", m, DBL_MAX, DBL_MAX, 0);
  double good[4] = {1, .002, .002, .002};
  check(score_ok(measure(r, 1, good, 1, 4)),
        "score NRMS accepts within-threshold control");
  check(qsaoracle_set_rounding(0) == 0, "set FE_DOWNWARD");
  check(!qsaoracle_environment(), "rounding guard rejects FE_DOWNWARD");
  check(qsaoracle_set_rounding(1) == 0 && qsaoracle_environment(),
        "restore FE_TONEAREST");
  const float values[] = {0.f,     -0.f,    0x1p-24f, 0x1p-25f, 0x1p-14f,
                          65504.f, 65520.f, INFINITY, -INFINITY};
  for (size_t i = 0; i < sizeof values / sizeof values[0]; i++)
    check(qsa_f2h(values[i]) == half_(values[i]), "independent FP16 boundary");
  check(qsa_f2h(nextafterf(0x1p-25f, INFINITY)) == 1,
        "adjacent zero-to-nonzero FP16 transition");
  const float amps[] = {0.f, 1e-10f, 1e-6f, 1.f, 3000.f, 65504.f * 126.f};
  for (unsigned j = 0; j < sizeof amps / sizeof amps[0]; j++) {
    float x[32], deq[32], mx = 0;
    int8_t codes[32];
    uint16_t scale;
    for (unsigned i = 0; i < 32; i++) {
      x[i] = (float)(-1. + 2. * i / 31.) * amps[j];
      mx = fmaxf(mx, fabsf(x[i]));
    }
    quant_(x, 32, deq, codes, &scale);
    float g = fmaxf(mx, 0x1p-100f), inv = (1.f / g) * 127.f;
    uint16_t want = half_(g * (1.f / 127.f));
    check(scale == want, "independent quantizer scale");
    for (unsigned i = 0; i < 32; i++) {
      check(codes[i] == (int8_t)nearbyintf(x[i] * inv),
            "independent quantizer code");
      check(deq[i] == (float)codes[i] * unhalf_(want),
            "independent quantizer dequantization");
    }
    char name[48];
    snprintf(name, sizeof name, "quantizer_amplitude_%u", j);
    record(name, measure(x, 0, deq, 0, 32), DBL_MAX, DBL_MAX, 0);
  }
  float x[32] = {0}, d[32];
  int8_t q[32];
  uint16_t scale;
  const float invalid[] = {NAN, INFINITY, -INFINITY, 1e8f};
  for (unsigned j = 0; j < 4; j++) {
    x[0] = invalid[j];
    check(qsaoracle_quant(x, 1, q, &scale, d) != 0,
          "quantizer rejects nonfinite/overflow input");
  }
  double maxrel = 0;
  for (unsigned i = 0; i <= 17400; i++) {
    float a = (float)(-87. + .01 * i), y = exp_gpu(a);
    check(isfinite(y) && y > 0, "finite positive polynomial exp");
    maxrel = fmax(maxrel, fabs((double)y / exp((double)a) - 1.));
  }
  check(maxrel < 2e-6, "polynomial exp relative error <2e-6");
  check(exp_gpu(-1000.f) == exp_gpu(-87.f) && exp_gpu(1000.f) == exp_gpu(87.f),
        "exp clamp boundaries");
  check(isinf(rcp_gpu(0.f)) && signbit(rcp_gpu(-0.f)),
        "reciprocal signed zero");
  const float rx[] = {1e-30f, -1e-30f, 1.f, 3.f, 1e30f};
  for (unsigned i = 0; i < 5; i++)
    check(bits(rcp_gpu(rx[i])) == bits(1.f / rx[i]),
          "correctly rounded reciprocal");
  unsigned n = 2052;
  float *raw = alloc_((size_t)n * 128, 4), *cs = alloc_((size_t)n * 32, 4),
        *sn = alloc_((size_t)n * 32, 4), iq[512], query[512] = {0}, qn[128],
        kn[128], scores[513];
  uint16_t *pool = alloc_(513 * 128, 2);
  uint32_t ids[512];
  for (unsigned i = 0; i < n * 32; i++)
    cs[i] = 1;
  for (unsigned i = 0; i < 128; i++)
    qn[i] = kn[i] = 1;
  check(!qsaoracle_index(raw, n, query, qn, kn, cs, sn, iq, pool, scores),
        "zero indexer domain");
  for (unsigned i = 0; i < 513; i++)
    check(bits(scores[i]) == 0, "canonical positive-zero scores");
  check(idx_select_gpu(scores, 513, 512, ids) == 512,
        "zero indexer selection count");
  for (unsigned i = 0; i < 512; i++)
    check(ids[i] == i, "zero indexer lower-ID selection");
  const float ibad[] = {NAN, INFINITY, -INFINITY, 1e5f};
  for (unsigned i = 0; i < 4; i++) {
    raw[0] = ibad[i];
    check(qsaoracle_index(raw, n, query, qn, kn, cs, sn, iq, pool, scores) != 0,
          "invalid raw-key domain");
  }
  raw[0] = 0;
  query[0] = NAN;
  check(qsaoracle_index(raw, n, query, qn, kn, cs, sn, iq, pool, scores) != 0,
        "invalid index query");
  query[0] = 0;
  qn[0] = INFINITY;
  check(qsaoracle_index(raw, n, query, qn, kn, cs, sn, iq, pool, scores) != 0,
        "invalid gamma");
  qn[0] = 1;
  cs[0] = NAN;
  check(qsaoracle_index(raw, n, query, qn, kn, cs, sn, iq, pool, scores) != 0,
        "invalid RoPE");
  float *zero = alloc_(6144, 4), *po = alloc_(49152, 4), pm[192] = {0};
  check(qsaoracle_attention(zero, zero, zero, 0, 0, zero, zero, pm, pm, po) !=
            0,
        "empty attention rejected");
  free(raw);
  free(cs);
  free(sn);
  free(pool);
  free(zero);
  free(po);
  return end_();
}
static void full_chain_check(const char *dir, const float *expected) {
  unsigned n = 2052, nv = 513;
  float *raw = read_(dir, "raw_keys.f32", (size_t)n * 128, 4),
        *query = read_(dir, "index_query.f32", 512, 4),
        *qn = read_(dir, "q_gamma.f32", 128, 4),
        *kn = read_(dir, "k_gamma.f32", 128, 4),
        *cs = alloc_((size_t)n * 32, 4), *sn = alloc_((size_t)n * 32, 4);
  uint16_t *old = read_(dir, "local_pool_bits.u16", (size_t)nv * 128, 2),
           *pool = alloc_((size_t)nv * 128, 2);
  float iq[512], score[513], riq[512],
      *rp = alloc_((size_t)nv * 128, 4), *hq = read_(dir, "hf_iq.f32", 512, 4),
      *hp = read_(dir, "hf_pool.f32", (size_t)nv * 128, 4);
  rope_(n, cs, sn);
  if (qsaoracle_index(raw, n, query, qn, kn, cs, sn, iq, pool, score))
    die("full-chain candidate execution failed");
  if (capture_dir) {
    save_(capture_dir, "full_chain_iq.f32", iq, 512, 4);
    save_(capture_dir, "full_chain_pool.u16", pool, (size_t)nv * 128, 2);
    save_(capture_dir, "full_chain_score.f32", score, nv, 4);
  }
  check(!memcmp(score, expected, sizeof score),
        "full-chain current score bits equal external-capture run");
  for (size_t i = 0; i < (size_t)nv * 128; i++)
    check(qsa_h2f(pool[i]) == qsa_h2f(old[i]),
          "full-chain pooled-key replay (zero signs ignored)");
  for (unsigned h = 0; h < 4; h++)
    norm_ref(query + h * 128, qn, cs + (size_t)(n - 1) * 32,
             sn + (size_t)(n - 1) * 32, 0, riq + h * 128);
  for (unsigned b = 0; b < nv; b++) {
    float mean[128];
    for (unsigned z = 0; z < 128; z++) {
      double s = 0;
      for (unsigned t = 0; t < 4; t++)
        s += raw[((size_t)b * 4 + t) * 128 + z];
      mean[z] = (float)(s * .25);
    }
    norm_ref(mean, kn, cs + (size_t)b * 128, sn + (size_t)b * 128, 0,
             rp + (size_t)b * 128);
  }
  record("full_chain_independent_query_vs_HF", measure(hq, 0, riq, 0, 512),
         1e-5, 1e-5, 1);
  record("full_chain_independent_pool_vs_HF",
         measure(hp, 0, rp, 0, (size_t)nv * 128), 1e-5, 1e-5, 1);
  free(raw);
  free(query);
  free(qn);
  free(kn);
  free(cs);
  free(sn);
  free(old);
  free(pool);
  free(rp);
  free(hq);
  free(hp);
}
static int cutoff(const char *dir, unsigned n, int full) {
  if (n < 513 || n > 1024 || (full && n != 513))
    die("cutoff score count");
  begin_(full ? "frozen_full_chain_cutoff" : "frozen_cutoff", n);
  float *raw = read_(dir, "raw.f32", n, 4),
        *scaled = read_(dir, "scaled.f32", n, 4),
        *hf = read_(dir, "hf_score.f32", n, 4);
  uint32_t *hid = read_(dir, "hf_ids.u32", 512, 4), a[512], b[512];
  for (unsigned i = 0; i < n; i++) {
    if (!isfinite(raw[i]) || !isfinite(scaled[i]) || !isfinite(hf[i]))
      die("nonfinite cutoff fixture");
    check(bits(raw[i] / 11.313708305358887f) == bits(scaled[i]),
          "pinned FP32 division/cast sequence");
  }
  if (idx_select_gpu(raw, n, 512, a) != 512 ||
      idx_select_gpu(scaled, n, 512, b) != 512)
    die("cutoff candidate selection execution failed");
  record("scaled_score_error", measure(hf, 0, scaled, 0, n), DBL_MAX, .005, 1);
  if (capture_dir) {
    save_(capture_dir, "raw_score.f32", raw, n, 4);
    save_(capture_dir, "scaled_score.f32", scaled, n, 4);
    save_(capture_dir, "selected_ids.u32", a, 512, 4);
    save_(capture_dir, "scaled_selected_ids.u32", b, 512, 4);
  }
  check(lower_policy(raw, n, a, 512) && lower_policy(scaled, n, b, 512),
        "lower-ID cutoff policy");
  check(admissible(hf, n, hid, 512), "external top-k cutoff admissibility");
  if (raw[511] != raw[512] && scaled[511] == scaled[512]) {
    float lo = fminf(raw[511], raw[512]), hi = fmaxf(raw[511], raw[512]);
    check(nextafterf(lo, INFINITY) == hi, "adjacent FP32 cutoff scores");
    for (unsigned i = 0; i < 511; i++)
      check(raw[i] > hi, "511 scores strictly above cutoff pair");
    check(overlap_(a, b, 512) == (raw[511] < raw[512] ? 511u : 512u),
          "division tie changes expected ID order");
  }
  if (full)
    full_chain_check(dir, raw);
  free(raw);
  free(scaled);
  free(hf);
  free(hid);
  return end_();
}
static int frozen_index(const char *dir, unsigned n) {
  begin_("frozen_index_scores", n);
  unsigned nv = n / 4, k = nv < 512 ? nv : 512;
  float *iq = read_(dir, "local_iq.f32", 512, 4),
        *hq = read_(dir, "hf_iq.f32", 512, 4),
        *hp = read_(dir, "hf_pool.f32", (size_t)nv * 128, 4),
        *hs = read_(dir, "hf_score.f32", nv, 4),
        *raw = read_(dir, "raw_score.f32", nv, 4);
  uint16_t *pool = read_(dir, "local_pool_bits.u16", (size_t)nv * 128, 2);
  uint32_t *hid = read_(dir, "hf_ids.u32", k, 4), ids[512];
  index_eval(nv, iq, pool, hq, hp, hs, hid, raw, NULL, ids);
  free(iq);
  free(hq);
  free(hp);
  free(hs);
  free(raw);
  free(pool);
  free(hid);
  return end_();
}
static int frozen_attention(const char *dir, unsigned n) {
  begin_("frozen_attention", n);
  float *q = read_(dir, "query.f32", 6144, 4),
        *k = read_(dir, "key.f32", (size_t)n * 512, 4),
        *v = read_(dir, "value.f32", (size_t)n * 512, 4),
        *g = read_(dir, "gate.f32", 6144, 4);
  double *ref = read_(dir, "reference.f64", 6144, 8),
         *qr = read_(dir, "quant_reference.f64", 6144, 8);
  attention_eval(q, k, v, n, g, ref, qr, NULL);
  free(q);
  free(k);
  free(v);
  free(g);
  free(ref);
  free(qr);
  return end_();
}
static int frozen_bf16(const char *dir) {
  begin_("frozen_BF16_boundary", 20);
  float *before = read_(dir, "mean_before_cast.f32", 640, 4),
        *after = read_(dir, "mean_after_cast.f32", 640, 4),
        *normalized = read_(dir, "normalized.f32", 640, 4), native[640],
        one[128], cs[32], sn[32] = {0};
  unsigned changed = 0;
  for (unsigned i = 0; i < 640; i++) {
    check(bits(bf16(before[i])) == bits(after[i]),
          "FP32 mean cast to BF16 before norm");
    changed += before[i] != after[i];
  }
  check(changed > 0, "BF16 probe actually rounds");
  for (unsigned i = 0; i < 128; i++)
    one[i] = 1;
  for (unsigned i = 0; i < 32; i++)
    cs[i] = 1;
  for (unsigned b = 0; b < 5; b++)
    norm_ref(after + b * 128, one, cs, sn, 1, native + b * 128);
  record("BF16_norm_independent_vs_HF", measure(normalized, 0, native, 0, 640),
         3e-5, 3e-5, 1);
  free(before);
  free(after);
  free(normalized);
  return end_();
}
static int frozen_sparse(const char *dir, unsigned n) {
  begin_("frozen_sparse_outputs", n);
  unsigned nv = n / 4, k = nv < 512 ? nv : 512, nk = 4 * k + n % 4;
  uint32_t *lt = read_(dir, "local_token_ids.u32", nk, 4),
           *rt = read_(dir, "hf_token_ids.u32", nk, 4), lb[512], rb[512];
  for (unsigned j = 0; j < k; j++) {
    lb[j] = lt[4 * j] / 4;
    rb[j] = rt[4 * j] / 4;
    for (unsigned t = 0; t < 4; t++)
      check(lt[j * 4 + t] == lb[j] * 4 + t && rt[j * 4 + t] == rb[j] * 4 + t,
            "whole selected blocks");
  }
  for (unsigned t = 0; t < n % 4; t++)
    check(lt[k * 4 + t] == nv * 4 + t && rt[k * 4 + t] == nv * 4 + t,
          "all visible tail tokens");
  float *q = read_(dir, "query.f32", 6144, 4),
        *g = read_(dir, "gate.f32", 6144, 4);
  double *ref = read_(dir, "external_selection_output.f64", 6144, 8);
  sparse_eval(n, lb, rb, q, g, ref, NULL);
  free(lt);
  free(rt);
  free(q);
  free(g);
  free(ref);
  return end_();
}
int main(int argc, char **argv) {
  uint32_t endian = 1;
  if (*(unsigned char *)&endian != 1 || sizeof(float) != 4 ||
      sizeof(double) != 8 || FLT_MANT_DIG != 24 || DBL_MANT_DIG != 53 ||
      !qsaoracle_environment())
    die("requires IEEE little-endian binary32/binary64, FE_TONEAREST, "
        "FLT_EVAL_METHOD=0");
  if (argc == 2 && !strcmp(argv[1], "--self-test"))
    return self_test(0);
  if (argc == 2 && !strcmp(argv[1], "--negative-score"))
    return self_test(1);
  if (argc == 3 && !strcmp(argv[1], "--bf16"))
    return frozen_bf16(argv[2]);
  capture_dir = getenv("QSA_ORACLE_CAPTURE_DIR");
  if (capture_dir && (!*capture_dir || mkdir(capture_dir, 0755)))
    die("capture directory must be new");
  const char *cubin = getenv("GPU_CUBIN");
  if (cubin) {
#ifdef QSA_ORACLE_TARGET
    const char *dev = getenv("GPU_DEVICE");
    char *end;
    errno = 0;
    long ordinal = dev ? strtol(dev, &end, 10) : 0;
    if (dev && (errno || !*dev || *end || ordinal < 0 || ordinal > INT_MAX))
      die("invalid GPU_DEVICE");
    if (!*cubin || qsaoracle_gpu_init(cubin, (int)ordinal))
      die("requested GPU initialization failed");
    gpu_mode = 1;
    if (atexit(qsaoracle_gpu_finish))
      die("cannot register GPU cleanup");
#else
    die("GPU requested from a CPU-only binary; build qsa_oracle_gpu");
#endif
  }

  if (argc == 4) {
    unsigned n = count_(argv[3]);
    if (!strcmp(argv[1], "--generate-index"))
      return generated_index(argv[2], n);
    if (!strcmp(argv[1], "--generate-attention"))
      return generated_attention(argv[2], n);
    if (!strcmp(argv[1], "--cutoff"))
      return cutoff(argv[2], n, 0);
    if (!strcmp(argv[1], "--full-chain"))
      return cutoff(argv[2], n, 1);
    if (!strcmp(argv[1], "--index"))
      return frozen_index(argv[2], n);
    if (!strcmp(argv[1], "--attention"))
      return frozen_attention(argv[2], n);
    if (!strcmp(argv[1], "--sparse"))
      return frozen_sparse(argv[2], n);
  }
  die("usage: qsa_oracle --self-test | --negative-score | --bf16 DIR | "
      "--cutoff|--full-chain|--index|--attention|--sparse|--generate-index|--"
      "generate-attention DIR LENGTH");
  return 2;
}
