/* Candidate bridge; independent native mathematics and frozen pinned HF
 * probes are in qsa_oracle.c. These scalar routines are the candidate. */
#include "attn.h"
#include "fmath.h"
#include "qsa.h"
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#ifdef __FAST_MATH__
#error qsa oracle requires strict floating point
#endif
_Static_assert(sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24,
               "IEEE binary32 required");
int qsaoracle_environment(void) {
  return fegetround() == FE_TONEAREST && FLT_EVAL_METHOD == 0;
}
int qsaoracle_set_rounding(int nearest) {
  return fesetround(nearest ? FE_TONEAREST : FE_DOWNWARD);
}
static int finite_(const float *x, size_t n, float bound) {
  for (size_t i = 0; i < n; i++)
    if (!isfinite(x[i]) || fabsf(x[i]) > bound)
      return 0;
  return 1;
}
int qsaoracle_index(const float *raw, unsigned n, const float *query,
                    const float *qn, const float *kn, const float *cs,
                    const float *sn, float *iq, uint16_t *pool, float *score) {
  if (!qsaoracle_environment() || !n || !finite_(raw, (size_t)n * 128, 1e4f) ||
      !finite_(query, 512, 1e4f) || !finite_(qn, 128, 16.f) ||
      !finite_(kn, 128, 16.f) || !finite_(cs, (size_t)n * 32, 1.f) ||
      !finite_(sn, (size_t)n * 32, 1.f))
    return 1;
  for (int h = 0; h < 4; h++)
    idx_norm_rope_gpu(query + h * 128, qn, 1e-6f, cs + (size_t)(n - 1) * 32,
                      sn + (size_t)(n - 1) * 32, iq + h * 128);
  for (unsigned b = 0; b < n / 4; b++)
    idx_pool_gpu(raw + (size_t)b * 512, kn, 1e-6f, cs + (size_t)b * 128,
                 sn + (size_t)b * 128, pool + (size_t)b * 128);
  idx_scores_gpu(iq, pool, n / 4, score);
  return 0;
}
int qsaoracle_quant(const float *x, unsigned groups, int8_t *q, uint16_t *scale,
                    float *deq) {
  if (!qsaoracle_environment() ||
      !finite_(x, (size_t)groups * 32, 65504.f * 127.f))
    return 1;
  for (unsigned b = 0; b < groups; b++) {
    kv_quant32_gpu(x + (size_t)b * 32, q + (size_t)b * 32, scale + b);
    if (!isfinite(qsa_h2f(scale[b])))
      return 1;
    for (int d = 0; d < 32; d++)
      deq[(size_t)b * 32 + d] = kv_deq_gpu(q[(size_t)b * 32 + d], scale[b]);
  }
  return 0;
}
int qsaoracle_attention(const float *query, const float *key,
                        const float *value, unsigned n, unsigned capacity,
                        const float *gate, float *out, float *pm, float *pd,
                        float *po) {
  unsigned need = ((n + 8191) / 8192) * 8192;
  if (!qsaoracle_environment() || !n || capacity < need ||
      !finite_(query, 6144, 1e4f) ||
      !finite_(key, (size_t)capacity * 512, 1e4f) ||
      !finite_(value, (size_t)capacity * 512, 1e4f) ||
      !finite_(gate, 6144, 80.f))
    return 1;
  for (size_t i = (size_t)n * 512; i < (size_t)capacity * 512; i++)
    if (key[i] != 0.f || value[i] != 0.f)
      return 1;
  float *kc = calloc(ATTN_KC_FLOATS(capacity), sizeof(float)),
        *g = calloc(12288, sizeof(float));
  if (!kc || !g) {
    free(kc);
    free(g);
    return 1;
  }
  for (unsigned j = 0; j < n; j++)
    for (int h = 0; h < 2; h++)
      for (int d = 0; d < 256; d++)
        kc[attn_kidx(j, h, d)] = key[(size_t)j * 512 + 256 * h + d];
  for (int h = 0; h < 24; h++)
    memcpy(g + h * 512 + 256, gate + h * 256, 256 * sizeof(float));
  attn_partials_gpu(query, kc, value, n, pm, pd, po);
  attn_merge_gpu(pm, pd, po, g, out);
  free(kc);
  free(g);
  return !finite_(out, 6144, 1e4f);
}
