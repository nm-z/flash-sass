/* Target-SM52 candidate adapter for the independent native QSA oracle.
 * Executes IDXQK (normalization, RoPE, pooling), IDXSCORE, sparse IDXSEL,
 * ATTNQKQ quantization, ATTNPARTN and ATTNMERGE. Host validation, dense
 * all-block selection, dequantization and fixture construction are not GPU
 * measurements. Whole-checkpoint logits are outside this adapter.
 */
#include "qsa_oracle_gpu.h"
#include "attn.h"
#include "qsa.h"
#include "dieops.h"
#include <cuda.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
CUresult cuCtxCreate_v2(CUcontext *, unsigned int, CUdevice);
typedef struct {
  uint32_t w[32];
} gpu_desc;
static CUcontext context;
static CUmodule module;
static CUfunction function;
static CUdeviceptr program, control;
static unsigned long launch_count;
static int ck(CUresult r, const char *call) {
  if (r == CUDA_SUCCESS)
    return 0;
  const char *error = "unknown CUDA error";
  (void)cuGetErrorString(r, &error);
  fprintf(stderr, "qsa_oracle_gpu: %s: %s (%d)\n", call, error, (int)r);
  return -1;
}
#define TRY(call)                                                              \
  do {                                                                         \
    if (ck((call), #call))                                                     \
      goto done;                                                               \
  } while (0)
static int release(CUdeviceptr *p, unsigned n) {
  int failed = 0;
  for (unsigned i = 0; i < n; i++)
    if (p[i] && ck(cuMemFree(p[i]), "cuMemFree"))
      failed = 1;
  return failed;
}
static void init_desc(gpu_desc *d, unsigned op, unsigned units) {
  memset(d, 0, sizeof *d);
  d->w[0] = op;
  d->w[1] = units;
}
static void ptr(gpu_desc *d, unsigned address, CUdeviceptr p) {
  unsigned w = 2 + (address - 0x140) / 4;
  d->w[w] = (uint32_t)p;
  d->w[w + 1] = (uint32_t)(p >> 32);
}
static void u32(gpu_desc *d, unsigned address, uint32_t v) {
  d->w[2 + (address - 0x140) / 4] = v;
}
static int run(gpu_desc *d, unsigned count, unsigned token) {
  int status = -1;
  if (!context || !count || count > 1024 || token == UINT_MAX)
    return -1;
  TRY(cuMemcpyHtoD(program, d, (size_t)count * sizeof *d));
  TRY(cuMemsetD8(control, 0, 8192));
  TRY(cuMemcpyHtoD(control + 8, &token, sizeof token));
  unsigned end = token + 1;
  void *args[] = {&program, &count, &control, &end};
  TRY(cuLaunchKernel(function, 16, 1, 1, 256, 1, 1, 0, 0, args, NULL));
  launch_count++;
  TRY(cuCtxSynchronize());
  status = 0;
done:
  return status;
}
void qsaoracle_gpu_finish(void) {
  CUdeviceptr p[] = {program, control};
  if (context)
    (void)release(p, 2);
  if (module)
    (void)ck(cuModuleUnload(module), "cuModuleUnload");
  if (context)
    (void)ck(cuCtxDestroy(context), "cuCtxDestroy");
  context = NULL;
  module = NULL;
  function = NULL;
  program = control = 0;
}
int qsaoracle_gpu_init(const char *cubin, int ordinal) {
  int status = -1, major = 0, minor = 0, driver = 0;
  CUdevice device;
  char name[128] = {0};
  if (!cubin || ordinal < 0 || context)
    return -1;
  TRY(cuInit(0));
  TRY(cuDeviceGet(&device, ordinal));
  TRY(cuDeviceGetName(name, sizeof name, device));
  TRY(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                           device));
  TRY(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                           device));
  TRY(cuDriverGetVersion(&driver));
  if (major != 5 || minor != 2) {
    fprintf(stderr, "qsa_oracle_gpu: requires SM52, device %s is SM%d%d\n",
            name, major, minor);
    goto done;
  }
  TRY(cuCtxCreate_v2(&context, CU_CTX_SCHED_SPIN, device));
  TRY(cuModuleLoad(&module, cubin));
  TRY(cuModuleGetFunction(&function, module, "die"));
  TRY(cuMemAlloc(&program, 1024 * sizeof(gpu_desc)));
  TRY(cuMemAlloc(&control, 8192));
  launch_count = 0;
  fprintf(stderr,
          "qsa_oracle_gpu: device=%s ordinal=%d sm=%d.%d driver=%d image=%s\n",
          name, ordinal, major, minor, driver, cubin);
  status = 0;
done:
  if (status)
    qsaoracle_gpu_finish();
  return status;
}
unsigned long qsaoracle_gpu_launch_count(void) { return launch_count; }
int qsaoracle_gpu_quant(const float *x, unsigned groups, int8_t *codes,
                        uint16_t *scales, float *dequantized) {
  enum { BATCH = 512 };
  enum { VALUE, QUERY, KEY, NORM, CS, SN, QOUT, KQ, VQ, KS, VS, NBUF };
  /* Values are already at the quantizer boundary. ATTNQKQ uses the same
   * production q8 sequence for V and normalized K. Feeding these fixtures
   * into V isolates that sequence from K normalization and RoPE. Each
   * descriptor uses position zero and its own V output pointers; bounded
   * batches do not allocate a cache at the fixture's context length. */
  CUdeviceptr p[NBUF] = {0};
  float *staging = NULL;
  gpu_desc *d = NULL;
  int status = -1;
  if (!context || groups > 262144u * 16u)
    return -1;
  if (!groups)
    return 0;
  if (!x || !codes || !scales || !dequantized)
    return -1;
  for (size_t i = 0; i < (size_t)groups * 32; i++)
    if (!isfinite(x[i]) || fabsf(x[i]) > 65504.f * 127.f)
      return -1;
  staging = calloc((size_t)BATCH * 512, sizeof *staging);
  d = malloc((size_t)BATCH * sizeof *d);
  if (!staging || !d)
    goto done;
  float norm[256], cosine[32];
  for (unsigned i = 0; i < 256; i++)
    norm[i] = 1.f;
  for (unsigned i = 0; i < 32; i++)
    cosine[i] = 1.f;
  TRY(cuMemAlloc(&p[VALUE], (size_t)BATCH * 512 * 4));
  TRY(cuMemAlloc(&p[QUERY], 12288 * 4));
  TRY(cuMemAlloc(&p[KEY], 512 * 4));
  TRY(cuMemAlloc(&p[NORM], 256 * 4));
  TRY(cuMemAlloc(&p[CS], 32 * 4));
  TRY(cuMemAlloc(&p[SN], 32 * 4));
  TRY(cuMemAlloc(&p[QOUT], 6144 * 4));
  TRY(cuMemAlloc(&p[KQ], 512));
  TRY(cuMemAlloc(&p[VQ], (size_t)BATCH * 512));
  TRY(cuMemAlloc(&p[KS], 16 * 2));
  TRY(cuMemAlloc(&p[VS], (size_t)BATCH * 16 * 2));
  TRY(cuMemsetD8(p[QUERY], 0, 12288 * 4));
  TRY(cuMemsetD8(p[KEY], 0, 512 * 4));
  TRY(cuMemcpyHtoD(p[NORM], norm, 256 * 4));
  TRY(cuMemcpyHtoD(p[CS], cosine, 32 * 4));
  TRY(cuMemsetD8(p[SN], 0, 32 * 4));
  const float epsilon = 1e-6f;
  uint32_t epsilon_bits;
  memcpy(&epsilon_bits, &epsilon, sizeof epsilon_bits);
  for (unsigned base = 0; base < groups;) {
    unsigned count = groups - base;
    if (count > BATCH * 16u)
      count = BATCH * 16u;
    unsigned rows = (count + 15) / 16;
    memset(staging, 0, (size_t)rows * 512 * 4);
    memcpy(staging, x + (size_t)base * 32, (size_t)count * 32 * 4);
    for (unsigned i = 0; i < rows; i++) {
      init_desc(&d[i], OP_ATTNQKQ, 26);
      ptr(&d[i], 0x140, p[QUERY]);
      ptr(&d[i], 0x148, p[KEY]);
      ptr(&d[i], 0x150, p[VALUE] + (size_t)i * 512 * 4);
      ptr(&d[i], 0x158, p[NORM]);
      ptr(&d[i], 0x160, p[NORM]);
      ptr(&d[i], 0x168, p[CS]);
      ptr(&d[i], 0x170, p[SN]);
      ptr(&d[i], 0x178, p[QOUT]);
      ptr(&d[i], 0x180, p[KQ]);
      ptr(&d[i], 0x188, p[VQ] + (size_t)i * 512);
      u32(&d[i], 0x190, 0);
      u32(&d[i], 0x194, epsilon_bits);
      ptr(&d[i], 0x198, p[KS]);
      ptr(&d[i], 0x1a0, p[VS] + (size_t)i * 16 * 2);
    }
    TRY(cuMemcpyHtoD(p[VALUE], staging, (size_t)rows * 512 * 4));
    /* -128 and a NaN scale cannot be valid finite-domain quantizer output. */
    TRY(cuMemsetD8(p[VQ], 0x80, (size_t)rows * 512));
    TRY(cuMemsetD16(p[VS], 0x7e00, (size_t)rows * 16));
    if (run(d, rows, 0))
      goto done;
    TRY(cuMemcpyDtoH(codes + (size_t)base * 32, p[VQ], (size_t)count * 32));
    TRY(cuMemcpyDtoH(scales + base, p[VS], (size_t)count * 2));
    for (unsigned b = 0; b < count; b++) {
      uint16_t h = scales[base + b];
      if ((h & 0x7c00u) == 0x7c00u || (h & 0x8000u))
        goto done;
      float scale = qsa_h2f(h);
      for (unsigned i = 0; i < 32; i++) {
        size_t at = (size_t)(base + b) * 32 + i;
        if (codes[at] == INT8_MIN)
          goto done;
        dequantized[at] = (float)codes[at] * scale;
      }
    }
    base += count;
  }
  status = 0;
done:
  if (release(p, NBUF))
    status = -1;
  free(staging);
  free(d);
  return status;
}
static int score_and_select(const float *iq, const uint16_t *pool, unsigned nv,
                            unsigned nblk, float *scores, uint32_t *ids) {
  int status = -1;
  CUdeviceptr p[5] = {0};
  uint16_t *packed = NULL;
  if (!context || nv > 65536 || nblk > 65536)
    return -1;
  if (!nv)
    return 0;
  if (!iq || !pool || !scores)
    return -1;
  for (unsigned i = 0; i < 512; i++)
    if (!isfinite(iq[i]))
      return -1;
  unsigned groups = (nv + 31) / 32 + 1;
  size_t packed_count = (size_t)groups * 4096;
  int sparse_selection = ids && nblk && nv > nblk;
  packed = calloc(packed_count, sizeof *packed);
  if (!packed)
    return -1;
  for (unsigned b = 0; b < nv; b++)
    for (unsigned k = 0; k < 128; k++) {
      uint16_t v = pool[(size_t)b * 128 + k];
      if ((v & 0x7c00u) == 0x7c00u)
        goto done;
      size_t i = (size_t)(b >> 5) * 4096 + (size_t)(k >> 2) * 128 +
                 (b & 31) * 4 + (k & 3);
      packed[i] = v;
    }
  TRY(cuMemAlloc(&p[0], packed_count * 2));
  TRY(cuMemAlloc(&p[1], 512 * 4));
  TRY(cuMemAlloc(&p[2], ((size_t)groups * 32 + 32) * 4));
  TRY(cuMemAlloc(&p[3], 1280 * 4));
  TRY(cuMemcpyHtoD(p[0], packed, packed_count * 2));
  TRY(cuMemcpyHtoD(p[1], iq, 512 * 4));
  TRY(cuMemsetD32(p[2], 0x7fc00000u, (size_t)groups * 32 + 32));
  TRY(cuMemsetD8(p[3], 0, 1280 * 4));
  gpu_desc d[6];
  init_desc(&d[0], OP_IDXSCORE, 16);
  ptr(&d[0], 0x140, p[0]);
  ptr(&d[0], 0x148, p[1]);
  ptr(&d[0], 0x150, p[2]);
  ptr(&d[0], 0x158, p[3]);
  u32(&d[0], 0x160, 4 * nv - 1);
  u32(&d[0], 0x164, 0); /* Force scoring at dense-bypass lengths. */
  if (sparse_selection) {
    TRY(cuMemAlloc(&p[4], (size_t)(nblk + 8) * 4));
    TRY(cuMemsetD32(p[4], UINT_MAX, nblk + 8));
    for (unsigned i = 0; i < 5; i++) {
      init_desc(&d[i + 1], OP_IDXSEL, 16);
      ptr(&d[i + 1], 0x140, p[2]);
      ptr(&d[i + 1], 0x148, p[3]); /* actual IDXSCORE histogram */
      ptr(&d[i + 1], 0x150, p[4]);
      u32(&d[i + 1], 0x158, 4 * nv - 1);
      u32(&d[i + 1], 0x15c, nblk);
      u32(&d[i + 1], 0x160, 2 + i);
    }
  }
  if (run(d, sparse_selection ? 6 : 1, 4 * nv - 1))
    goto done;
  TRY(cuMemcpyDtoH(scores, p[2], (size_t)nv * 4));
  for (unsigned b = 0; b < nv; b++)
    if (!isfinite(scores[b]) || scores[b] < 0)
      goto done;
  if (sparse_selection) {
    TRY(cuMemcpyDtoH(ids, p[4], (size_t)nblk * 4));
    for (unsigned i = 0; i < nblk; i++)
      if (ids[i] >= nv || (i && ids[i] <= ids[i - 1]))
        goto done;
  } else if (ids) {
    unsigned count = nv < nblk ? nv : nblk;
    for (unsigned i = 0; i < count; i++)
      ids[i] = i;
  }
  status = 0;
done:
  if (release(p, 5))
    status = -1;
  free(packed);
  return status;
}
int qsaoracle_gpu_scores(const float *iq, const uint16_t *pool, unsigned nv,
                         float *scores) {
  return score_and_select(iq, pool, nv, 0, scores, NULL);
}
unsigned qsaoracle_gpu_scores_select(const float *iq, const uint16_t *pool,
                                     unsigned nv, unsigned nblk, float *scores,
                                     uint32_t *ids) {
  if (nv && nblk && !ids)
    return UINT_MAX;
  if (score_and_select(iq, pool, nv, nblk, scores, ids))
    return UINT_MAX;
  return nv < nblk ? nv : nblk;
}
unsigned qsaoracle_gpu_select(const float *scores, unsigned nv, unsigned nblk,
                              uint32_t *ids) {
  unsigned answer = UINT_MAX;
  CUdeviceptr p[3] = {0};
  float *canonical = NULL;
  uint32_t hist[1280] = {0};
  if (!context || nv > 65536 || nblk > 65536)
    return UINT_MAX;
  if (!nblk || !nv)
    return 0;
  if (!scores || !ids)
    return UINT_MAX;
  for (unsigned i = 0; i < nv; i++)
    if (!isfinite(scores[i]) || scores[i] < 0)
      return UINT_MAX;
  if (nv <= nblk) {
    for (unsigned i = 0; i < nv; i++)
      ids[i] = i;
    return nv;
  }
  canonical = malloc((size_t)nv * 4);
  if (!canonical)
    return UINT_MAX;
  for (unsigned i = 0; i < nv; i++) {
    uint32_t bits;
    canonical[i] = scores[i] == 0 ? 0 : scores[i];
    memcpy(&bits, &canonical[i], 4);
    hist[bits >> 23]++;
  }
  TRY(cuMemAlloc(&p[0], (size_t)(nv + 8) * 4));
  TRY(cuMemAlloc(&p[1], 1280 * 4));
  TRY(cuMemAlloc(&p[2], (size_t)(nblk + 8) * 4));
  TRY(cuMemcpyHtoD(p[0], canonical, (size_t)nv * 4));
  TRY(cuMemcpyHtoD(p[1], hist, sizeof hist));
  TRY(cuMemsetD32(p[2], UINT_MAX, nblk + 8));
  gpu_desc d[5];
  for (unsigned i = 0; i < 5; i++) {
    init_desc(&d[i], OP_IDXSEL, 16);
    ptr(&d[i], 0x140, p[0]);
    ptr(&d[i], 0x148, p[1]);
    ptr(&d[i], 0x150, p[2]);
    u32(&d[i], 0x158, 4 * nv - 1);
    u32(&d[i], 0x15c, nblk);
    u32(&d[i], 0x160, 2 + i);
  }
  if (run(d, 5, 4 * nv - 1))
    goto done;
  TRY(cuMemcpyDtoH(ids, p[2], (size_t)nblk * 4));
  for (unsigned i = 0; i < nblk; i++)
    if (ids[i] >= nv || (i && ids[i] <= ids[i - 1]))
      goto done;
  answer = nblk;
done:
  if (release(p, 3))
    answer = UINT_MAX;
  free(canonical);
  return answer;
}
int qsaoracle_gpu_attention(const float *query, const float *key,
                            const float *value, unsigned n, unsigned capacity,
                            const float *gate, float *out, float *pm, float *pd,
                            float *po) {
  int status = -1;
  CUdeviceptr p[8] = {0};
  float *packed = NULL, *packed_gate = NULL;
  if (!context || !query || !key || !value || !gate || !out || !pm || !pd ||
      !po || !n || n > 262144 || capacity > 524288 ||
      capacity < ((n + 8191) / 8192) * 8192)
    return -1;
  for (unsigned i = 0; i < 6144; i++)
    if (!isfinite(query[i]) || fabsf(query[i]) > 1e4f || !isfinite(gate[i]) ||
        fabsf(gate[i]) > 80.f)
      return -1;
  for (size_t i = 0; i < (size_t)n * 512; i++)
    if (!isfinite(key[i]) || !isfinite(value[i]) || fabsf(key[i]) > 1e4f ||
        fabsf(value[i]) > 1e4f)
      return -1;
  for (size_t i = (size_t)n * 512; i < (size_t)capacity * 512; i++)
    if (key[i] != 0 || value[i] != 0)
      return -1;
  /* Allocation also covers the production kernel's stray prefetch. */
  unsigned physical = (n + 8192 + 64 + 31) / 32 * 32;
  if (physical < capacity)
    physical = (capacity + 31) / 32 * 32;
  size_t key_count = ATTN_KC_FLOATS(physical);
  size_t value_count = (size_t)physical * 512;
  packed = calloc(key_count, sizeof *packed);
  packed_gate = calloc(12288, sizeof *packed_gate);
  if (!packed || !packed_gate)
    goto done;
  for (unsigned j = 0; j < n; j++)
    for (unsigned h = 0; h < 2; h++)
      for (unsigned d = 0; d < 256; d++)
        packed[attn_kidx(j, h, d)] = key[(size_t)j * 512 + h * 256 + d];
  for (unsigned h = 0; h < 24; h++)
    memcpy(packed_gate + h * 512 + 256, gate + h * 256, 256 * 4);
  TRY(cuMemAlloc(&p[0], 6144 * 4));
  TRY(cuMemAlloc(&p[1], key_count * 4));
  TRY(cuMemAlloc(&p[2], value_count * 4));
  TRY(cuMemAlloc(&p[3], 12288 * 4));
  TRY(cuMemAlloc(&p[4], 6144 * 4));
  TRY(cuMemAlloc(&p[5], 24 * 8 * 4));
  TRY(cuMemAlloc(&p[6], 24 * 8 * 4));
  TRY(cuMemAlloc(&p[7], 24 * 8 * 256 * 4));
  TRY(cuMemcpyHtoD(p[0], query, 6144 * 4));
  TRY(cuMemcpyHtoD(p[1], packed, key_count * 4));
  TRY(cuMemsetD8(p[2], 0, value_count * 4));
  TRY(cuMemcpyHtoD(p[2], value, (size_t)n * 512 * 4));
  TRY(cuMemcpyHtoD(p[3], packed_gate, 12288 * 4));
  TRY(cuMemsetD32(p[4], 0x7fc00000u, 6144));
  TRY(cuMemsetD32(p[5], 0x7fc00000u, 24 * 8));
  TRY(cuMemsetD32(p[6], 0x7fc00000u, 24 * 8));
  TRY(cuMemsetD32(p[7], 0x7fc00000u, 24 * 8 * 256));
  gpu_desc d[2];
  init_desc(&d[0], OP_ATTNPARTN, 32);
  ptr(&d[0], 0x140, p[0]);
  ptr(&d[0], 0x148, p[1]);
  ptr(&d[0], 0x150, p[2]);
  ptr(&d[0], 0x158, p[5]);
  ptr(&d[0], 0x160, p[6]);
  ptr(&d[0], 0x168, p[7]);
  u32(&d[0], 0x170, n);
  u32(&d[0], 0x174, (n + 3) / 4);
  init_desc(&d[1], OP_ATTNMERGE, 24);
  ptr(&d[1], 0x140, p[5]);
  ptr(&d[1], 0x148, p[6]);
  ptr(&d[1], 0x150, p[7]);
  ptr(&d[1], 0x158, p[3]);
  ptr(&d[1], 0x160, p[4]);
  if (run(d, 2, n - 1))
    goto done;
  TRY(cuMemcpyDtoH(out, p[4], 6144 * 4));
  TRY(cuMemcpyDtoH(pm, p[5], 24 * 8 * 4));
  TRY(cuMemcpyDtoH(pd, p[6], 24 * 8 * 4));
  TRY(cuMemcpyDtoH(po, p[7], 24 * 8 * 256 * 4));
  for (unsigned i = 0; i < 6144; i++)
    if (!isfinite(out[i]))
      goto done;
  for (unsigned i = 0; i < 24 * 8; i++)
    if (!isfinite(pd[i]) || pd[i] < 0 || (pd[i] > 0 && !isfinite(pm[i])))
      goto done;
  for (unsigned i = 0; i < 24 * 8 * 256; i++)
    if (!isfinite(po[i]))
      goto done;
  status = 0;
done:
  if (release(p, 8))
    status = -1;
  free(packed);
  free(packed_gate);
  return status;
}
int qsaoracle_gpu_index(const float *raw, unsigned n, const float *query,
                        const float *qn, const float *kn, const float *cs,
                        const float *sn, float *iq, uint16_t *pool,
                        float *scores) {
  enum { BATCH = 1024 };
  int status = -1;
  CUdeviceptr p[9] = {0};
  float *staging = NULL;
  uint16_t *packed = NULL;
  gpu_desc *d = NULL;
  if (!context || !n || n > 262144 || !raw || !query || !qn || !kn || !cs ||
      !sn || !iq || (n >= 4 && (!pool || !scores)))
    return -1;
  for (size_t i = 0; i < (size_t)n * 128; i++)
    if (!isfinite(raw[i]) || fabsf(raw[i]) > 1e4f)
      return -1;
  for (unsigned i = 0; i < 512; i++)
    if (!isfinite(query[i]) || fabsf(query[i]) > 1e4f)
      return -1;
  for (unsigned i = 0; i < 128; i++)
    if (!isfinite(qn[i]) || !isfinite(kn[i]) || fabsf(qn[i]) > 16.f ||
        fabsf(kn[i]) > 16.f)
      return -1;
  for (size_t i = 0; i < (size_t)n * 32; i++)
    if (!isfinite(cs[i]) || !isfinite(sn[i]) || fabsf(cs[i]) > 1.f ||
        fabsf(sn[i]) > 1.f)
      return -1;
  unsigned nv = n / 4, groups = (nv + 31) / 32 + 1;
  size_t packed_count = (size_t)groups * 4096;
  size_t rope_bytes = (size_t)n * 32 * 4;
  staging = malloc((size_t)BATCH * 640 * 4);
  packed = malloc(packed_count * 2);
  d = malloc((size_t)BATCH * sizeof *d);
  if (!staging || !packed || !d)
    goto done;
  TRY(cuMemAlloc(&p[0], (size_t)BATCH * 640 * 4));
  TRY(cuMemAlloc(&p[1], 128 * 4));
  TRY(cuMemAlloc(&p[2], 128 * 4));
  TRY(cuMemAlloc(&p[3], rope_bytes));
  TRY(cuMemAlloc(&p[4], rope_bytes));
  TRY(cuMemAlloc(&p[5], 512 * 4));
  TRY(cuMemAlloc(&p[6], 512 * 4));
  TRY(cuMemAlloc(&p[7], packed_count * 2));
  TRY(cuMemAlloc(&p[8], 1280 * 4));
  TRY(cuMemcpyHtoD(p[1], qn, 128 * 4));
  TRY(cuMemcpyHtoD(p[2], kn, 128 * 4));
  TRY(cuMemcpyHtoD(p[3], cs, rope_bytes));
  TRY(cuMemcpyHtoD(p[4], sn, rope_bytes));
  TRY(cuMemsetD32(p[5], 0x7fc00000u, 512));
  TRY(cuMemsetD8(p[6], 0, 512 * 4));
  TRY(cuMemsetD8(p[7], 0, packed_count * 2));
  TRY(cuMemsetD8(p[8], 0, 1280 * 4));
  float epsilon = 1e-6f;
  uint32_t epsilon_bits;
  memcpy(&epsilon_bits, &epsilon, 4);
  for (unsigned base = 0; base < n; base += BATCH) {
    unsigned count = n - base < BATCH ? n - base : BATCH;
    for (unsigned i = 0; i < count; i++) {
      /* Earlier query outputs are overwritten; every raw key is real. */
      memcpy(staging + (size_t)i * 640, query, 512 * 4);
      memcpy(staging + (size_t)i * 640 + 512, raw + (size_t)(base + i) * 128,
             128 * 4);
      init_desc(&d[i], OP_IDXQK, 6);
      ptr(&d[i], 0x140, p[0] + (size_t)i * 640 * 4);
      ptr(&d[i], 0x148, p[1]);
      ptr(&d[i], 0x150, p[2]);
      ptr(&d[i], 0x158, p[3]);
      ptr(&d[i], 0x160, p[4]);
      ptr(&d[i], 0x168, p[5]);
      ptr(&d[i], 0x170, p[6]);
      ptr(&d[i], 0x178, p[7]);
      ptr(&d[i], 0x180, p[8]);
      u32(&d[i], 0x188, base + i);
      u32(&d[i], 0x18c, epsilon_bits);
    }
    TRY(cuMemcpyHtoD(p[0], staging, (size_t)count * 640 * 4));
    if (run(d, count, 0))
      goto done;
  }
  TRY(cuMemcpyDtoH(iq, p[5], 512 * 4));
  TRY(cuMemcpyDtoH(packed, p[7], packed_count * 2));
  for (unsigned i = 0; i < 512; i++)
    if (!isfinite(iq[i]))
      goto done;
  for (unsigned b = 0; b < nv; b++)
    for (unsigned k = 0; k < 128; k++) {
      size_t i = (size_t)(b >> 5) * 4096 + (size_t)(k >> 2) * 128 +
                 (b & 31) * 4 + (k & 3);
      uint16_t v = packed[i];
      if ((v & 0x7c00u) == 0x7c00u)
        goto done;
      pool[(size_t)b * 128 + k] = v;
    }
  status = 0;
done:
  if (release(p, 9))
    status = -1;
  free(staging);
  free(packed);
  free(d);
  if (status)
    return status;
  return qsaoracle_gpu_scores(iq, pool, n / 4, scores);
}
