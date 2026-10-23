/* Finite production QSA operations for a kernel-replay hardware profiler.
 * This uses the unchanged die cubin and production descriptor layout. Each
 * launch is one token, with no peer, IPC, disk pump, or CPU progress dependency.
 * Counters include die dispatch and local grid barriers. KV/query/pool/selector
 * payloads are deterministic nonzero synthetic data. --ids imports checkpoint
 * selected BLOCK IDs, not checkpoint KV values. Caller checks nvidia-smi AND ps
 * and holds the shared-host validation lock before running this executable.
 */
#define _POSIX_C_SOURCE 200809L
#include "attn.h"
#include "dieops.h"
#include <cuda.h>
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
CUresult cuCtxCreate_v2(CUcontext *, unsigned int, CUdevice);
CUresult cuProfilerStart(void);
CUresult cuProfilerStop(void);
#ifndef FLASH_COUNTERS_REV
#define FLASH_COUNTERS_REV "unspecified"
#endif
#define CK(call) do { CUresult r_ = (call); if (r_ != CUDA_SUCCESS) { \
 const char *s_ = "?"; cuGetErrorString(r_, &s_); \
 fprintf(stderr, "%s:%d: %s: %s (%d)\n", __FILE__, __LINE__, #call, s_, r_); exit(1); } } while (0)
#define BUDGET 512u
#define FNV_OFFSET UINT64_C(14695981039346656037)
#define FNV_PRIME UINT64_C(1099511628211)
typedef struct { uint32_t w[32]; } descriptor;
typedef struct { CUdeviceptr d; void *h; size_t n; } buffer;
typedef struct { float score; unsigned id; } rank_item;
typedef struct {
 unsigned iteration, selected_count, attended_keys, tail_keys;
 uint64_t ids_checksum, output_checksum;
 double host_launch_sync_ms; float cuda_event_ms; int valid;
} record;
static CUfunction kernel;
static CUdeviceptr program, control;
static const char *operation = "gather", *placement = "host", *layout = "random";
static const char *cubin = "die.cubin", *ids_path, *output_path;
static unsigned depth = 262144, iterations = 1, seed = 4242, warmup = 0;
static int gpu, profile;
static uint64_t fnv(uint64_t h, const void *p, size_t n) {
 const unsigned char *b = p;
 for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= FNV_PRIME; }
 return h;
}
static uint64_t hash_file(const char *path) {
 FILE *f = fopen(path, "rb"); if (!f) { perror(path); exit(2); }
 unsigned char b[65536]; size_t n; uint64_t h = FNV_OFFSET;
 while ((n = fread(b, 1, sizeof b, f))) h = fnv(h, b, n);
 if (ferror(f)) { perror(path); exit(2); }
 fclose(f); return h;
}
static void *alloc(size_t n) {
 void *p = calloc(1, n ? n : 1); if (!p) { perror("calloc"); exit(2); } return p;
}
static void fail(const char *s) { fprintf(stderr, "%s\n", s); exit(2); }
static unsigned number(const char *s) {
 char *end; errno = 0; unsigned long v = strtoul(s, &end, 10);
 if (errno || !*s || *end || *s == '-' || v > 0x7ffffffful) fail("invalid unsigned integer");
 return (unsigned)v;
}
static void quote(FILE *f, const char *s) {
 fputc('"', f);
 for (; *s; ++s) {
  unsigned char c = (unsigned char)*s;
  if (c == '"' || c == '\\') { fputc('\\', f); fputc(c, f); }
  else if (c < 32) fprintf(f, "\\u%04x", c); else fputc(c, f);
 }
 fputc('"', f);
}
static double now_ms(void) {
 struct timespec t; if (clock_gettime(CLOCK_MONOTONIC, &t)) { perror("clock_gettime"); exit(2); }
 return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}
static buffer device_buffer(size_t n) {
 buffer b = {0, NULL, n}; CK(cuMemAlloc(&b.d, n)); CK(cuMemsetD8(b.d, 0, n)); return b;
}
static buffer source_buffer(size_t n) {
 if (!strcmp(placement, "vram")) return device_buffer(n);
 buffer b = {0, NULL, n};
 CK(cuMemHostAlloc(&b.h, n, CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_PORTABLE));
 CK(cuMemHostGetDevicePointer(&b.d, b.h, 0)); memset(b.h, 0, n); return b;
}
static void release(buffer b) { if (b.h) CK(cuMemFreeHost(b.h)); else CK(cuMemFree(b.d)); }
static void upload(buffer b, const void *src, size_t n) {
 if (n > b.n) fail("internal upload exceeds allocation");
 if (b.h) memcpy(b.h, src, n); else CK(cuMemcpyHtoD(b.d, src, n));
}
static void init(descriptor *d, unsigned op, unsigned units) {
 memset(d, 0, sizeof *d); d->w[0] = op; d->w[1] = units;
}
static void ptr(descriptor *d, unsigned addr, CUdeviceptr p) {
 unsigned w = 2 + (addr - 0x140) / 4; d->w[w] = (uint32_t)p; d->w[w + 1] = (uint32_t)(p >> 32);
}
static void val(descriptor *d, unsigned addr, unsigned v) { d->w[2 + (addr - 0x140) / 4] = v; }
static void run(const descriptor *p, unsigned phases, unsigned position, record *r) {
 CK(cuMemcpyHtoD(program, p, phases * sizeof *p));
 /* Every finite launch starts from an independent local barrier generation. */
 CK(cuMemsetD8(control, 0, 8192)); CK(cuMemcpyHtoD(control + 8, &position, sizeof position));
 unsigned end = position + 1; void *args[] = { &program, &phases, &control, &end };
 CUevent start, stop; CK(cuEventCreate(&start, CU_EVENT_DEFAULT)); CK(cuEventCreate(&stop, CU_EVENT_DEFAULT));
 if (profile && r) CK(cuProfilerStart());
 double before = now_ms(); CK(cuEventRecord(start, NULL));
 CK(cuLaunchKernel(kernel, 16, 1, 1, 256, 1, 1, 0, NULL, args, NULL));
 CK(cuEventRecord(stop, NULL)); CK(cuEventSynchronize(stop));
 double elapsed = now_ms() - before;
 if (profile && r) CK(cuProfilerStop());
 float ms; CK(cuEventElapsedTime(&ms, start, stop));
 CK(cuEventDestroy(start)); CK(cuEventDestroy(stop));
 if (r) { r->host_launch_sync_ms = elapsed; r->cuda_event_ms = ms; }
}
static uint32_t mix(uint32_t x) {
 x ^= x >> 16; x *= UINT32_C(0x7feb352d); x ^= x >> 15; x *= UINT32_C(0x846ca68b); return x ^ (x >> 16);
}
static int cmp_unsigned(const void *a, const void *b) {
 unsigned x = *(const unsigned *)a, y = *(const unsigned *)b; return (x > y) - (x < y);
}
static int cmp_rank(const void *a, const void *b) {
 const rank_item *x = a, *y = b;
 if (x->score != y->score) return x->score > y->score ? -1 : 1;
 return (x->id > y->id) - (x->id < y->id);
}
static int8_t kv_value(size_t i, unsigned salt) {
 uint32_t x = mix((uint32_t)i ^ salt); int q = (int)(x % 127u) + 1; return (int8_t)((x & 128u) ? -q : q);
}
static void make_ids(unsigned *ids, unsigned nv, unsigned ns, unsigned it) {
 if (ids_path) {
  FILE *f = fopen(ids_path, "r"); if (!f) { perror(ids_path); exit(2); }
  for (unsigned i = 0; i < ns; ++i) {
   unsigned long v;
   if (fscanf(f, "%lu", &v) != 1 || v >= nv || (i && v <= ids[i - 1]))
    fail("IDs must have exact count, strictly increasing and within complete blocks");
   ids[i] = (unsigned)v;
  }
  int c; do { c = fgetc(f); } while (c == ' ' || c == '\t' || c == '\n' || c == '\r');
  if (c != EOF || ferror(f)) fail("extra or invalid content after selected IDs");
  fclose(f);
  if (nv <= BUDGET) for (unsigned i = 0; i < ns; ++i) if (ids[i] != i) fail("dense operation uses identity IDs");
  return;
 }
 if (nv <= BUDGET) { for (unsigned i = 0; i < ns; ++i) ids[i] = i; return; }
 unsigned shift = mix(seed + it) % nv;
 if (!strcmp(layout, "clustered")) {
  unsigned first = shift % (nv - ns + 1); for (unsigned i = 0; i < ns; ++i) ids[i] = first + i;
 } else if (!strcmp(layout, "even")) {
  for (unsigned i = 0; i < ns; ++i) ids[i] = ((uint64_t)i * nv / ns + shift) % nv;
 } else {
  unsigned *perm = alloc(nv * sizeof *perm); for (unsigned i = 0; i < nv; ++i) perm[i] = i;
  uint32_t state = seed ^ mix(it + 1);
  for (unsigned i = 0; i < ns; ++i) {
   state = state * 1664525u + 1013904223u;
   unsigned j = i + state % (nv - i), tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp; ids[i] = perm[i];
  }
  free(perm);
 }
 qsort(ids, ns, sizeof *ids, cmp_unsigned);
}
static void compact_cpu(const unsigned *ids, unsigned ns, unsigned nv, unsigned keys,
                        unsigned salt, float *kc, float *vc) {
 for (unsigned slot = 0; slot < keys; ++slot) {
  unsigned b = slot / 4, src = b < ns ? ids[b] : nv, pos = src * 4 + slot % 4;
  for (int h = 0; h < 2; ++h) for (int d = 0; d < 256; ++d) {
   size_t at = (size_t)pos * 512 + h * 256 + d;
   kc[attn_kidx(slot, h, d)] = kv_value(at, salt) * (1.0f / 128);
   vc[(size_t)slot * 512 + h * 256 + d] = kv_value(at, salt ^ 0xa5366b4du) * (1.0f / 64);
  }
 }
}
static void usage(FILE *f) {
 fprintf(f, "decode_counters --gpu N --cubin FILE --depth N --op gather|indexer|selection|attention\n"
            " --place host|vram --layout clustered|even|random|real [--ids FILE]\n"
            " --iterations N --warmup N --seed N --output FILE [--profile]\n"
            "Finite one-die operations only; caller verifies shared-host GPU idleness.\n");
}
int main(int argc, char **argv) {
 for (int a = 1; a < argc; ++a) {
  const char *k = argv[a];
  if (!strcmp(k, "--help")) { usage(stdout); return 0; }
  if (!strcmp(k, "--profile")) { profile = 1; continue; }
  if (++a == argc) { usage(stderr); return 2; }
  const char *v = argv[a];
  if (!strcmp(k, "--gpu")) gpu = (int)number(v);
  else if (!strcmp(k, "--cubin")) cubin = v;
  else if (!strcmp(k, "--depth")) depth = number(v);
  else if (!strcmp(k, "--op")) operation = v;
  else if (!strcmp(k, "--place")) placement = v;
  else if (!strcmp(k, "--layout")) layout = v;
  else if (!strcmp(k, "--ids")) ids_path = v;
  else if (!strcmp(k, "--iterations")) iterations = number(v);
  else if (!strcmp(k, "--warmup")) warmup = number(v);
  else if (!strcmp(k, "--seed")) seed = number(v);
  else if (!strcmp(k, "--output")) output_path = v;
  else { usage(stderr); return 2; }
 }
 if (!depth || depth > 262144 || !iterations || iterations > 1000 || warmup > 1000)
  fail("bounds: depth 1..262144, iterations 1..1000, warmup 0..1000");
 if (strcmp(placement, "host") && strcmp(placement, "vram")) fail("invalid placement");
 if (strcmp(layout, "clustered") && strcmp(layout, "even") && strcmp(layout, "random") && strcmp(layout, "real"))
  fail("invalid layout");
 if ((!strcmp(layout, "real")) != (ids_path != NULL)) fail("--layout real and --ids required together");
 int gather = !strcmp(operation, "gather"), indexer = !strcmp(operation, "indexer");
 int selection = !strcmp(operation, "selection"), attention = !strcmp(operation, "attention");
 if (!gather && !indexer && !selection && !attention) fail("invalid operation");
 if (!gather && strcmp(placement, "vram")) fail("nongather operations require --place vram");
 unsigned position = depth - 1, nv = depth / 4, ns = nv < BUDGET ? nv : BUDGET;
 unsigned keys = 4 * ns + depth % 4, padded = (keys + 8192 + 64 + 31) / 32 * 32;
 unsigned groups = (nv + 31) / 32 + 1, npos = (depth + 3) / 4 * 4;
 uint64_t cubin_hash = hash_file(cubin), input_hash = ids_path ? hash_file(ids_path) : 0;
 CK(cuInit(0)); CUdevice dev; CUcontext ctx; CUmodule mod;
 CK(cuDeviceGet(&dev, gpu)); CK(cuCtxCreate_v2(&ctx, CU_CTX_SCHED_SPIN, dev));
 char name[256], pci[32], hostname[256] = "unknown"; int major, minor, sms, driver;
 CK(cuDeviceGetName(name, sizeof name, dev)); CK(cuDeviceGetPCIBusId(pci, sizeof pci, dev));
 CK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
 CK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
 CK(cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
 CK(cuDriverGetVersion(&driver)); gethostname(hostname, sizeof hostname - 1);
 if (major != 5 || sms < 16) fail("production die requires Maxwell with at least 16 SMs");
 CK(cuModuleLoad(&mod, cubin)); CK(cuModuleGetFunction(&kernel, mod, "die"));
 buffer prog = device_buffer(128 * 8), ctl = device_buffer(8192);
 program = prog.d; control = ctl.d;
 record *records = alloc(iterations * sizeof *records);
 unsigned ids[BUDGET], gotids[BUDGET];
 buffer sel = device_buffer(BUDGET * 4), kc = {0}, vc = {0}, kq = {0}, vq = {0}, ks = {0}, vs = {0};
 buffer iq = {0}, pkt = {0}, sc = {0}, hist = {0}, qo = {0}, gate = {0}, pm = {0}, pd = {0}, po = {0}, out = {0};
 float *hkc = NULL, *hvc = NULL, *hsc = NULL;
 uint16_t *hpool = NULL; rank_item *rank = NULL;
 size_t kv_bytes = (size_t)npos * 512, scale_bytes = (size_t)npos * 32;
 size_t compact_bytes = (size_t)padded * 512 * sizeof(float), pool_bytes = (size_t)groups * 8192;
 unsigned salt = mix(seed);
 descriptor desc[5]; unsigned phases = 1;
 if (gather || attention) {
  kc = device_buffer(compact_bytes); vc = device_buffer(compact_bytes);
  hkc = alloc(compact_bytes); hvc = alloc(compact_bytes);
 }
 if (gather) {
  kq = source_buffer(kv_bytes); vq = source_buffer(kv_bytes);
  ks = source_buffer(scale_bytes); vs = source_buffer(scale_bytes);
  int8_t *q = alloc(kv_bytes); uint16_t *s = alloc(scale_bytes);
  for (size_t i = 0; i < kv_bytes; ++i) q[i] = kv_value(i, salt);
  upload(kq, q, kv_bytes);
  for (size_t i = 0; i < kv_bytes; ++i) q[i] = kv_value(i, salt ^ 0xa5366b4du);
  upload(vq, q, kv_bytes);
  for (size_t i = 0; i < scale_bytes / 2; ++i) s[i] = 0x2000; /* 1/128 */
  upload(ks, s, scale_bytes);
  for (size_t i = 0; i < scale_bytes / 2; ++i) s[i] = 0x2400; /* 1/64 */
  upload(vs, s, scale_bytes); free(q); free(s);
  init(&desc[0], OP_KVGATHER, BUDGET + 1);
  ptr(&desc[0], 0x140, kq.d); ptr(&desc[0], 0x148, ks.d); ptr(&desc[0], 0x150, vq.d); ptr(&desc[0], 0x158, vs.d);
  ptr(&desc[0], 0x160, kc.d); ptr(&desc[0], 0x168, vc.d); ptr(&desc[0], 0x170, sel.d);
  val(&desc[0], 0x178, position); val(&desc[0], 0x17c, BUDGET);
 }
 if (indexer || selection) {
  sc = device_buffer((nv + 64) * sizeof(float)); hist = device_buffer(1280 * 4);
  hsc = alloc((nv + 64) * sizeof(float));
 }
 if (indexer) {
  iq = device_buffer(512 * sizeof(float)); pkt = device_buffer(pool_bytes); hpool = alloc(pool_bytes);
  for (unsigned b = 0; b < nv; ++b) for (unsigned d = 0; d < 128; ++d) {
   size_t i = (size_t)(b / 32) * 4096 + (d / 4) * 128 + (b % 32) * 4 + d % 4;
   uint32_t v = mix(b * 128 + d + salt);
   hpool[i] = (uint16_t)(0x2800 | (v & 0x7ff) | ((v & 0x800) ? 0x8000 : 0));
  }
  upload(pkt, hpool, pool_bytes);
  init(&desc[0], OP_IDXSCORE, 16);
  ptr(&desc[0], 0x140, pkt.d); ptr(&desc[0], 0x148, iq.d); ptr(&desc[0], 0x150, sc.d); ptr(&desc[0], 0x158, hist.d);
  val(&desc[0], 0x160, position); val(&desc[0], 0x164, BUDGET);
 }
 if (selection) {
  rank = alloc((nv + 1) * sizeof *rank); phases = 5;
  for (unsigned p = 0; p < phases; ++p) {
   init(&desc[p], OP_IDXSEL, 16); ptr(&desc[p], 0x140, sc.d); ptr(&desc[p], 0x148, hist.d); ptr(&desc[p], 0x150, sel.d);
   val(&desc[p], 0x158, position); val(&desc[p], 0x15c, BUDGET); val(&desc[p], 0x160, 2 + p);
  }
 }
 if (attention) {
  qo = device_buffer(6144 * 4); gate = device_buffer(12288 * 4);
  pm = device_buffer(24 * 8 * 4); pd = device_buffer(24 * 8 * 4); po = device_buffer(24 * 8 * 256 * 4); out = device_buffer(6144 * 4);
  phases = 2; init(&desc[0], OP_ATTNPARTN, 32);
  ptr(&desc[0], 0x140, qo.d); ptr(&desc[0], 0x148, kc.d); ptr(&desc[0], 0x150, vc.d);
  ptr(&desc[0], 0x158, pm.d); ptr(&desc[0], 0x160, pd.d); ptr(&desc[0], 0x168, po.d);
  val(&desc[0], 0x170, depth); val(&desc[0], 0x174, BUDGET);
  init(&desc[1], OP_ATTNMERGE, 24); ptr(&desc[1], 0x140, pm.d); ptr(&desc[1], 0x148, pd.d); ptr(&desc[1], 0x150, po.d);
  ptr(&desc[1], 0x158, gate.d); ptr(&desc[1], 0x160, out.d);
 }
 int all_valid = 1;
 for (unsigned j = 0; j < warmup + iterations; ++j) {
  record *r = j < warmup ? NULL : &records[j - warmup]; unsigned it = j < warmup ? 0 : j - warmup;
  make_ids(ids, nv, ns, it); upload(sel, ids, ns * sizeof *ids);
  uint64_t id_hash = fnv(FNV_OFFSET, ids, ns * sizeof *ids); uint32_t hhist[1280] = {0};
  if (indexer) {
   float query[512];
   for (unsigned d = 0; d < 512; ++d) query[d] = kv_value(d, salt ^ mix(it + 9)) * (1.0f / 128);
   upload(iq, query, sizeof query); upload(hist, hhist, sizeof hhist);
  }
  if (selection) {
   for (unsigned b = 0; b < nv; ++b) hsc[b] = (mix(b ^ salt ^ mix(it)) % 65536) * (1.0f / 65536);
   for (unsigned b = 0; b < ns; ++b) hsc[ids[b]] += 2.0f;
   for (unsigned b = 0; b < nv; ++b) {
    uint32_t bits; memcpy(&bits, &hsc[b], 4); ++hhist[bits >> 23]; rank[b].score = hsc[b]; rank[b].id = b;
   }
   upload(sc, hsc, nv * sizeof *hsc); upload(hist, hhist, sizeof hhist);
  }
  if (attention) {
   compact_cpu(ids, ns, nv, keys, salt, hkc, hvc); upload(kc, hkc, compact_bytes); upload(vc, hvc, compact_bytes);
   float query[6144], gates[12288];
   for (unsigned d = 0; d < 6144; ++d) query[d] = kv_value(d, salt ^ mix(it + 37)) * (1.0f / 128);
   for (unsigned d = 0; d < 12288; ++d) gates[d] = kv_value(d, salt ^ mix(it + 73)) * (1.0f / 128);
   upload(qo, query, sizeof query); upload(gate, gates, sizeof gates);
  }
  run(desc, phases, position, r);
  uint64_t output_hash = FNV_OFFSET; int valid = 1;
  if (gather) {
   CK(cuMemcpyDtoH(hkc, kc.d, compact_bytes)); CK(cuMemcpyDtoH(hvc, vc.d, compact_bytes));
   for (unsigned slot = 0; slot < keys; ++slot) {
    unsigned b = slot / 4, src = b < ns ? ids[b] : nv, pos = src * 4 + slot % 4;
    for (int h = 0; h < 2; ++h) for (int d = 0; d < 256; ++d) {
     size_t at = (size_t)pos * 512 + h * 256 + d;
     float k = hkc[attn_kidx(slot, h, d)], v = hvc[(size_t)slot * 512 + h * 256 + d];
     valid &= k == kv_value(at, salt) * (1.0f / 128);
     valid &= v == kv_value(at, salt ^ 0xa5366b4du) * (1.0f / 64);
     output_hash = fnv(fnv(output_hash, &k, 4), &v, 4);
    }
   }
  } else if (indexer) {
   CK(cuMemcpyDtoH(hsc, sc.d, nv * sizeof *hsc)); CK(cuMemcpyDtoH(hhist, hist.d, sizeof hhist));
   unsigned total = 0; for (unsigned b = 0; b < 256; ++b) total += hhist[b];
   valid &= total == (nv > BUDGET ? nv : 0);
   for (unsigned b = 0; b < nv; ++b) valid &= isfinite(hsc[b]) && hsc[b] >= 0;
   output_hash = fnv(FNV_OFFSET, hsc, nv * sizeof *hsc);
  } else if (selection) {
   CK(cuMemcpyDtoH(gotids, sel.d, ns * sizeof *gotids)); qsort(rank, nv, sizeof *rank, cmp_rank);
   unsigned expected[BUDGET]; for (unsigned i = 0; i < ns; ++i) expected[i] = rank[i].id;
   qsort(expected, ns, sizeof *expected, cmp_unsigned);
   if (nv > BUDGET) valid &= !memcmp(expected, gotids, ns * sizeof *gotids);
   valid &= !memcmp(ids, gotids, ns * sizeof *gotids); output_hash = fnv(FNV_OFFSET, gotids, ns * sizeof *gotids);
  } else {
   float values[6144]; CK(cuMemcpyDtoH(values, out.d, sizeof values));
   for (unsigned d = 0; d < 6144; ++d) valid &= isfinite(values[d]);
   output_hash = fnv(FNV_OFFSET, values, sizeof values);
  }
  all_valid &= valid;
  if (r) {
   r->iteration = it; r->selected_count = ns; r->attended_keys = keys; r->tail_keys = depth % 4;
   r->ids_checksum = id_hash; r->output_checksum = output_hash; r->valid = valid;
  }
 }
 FILE *f = output_path ? fopen(output_path, "w") : stdout; if (!f) { perror(output_path); return 2; }
 fprintf(f, "{\n  \"schema\":\"flash.decode_counters.v1\",\n  \"revision\":"); quote(f, FLASH_COUNTERS_REV);
 fprintf(f, ",\n  \"hostname\":"); quote(f, hostname);
 fprintf(f, ",\n  \"gpu\":%d,\"device_name\":", gpu); quote(f, name);
 fprintf(f, ",\"pci_bus_id\":"); quote(f, pci);
 fprintf(f, ",\"compute_capability\":\"%d.%d\",\"sms\":%d,\"driver_api\":%d", major, minor, sms, driver);
 fprintf(f, ",\n  \"cubin\":"); quote(f, cubin);
 fprintf(f, ",\"cubin_fnv1a64\":\"%016" PRIx64 "\",\n  \"operation\":", cubin_hash); quote(f, operation);
 fprintf(f, ",\"placement\":"); quote(f, placement); fprintf(f, ",\"layout\":"); quote(f, layout);
 fprintf(f, ",\n  \"payload_source\":\"deterministic_nonzero_synthetic\",\"selected_ids_source\":");
 quote(f, ids_path ? "imported_checkpoint_frame" : "synthetic_layout");
 fprintf(f, ",\"ids_file\":"); if (ids_path) quote(f, ids_path); else fprintf(f, "null");
 fprintf(f, ",\"ids_file_fnv1a64\":\"%016" PRIx64 "\"", input_hash);
 fprintf(f, ",\n  \"depth\":%u,\"position\":%u,\"complete_blocks\":%u,\"block_budget\":%u,"
            "\"seed\":%u,\"warmup_launches\":%u,\"measured_launches\":%u", depth, position, nv, BUDGET, seed, warmup, iterations);
 fprintf(f, ",\n  \"launch\":{\"grid\":[16,1,1],\"block\":[256,1,1],\"phases\":%u,\"tokens\":1},"
            "\n  \"profile_enabled\":%s,\"timings_include_profiler_overhead\":%s,"
            "\n  \"counter_scope\":\"finite operation plus production die dispatch and local barriers; no peer or host wait\","
            "\n  \"validation\":", phases, profile ? "true" : "false", profile ? "true" : "false");
 quote(f, gather ? "all attended K/V values exact against independently computed source pattern" :
          selection ? "all selected IDs exact against CPU top-k ordering" :
          indexer ? "histogram count and finite nonnegative scores" : "finite output only");
 fprintf(f, ",\n  \"allocated_source_bytes\":%zu,\"source_is_mapped_host\":%s,"
            "\n  \"all_valid\":%s,\n  \"records\":[",
            gather ? 2 * (kv_bytes + scale_bytes) : indexer ? pool_bytes : 0,
            gather && !strcmp(placement, "host") ? "true" : "false", all_valid ? "true" : "false");
 for (unsigned i = 0; i < iterations; ++i) {
  record *r = &records[i];
  fprintf(f, "%s\n    {\"iteration\":%u,\"selected_count\":%u,\"attended_keys\":%u,\"tail_keys\":%u,"
             "\"ids_fnv1a64\":\"%016" PRIx64 "\",\"output_fnv1a64\":\"%016" PRIx64 "\","
             "\"host_launch_sync_ms\":%.6f,\"cuda_event_ms\":%.6f,\"valid\":%s}",
             i ? "," : "", r->iteration, r->selected_count, r->attended_keys, r->tail_keys,
             r->ids_checksum, r->output_checksum, r->host_launch_sync_ms, r->cuda_event_ms, r->valid ? "true" : "false");
 }
 fprintf(f, "\n  ]\n}\n");
 if (ferror(f) || (output_path && fclose(f))) { perror("output"); return 2; }
 buffer buffers[] = {sel,kc,vc,kq,vq,ks,vs,iq,pkt,sc,hist,qo,gate,pm,pd,po,out,prog,ctl};
 for (size_t i = 0; i < sizeof buffers / sizeof *buffers; ++i) if (buffers[i].d) release(buffers[i]);
 free(records); free(hkc); free(hvc); free(hsc); free(hpool); free(rank);
 CK(cuModuleUnload(mod)); CK(cuCtxDestroy(ctx)); return all_valid ? 0 : 1;
}
