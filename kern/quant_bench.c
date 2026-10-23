/* Paired OP_ATTNQKQ benchmark. One persistent launch repeats a fixed-position
 * descriptor. Event intervals exclude initialization, copies and validation.
 * The complete operation includes RMSNorm, RoPE, quantization, dispatch and
 * synchronization; paired differences estimate incremental clamp/trace cost.
 * This is not a whole-model performance measurement. */
#include <cuda.h>
#include "dieops.h"
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef FLASH_REV
#define FLASH_REV "unknown"
#endif
CUresult cuCtxCreate_v2(CUcontext *, unsigned int, CUdevice);
#define CK(call) do { CUresult r_ = (call); if (r_ != CUDA_SUCCESS) { \
 const char *s_ = "?"; cuGetErrorString(r_, &s_); \
 fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call, s_); exit(1); } } while (0)

enum { BOUNDED, UNBOUNDED, TRACE, NVARIANTS };
enum { NQG=12288, NQO=6144, NKV=512, NWEIGHT=256, NROT=32, NSCALE=16, NRAW=32, CTL_BYTES=8192 };
typedef struct { uint32_t w[32]; } desc_t;
typedef struct { CUdeviceptr device; void *host; size_t bytes; } buffer_t;
typedef struct {
 float qo[NQO];
 int8_t kq[NKV], vq[NKV];
 uint16_t ks[NSCALE], vs[NSCALE];
 float raw[NRAW];
} snapshot_t;
typedef struct {
 buffer_t qg, k, v, qn, kn, cs, sn, qo, kq, vq, ks, vs, raw;
 buffer_t program[NVARIANTS], control;
 CUmodule module[NVARIANTS]; CUfunction function[NVARIANTS];
 CUevent start, stop;
} bench_t;
static const char *variant_name[NVARIANTS] = {"bounded", "unbounded", "trace"};
static void fail(const char *s) { fprintf(stderr, "quant_bench: %s\n", s); exit(1); }
static void *allocate(size_t n, size_t size) {
 void *p = calloc(n, size); if (!p) fail("host allocation failed"); return p;
}
static buffer_t buffer_alloc(size_t bytes, int mapped) {
 buffer_t b = {0}; b.bytes = bytes;
 if (mapped) {
  CK(cuMemHostAlloc(&b.host, bytes, CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_PORTABLE));
  CK(cuMemHostGetDevicePointer(&b.device, b.host, 0)); memset(b.host, 0, bytes);
 } else { CK(cuMemAlloc(&b.device, bytes)); CK(cuMemsetD8(b.device, 0, bytes)); }
 return b;
}
static void buffer_write(const buffer_t *b, const void *src) {
 if (b->host) memcpy(b->host, src, b->bytes);
 else CK(cuMemcpyHtoD(b->device, src, b->bytes));
}
static void buffer_read(const buffer_t *b, void *dst) {
 if (b->host) memcpy(dst, b->host, b->bytes);
 else CK(cuMemcpyDtoH(dst, b->device, b->bytes));
}
static void buffer_fill(const buffer_t *b, unsigned char v) {
 if (b->host) memset(b->host, v, b->bytes);
 else CK(cuMemsetD8(b->device, v, b->bytes));
}
static void buffer_free(buffer_t *b) {
 if (b->host) CK(cuMemFreeHost(b->host));
 else if (b->device) CK(cuMemFree(b->device));
 memset(b, 0, sizeof *b);
}
static void desc_ptr(desc_t *d, unsigned addr, CUdeviceptr p) {
 unsigned w = 2 + (addr - 0x140) / 4; d->w[w] = (uint32_t)p; d->w[w+1] = (uint32_t)(p >> 32);
}
static desc_t make_descriptor(const bench_t *b, int variant) {
 desc_t d = {{0}}; d.w[0] = OP_ATTNQKQ;
 d.w[1] = 26; /* Work units: 24 query heads + 2 KV heads, not parameter words. */
 desc_ptr(&d, 0x140, b->qg.device); desc_ptr(&d, 0x148, b->k.device);
 desc_ptr(&d, 0x150, b->v.device); desc_ptr(&d, 0x158, b->qn.device);
 desc_ptr(&d, 0x160, b->kn.device); desc_ptr(&d, 0x168, b->cs.device);
 desc_ptr(&d, 0x170, b->sn.device); desc_ptr(&d, 0x178, b->qo.device);
 desc_ptr(&d, 0x180, b->kq.device); desc_ptr(&d, 0x188, b->vq.device);
 float eps = 1e-6f; memcpy(&d.w[2 + (0x194 - 0x140) / 4], &eps, sizeof eps);
 desc_ptr(&d, 0x198, b->ks.device); desc_ptr(&d, 0x1a0, b->vs.device);
 if (variant == TRACE) desc_ptr(&d, 0x1a8, b->raw.device);
 /* param_t=0 and w[31]=0: fixed position and pointers for every iteration. */
 return d;
}
static void initialize_fixture(bench_t *b, float k[NKV], float v[NKV]) {
 float *qg = allocate(NQG, sizeof *qg), weight[NWEIGHT], cs[NROT], sn[NROT];
 for (int i=0; i<NQG; ++i) qg[i] = (float)((i*13+17)%101-50)/32.f;
 for (int i=0; i<NKV; ++i) {
  k[i] = (float)((i*17+11)%97-48)/16.f;
  v[i] = (float)((i*29+7)%113-56)/8.f;
 }
 for (int i=0; i<NWEIGHT; ++i) weight[i] = 1.f;
 for (int i=0; i<NROT; ++i) { cs[i] = 1.f; sn[i] = 0.f; }
 buffer_write(&b->qg, qg); buffer_write(&b->k, k); buffer_write(&b->v, v);
 buffer_write(&b->qn, weight); buffer_write(&b->kn, weight);
 buffer_write(&b->cs, cs); buffer_write(&b->sn, sn); free(qg);
}
static void reset_outputs(bench_t *b) {
 CK(cuMemsetD32(b->qo.device, 0x7fc00000u, NQO));
 buffer_fill(&b->kq, 0x80); buffer_fill(&b->vq, 0x80);
 buffer_fill(&b->ks, 0xff); buffer_fill(&b->vs, 0xff); buffer_fill(&b->raw, 0xff);
}
static void read_snapshot(const bench_t *b, snapshot_t *s, int trace) {
 memset(s, 0, sizeof *s);
 buffer_read(&b->qo, s->qo); buffer_read(&b->kq, s->kq); buffer_read(&b->vq, s->vq);
 buffer_read(&b->ks, s->ks); buffer_read(&b->vs, s->vs);
 if (trace) buffer_read(&b->raw, s->raw);
}
static float half_value(uint16_t h) {
 unsigned e = (h >> 10) & 31u, f = h & 1023u;
 float x = !e ? ldexpf((float)f, -24) :
   e == 31u ? (f ? NAN : INFINITY) : ldexpf((float)(1024u+f), (int)e-25);
 return h & 0x8000u ? -x : x;
}
static void compare_snapshot(const snapshot_t *expected, const snapshot_t *actual, int variant, int sample) {
 if (memcmp(expected->qo, actual->qo, sizeof expected->qo) ||
     memcmp(expected->kq, actual->kq, sizeof expected->kq) ||
     memcmp(expected->vq, actual->vq, sizeof expected->vq) ||
     memcmp(expected->ks, actual->ks, sizeof expected->ks) ||
     memcmp(expected->vs, actual->vs, sizeof expected->vs)) {
  fprintf(stderr, "ordinary output mismatch: variant=%s sample=%d\n", variant_name[variant], sample); exit(1);
 }
}
/* Independent moderate-range check. K normalization uses double precision;
 * identity rotation and unit weights make the expected input explicit.
 * Tolerance includes integer rounding and FP16 scale rounding. */
static void check_vector(const char *name, const float input[NKV],
                         const int8_t code[NKV], const uint16_t scale[NSCALE]) {
 for (int group=0; group<NSCALE; ++group) {
  double maximum=0, s=half_value(scale[group]);
  for (int j=0; j<32; ++j) {
   double a=fabs((double)input[group*32+j]); if (a>maximum) maximum=a;
  }
  double expected=maximum/127., tolerance=expected/1024.+0x1p-24;
  if (!(s>0) || !isfinite(s) || fabs(s-expected)>tolerance) {
   fprintf(stderr, "%s scale mismatch group=%d observed=%.9g expected=%.9g\n", name, group, s, expected); exit(1);
  }
  double error=0.51*s+maximum/1024.+1e-5;
  for (int j=0; j<32; ++j) {
   int i=group*32+j;
   if (code[i]==INT8_MIN || fabs((double)code[i]*s-input[i])>error) {
    fprintf(stderr, "%s reconstruction mismatch component=%d\n", name, i); exit(1);
   }
  }
 }
}
static void check_reference(const snapshot_t *s, const float k[NKV],
                            const float v[NKV], float normalized[NKV]) {
 for (int i=0; i<NQO; ++i) if (!isfinite(s->qo[i])) fail("nonfinite/unwritten Q output");
 for (int h=0; h<2; ++h) {
  double sum=0;
  for (int j=0; j<256; ++j) { double x=k[h*256+j]; sum+=x*x; }
  double inv=1./sqrt(sum/256.+1e-6);
  for (int j=0; j<256; ++j) normalized[h*256+j]=(float)(k[h*256+j]*inv);
 }
 check_vector("K", normalized, s->kq, s->ks); check_vector("V", v, s->vq, s->vs);
}
static void check_trace(const snapshot_t *s, const float normalized[NKV], const float v[NKV]) {
 for (int kind=0; kind<2; ++kind) for (int g=0; g<NSCALE; ++g) {
  const float *input=kind?v:normalized; double expected=0;
  for (int j=0; j<32; ++j) { double a=fabs((double)input[g*32+j]); if (a>expected) expected=a; }
  double got=s->raw[kind*NSCALE+g];
  if (!(got>0) || !isfinite(got) || fabs(got-expected)>2e-5*expected+1e-6) {
   fprintf(stderr, "raw %c maximum mismatch group=%d observed=%.9g expected=%.9g\n",
           kind?'V':'K', g, got, expected); exit(1);
  }
 }
}
static float launch(bench_t *b, int variant, unsigned iterations, int timed) {
 unsigned nph=1; CUdeviceptr program=b->program[variant].device, control=b->control.device;
 void *parameters[]={&program, &nph, &control, &iterations};
 CK(cuMemsetD8(control, 0, CTL_BYTES));
 float ms = 0;
 if (timed) CK(cuEventRecord(b->start, 0));
 CK(cuLaunchKernel(b->function[variant], 16,1,1, 256,1,1, 0,0, parameters,NULL));
 if (timed) {
  CK(cuEventRecord(b->stop, 0)); CK(cuEventSynchronize(b->stop));
  CK(cuEventElapsedTime(&ms, b->start, b->stop));
  if (!(ms>0) || !isfinite(ms)) fail("invalid event interval");
 } else {
  CK(cuCtxSynchronize());
 }
 unsigned completed = 0;
 CK(cuMemcpyDtoH(&completed, control + 8, sizeof completed));
 if (completed != iterations) fail("persistent loop did not complete every requested operation");
 return ms;
}
static unsigned number(const char *s, unsigned lo, unsigned hi, const char *option) {
 char *end; errno=0; unsigned long n=strtoul(s, &end, 10);
 if (!*s || *s=='-' || errno || *end || n<lo || n>hi) {
  fprintf(stderr, "invalid %s: %s\n", option, s); exit(2);
 }
 return (unsigned)n;
}
static int compare_double(const void *a, const void *b) {
 double x=*(const double *)a, y=*(const double *)b; return (x>y)-(x<y);
}
static double median(const double *data, unsigned n) {
 double *copy=allocate(n, sizeof *copy); memcpy(copy, data, n*sizeof *copy);
 qsort(copy, n, sizeof *copy, compare_double);
 double m=n&1u ? copy[n/2] : (copy[n/2-1]+copy[n/2])/2.; free(copy); return m;
}
int main(int argc, char **argv) {
 const char *path[NVARIANTS]={0}; unsigned iterations=10000, samples=12; int host=0;
 for (int i=1; i<argc; ++i) {
  if (!strcmp(argv[i],"--help")) {
   puts("quant_bench --bounded FILE --unbounded FILE --trace FILE [--iterations N] [--samples N] [--host 0|1]"); return 0;
  }
  if (i+1>=argc) fail("option requires a value");
  const char *option=argv[i], *value=argv[++i];
  if (!strcmp(option,"--bounded")) path[BOUNDED]=value;
  else if (!strcmp(option,"--unbounded")) path[UNBOUNDED]=value;
  else if (!strcmp(option,"--trace")) path[TRACE]=value;
  else if (!strcmp(option,"--iterations")) iterations=number(value,1,1000000,option);
  else if (!strcmp(option,"--samples")) samples=number(value,1,1000,option);
  else if (!strcmp(option,"--host")) host=(int)number(value,0,1,option);
  else fail("unknown option");
 }
 for (int v=0; v<NVARIANTS; ++v) if (!path[v]) fail("all three cubin paths are required");
 CUdevice device; CUcontext context; char name[256]; int driver, major, minor;
 CK(cuInit(0)); CK(cuDeviceGet(&device,0)); CK(cuDeviceGetName(name,sizeof name,device));
 CK(cuDeviceGetAttribute(&major,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,device));
 CK(cuDeviceGetAttribute(&minor,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,device));
 CK(cuDriverGetVersion(&driver)); CK(cuCtxCreate_v2(&context, CU_CTX_MAP_HOST, device));
 bench_t b={0};
 b.qg=buffer_alloc(NQG*4,0); b.k=buffer_alloc(NKV*4,0); b.v=buffer_alloc(NKV*4,0);
 b.qn=buffer_alloc(NWEIGHT*4,0); b.kn=buffer_alloc(NWEIGHT*4,0);
 b.cs=buffer_alloc(NROT*4,0); b.sn=buffer_alloc(NROT*4,0); b.qo=buffer_alloc(NQO*4,0);
 b.kq=buffer_alloc(NKV,host); b.vq=buffer_alloc(NKV,host);
 b.ks=buffer_alloc(NSCALE*2,host); b.vs=buffer_alloc(NSCALE*2,host);
 b.raw=buffer_alloc(NRAW*4,1); b.control=buffer_alloc(CTL_BYTES,0);
 CK(cuEventCreate(&b.start,0)); CK(cuEventCreate(&b.stop,0));
 for (int v=0; v<NVARIANTS; ++v) {
  CK(cuModuleLoad(&b.module[v],path[v])); CK(cuModuleGetFunction(&b.function[v],b.module[v],"die"));
  b.program[v]=buffer_alloc(sizeof(desc_t),0);
  desc_t d=make_descriptor(&b,v); buffer_write(&b.program[v],&d);
 }
 float k[NKV], v_input[NKV], normalized[NKV]; snapshot_t expected, actual;
 initialize_fixture(&b,k,v_input); reset_outputs(&b); launch(&b,BOUNDED,1,0);
 read_snapshot(&b,&expected,0); check_reference(&expected,k,v_input,normalized);
 for (int v=UNBOUNDED; v<NVARIANTS; ++v) {
  reset_outputs(&b); launch(&b,v,1,0); read_snapshot(&b,&actual,v==TRACE);
  compare_snapshot(&expected,&actual,v,-1); if (v==TRACE) check_trace(&actual,normalized,v_input);
 }
 unsigned warm=iterations<1000?iterations:1000;
 for (int round=0; round<2; ++round) for (int v=0; v<NVARIANTS; ++v)
  launch(&b,round?NVARIANTS-1-v:v,warm,0);
 printf("# source_revision=%s\n# device=%s,compute_capability=%d.%d,driver_api=%d\n", FLASH_REV,name,major,minor,driver);
 printf("# kv_location=%s,trace_location=host_mapped\n",host?"host_mapped":"device");
 for (int v=0; v<NVARIANTS; ++v) printf("# %s=%s\n",variant_name[v],path[v]);
 printf("# operation=complete_OP_ATTNQKQ_including_dispatch_and_sync\n");
 printf("# fixed_position=0,ctas=16,threads_per_cta=256,work_units=26,descriptors=1,operations_per_launch=%u\n",iterations);
 printf("# warmup_rounds=2,warmup_operations_per_variant_per_round=%u\n",warm);
 printf("# independent_ordinary_KV_check=PASS,preflight_bitwise_agreement=PASS,raw_maxima_check=PASS\n");
 puts("sample,order,variant,host,iterations,total_ms,us_per_operation");
 static const int order[6][3]={{BOUNDED,UNBOUNDED,TRACE},{TRACE,UNBOUNDED,BOUNDED},
  {UNBOUNDED,TRACE,BOUNDED},{BOUNDED,TRACE,UNBOUNDED},{TRACE,BOUNDED,UNBOUNDED},{UNBOUNDED,BOUNDED,TRACE}};
 double *us[NVARIANTS];
 for (int v=0; v<NVARIANTS; ++v) us[v]=allocate(samples,sizeof(double));
 for (unsigned sample=0; sample<samples; ++sample) for (int slot=0; slot<NVARIANTS; ++slot) {
  int v=order[sample%6][slot]; float ms=launch(&b,v,iterations,1);
  us[v][sample]=(double)ms*1000./iterations;
  read_snapshot(&b,&actual,v==TRACE); compare_snapshot(&expected,&actual,v,(int)sample);
  if (v==TRACE) check_trace(&actual,normalized,v_input);
  printf("%u,%d,%s,%d,%u,%.9g,%.9g\n",sample,slot,variant_name[v],host,iterations,(double)ms,us[v][sample]); fflush(stdout);
 }
 double *clamp=allocate(samples,sizeof(double)), *trace=allocate(samples,sizeof(double));
 double *clamp_pct=allocate(samples,sizeof(double)), *trace_pct=allocate(samples,sizeof(double));
 for (unsigned s=0; s<samples; ++s) {
  clamp[s]=us[BOUNDED][s]-us[UNBOUNDED][s]; trace[s]=us[TRACE][s]-us[BOUNDED][s];
  clamp_pct[s]=100.*clamp[s]/us[UNBOUNDED][s]; trace_pct[s]=100.*trace[s]/us[BOUNDED][s];
  printf("# pair=%u,clamp_delta_us=%.9g,clamp_delta_percent=%.9g,trace_delta_us=%.9g,trace_delta_percent=%.9g\n",
         s,clamp[s],clamp_pct[s],trace[s],trace_pct[s]);
 }
 for (int v=0; v<NVARIANTS; ++v) printf("# median_us_per_operation,%s,%.9g\n",variant_name[v],median(us[v],samples));
 printf("# median_paired_clamp_delta_us=%.9g,median_paired_clamp_delta_percent=%.9g\n",median(clamp,samples),median(clamp_pct,samples));
 printf("# median_paired_trace_delta_us=%.9g,median_paired_trace_delta_percent=%.9g\n",median(trace,samples),median(trace_pct,samples));
 printf("# measured_launch_output_checks=%u,all_bitwise_agreement=PASS\n",samples*NVARIANTS);
 puts("# note=negative_deltas_retained;no_claim_of_statistical_significance");
 free(clamp); free(trace); free(clamp_pct); free(trace_pct);
 for (int v=0; v<NVARIANTS; ++v) { free(us[v]); buffer_free(&b.program[v]); CK(cuModuleUnload(b.module[v])); }
 CK(cuEventDestroy(b.start)); CK(cuEventDestroy(b.stop));
 buffer_free(&b.qg); buffer_free(&b.k); buffer_free(&b.v); buffer_free(&b.qn); buffer_free(&b.kn);
 buffer_free(&b.cs); buffer_free(&b.sn); buffer_free(&b.qo); buffer_free(&b.kq); buffer_free(&b.vq);
 buffer_free(&b.ks); buffer_free(&b.vs); buffer_free(&b.raw); buffer_free(&b.control);
 CK(cuCtxDestroy(context)); return 0;
}
