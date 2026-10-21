/* GPU kernel regression: strided tensors, key/query tails, large logits,
 * output guards, and sampled float64 softmax references across every head. */
#include "../../kern/vis_fa.h"
#include "ggml.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CK(call) do { CUresult rc = (call); if (rc) { fprintf(stderr, "%s: CUDA %d\n", #call, rc); exit(1); } } while (0)
static uint32_t rng = 96;
static float random_value(void) { rng = rng * 1664525u + 1013904223u; return ((int)(rng >> 16) - 32768) / 32768.0f; }

static void check(unsigned nq, unsigned nk) {
    size_t qc = (size_t)nq * 16 * 72, kc = (size_t)nk * 16 * 72;
    float *q = malloc(qc * 4), *out = malloc(qc * 4);
    ggml_fp16_t *k = malloc(kc * 2), *v = malloc(kc * 2);
    if (!q || !out || !k || !v) exit(1);
    for (size_t i = 0; i < qc; i++) q[i] = random_value();
    for (unsigned i = 0; i < 72; i++) q[i] *= 1024; /* unstable softmax would overflow */
    for (size_t i = 0; i < kc; i++) { k[i] = ggml_fp32_to_fp16(random_value()); v[i] = ggml_fp32_to_fp16(random_value()); }
    CUdeviceptr dq, dk, dv, base;
    CK(cuMemAlloc(&dq, qc * 4)); CK(cuMemAlloc(&dk, kc * 2)); CK(cuMemAlloc(&dv, kc * 2));
    CK(cuMemAlloc(&base, qc * 4 + 512)); CK(cuMemsetD8(base, 0xa5, qc * 4 + 512));
    CK(cuMemcpyHtoD(dq, q, qc * 4)); CK(cuMemcpyHtoD(dk, k, kc * 2)); CK(cuMemcpyHtoD(dv, v, kc * 2));
    struct vis_fa_desc desc = {.q=dq,.k=dk,.v=dv,.o=base+256,
        .qs={4,72*16*4,72*4},.ks={2,72*2,(uint64_t)72*nk*2},
        .vs={2,72*2,(uint64_t)72*nk*2},.os={4,72*16*4,72*4},
        .nq=nq,.nk=nk,.heads=16,.dim=72,.scale=1.0f/sqrtf(72)};
    struct vis_fa_stats stats;
    if (vis_fa_execute(&desc, &stats)) exit(1);
    CK(cuMemcpyDtoH(out, base+256, qc*4));
    unsigned char guard[512];
    CK(cuMemcpyDtoH(guard, base, 256)); CK(cuMemcpyDtoH(guard+256, base+256+qc*4, 256));
    for (unsigned i=0;i<512;i++) if (guard[i]!=0xa5) { fprintf(stderr,"output guard corrupted\n");exit(1); }
    for (size_t i=0;i<qc;i++) if (!isfinite(out[i])) { fprintf(stderr,"non-finite output at %zu\n",i);exit(1); }
    double max_error=0;
    const unsigned sample[] = {0,nq > 1 ? 1 : 0,nq/2,nq-1};
    for (unsigned h=0;h<16;h++) for (unsigned s=0;s<4;s++) {
        unsigned qi=sample[s];
        double *scores=malloc((size_t)nk*sizeof *scores), maximum=-INFINITY, sum=0, reference[72]={0};
        if (!scores) exit(1);
        for (unsigned j=0;j<nk;j++) {
            double dot=0;
            for (unsigned d=0;d<72;d++) dot+=(double)q[((size_t)qi*16+h)*72+d]*ggml_fp16_to_fp32(k[((size_t)h*nk+j)*72+d]);
            scores[j]=dot*desc.scale; if(scores[j]>maximum)maximum=scores[j];
        }
        for (unsigned j=0;j<nk;j++) {
            double p=exp(scores[j]-maximum);sum+=p;
            for(unsigned d=0;d<72;d++)reference[d]+=p*ggml_fp16_to_fp32(v[((size_t)h*nk+j)*72+d]);
        }
        for(unsigned d=0;d<72;d++) {
            double error=fabs(out[((size_t)qi*16+h)*72+d]-reference[d]/sum);
            if(error>max_error)max_error=error;
        }
        free(scores);
    }
    printf("{\"event\":\"fa_kernel_reference\",\"queries\":%u,\"keys\":%u,\"sampled_queries_per_head\":4,\"heads\":16,\"max_abs_error\":%.9g,\"kernel_ms\":%.3f,\"workspace_bytes\":%llu,\"guards\":\"pass\"}\n",
           nq,nk,max_error,stats.elapsed_ms,(unsigned long long)stats.workspace_bytes);
    fflush(stdout);
    if(max_error>0.0002) { fprintf(stderr,"GPU FA differs from float64 reference\n");exit(1); }
    CK(cuMemFree(dq));CK(cuMemFree(dk));CK(cuMemFree(dv));CK(cuMemFree(base));
    free(q);free(k);free(v);free(out);
}

int main(void) {
    if (!getenv("GPUQ_JOB_ID")) { fprintf(stderr,"run through gpuq\n");return 2; }
    CK(cuInit(0));
    CUdevice dev;CUcontext context;
    CK(cuDeviceGet(&dev,6)); CK(cuCtxCreate(&context,0,dev));
    check(1,1);check(13,35);check(64,64);check(2408,2408);check(13,65536);
    CK(cuCtxDestroy(context));
    return 0;
}
