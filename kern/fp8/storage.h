/* FP8 is authoritative between residual-producing operations. Existing GR
 * math uses transient FP32 workspaces; every read is refreshed from FP8.
 * This first implementation retains those workspaces, so it does not claim
 * that total allocated activation memory falls by four times. */
#ifdef FLASH_CHUNK_NATIVE
#include "chunk_ops.h"
#endif
static size_t fp8_bytes(unsigned n) { return (size_t)n+4u*(n/256u); }
static void fp8_conversion(prog_t *p,unsigned op,CUdeviceptr src,CUdeviceptr dst,unsigned n) {
    phase_t *a=prog_add(p,(int)op,n/256u);ph_ptr(a,0x140,src);ph_ptr(a,0x148,dst);ph_u32(a,0x150,n);
}
static int fp8_destination(const phase_t *a,int chunk,int *inplace) {
    *inplace=0;
#ifdef FLASH_CHUNK_NATIVE
    if(chunk) {
        if(a->w[0]==OP_SLAB_WRITE) { *inplace=1;return 2; }
        if(a->w[0]==OP_SLAB_PCONV) { *inplace=1;return 12; }
        return -1;
    }
#else
    (void)chunk;
#endif
    switch(a->w[0]) {
    case OP_EMBROW:case OP_XCOPY:return 4;
    case OP_HCWRITE:*inplace=1;return 6;
    case OP_HCWSUM:return 6;
    case OP_PLECONV:*inplace=1;return 10;
    case OP_BF16HCRWX:case 94:return 10;
    default:return -1;
    }
}
static void fp8_program(prog_t *p,const CUdeviceptr *work,const CUdeviceptr *encoded,unsigned count,unsigned n,int chunk,int *layers) {
    if(!residual_fp8)return;
    unsigned pack=chunk?119u:95u,unpack=pack+1;prog_t out;prog_init(&out);
    int *map=layers?malloc((size_t)(p->n*5+8)*sizeof *map):NULL;
    if(chunk)for(unsigned b=0;b<count;b++)fp8_conversion(&out,pack,work[b],encoded[b],n);
    for(int i=0;i<p->n;i++) {
        phase_t original=p->p[i];int inplace=0,dst=fp8_destination(&original,chunk,&inplace),first=out.n;
        for(unsigned b=0;b<count;b++) {
            int reads=0;
            for(int w=2;w<29;w++) {
                uint64_t pointer;memcpy(&pointer,&original.w[w],8);
                if(pointer==work[b] && (w!=dst || inplace))reads=1;
            }
            if(reads)fp8_conversion(&out,unpack,encoded[b],work[b],n);
        }
        phase_t *a=prog_add(&out,(int)original.w[0],original.w[1]);*a=original;
        if(dst>=0)for(unsigned b=0;b<count;b++) {
            uint64_t pointer;memcpy(&pointer,&original.w[dst],8);
            if(pointer==work[b]) { a->w[30]&=~4u;fp8_conversion(&out,pack,work[b],encoded[b],n);break; }
        }
        if(map)for(int j=first;j<out.n;j++)map[j]=layers[i];
    }
    /* The host draft/feature handoff consumes the final decoded workspace. */
    int before=out.n;for(unsigned b=0;b<count;b++)fp8_conversion(&out,unpack,encoded[b],work[b],n);
    if(map) {
        for(int j=before;j<out.n;j++)map[j]=p->n?layers[p->n-1]:-1;
        if(out.n>4096) { fprintf(stderr,"engine: FP8 program exceeds profiling capacity\n");longjmp(run_jb,1); }
        memcpy(layers,map,(size_t)out.n*sizeof *map);free(map);
    }
    free(p->p);p->p=out.p;p->n=out.n;p->cap=out.cap;
}
