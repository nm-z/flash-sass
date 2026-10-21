/* Export fixed-input attention experiments through the production SASS ops.
 * These deterministic projection fixtures have no checkpoint provenance and
 * use supplied block IDs. The production indexer is deliberately not invoked.
 * Pair the captures with qsa_attention_quality's independent FP64 equations. */
#define _POSIX_C_SOURCE 200809L
#include <cuda.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include "dieops.h"

#ifndef FLASH_REV
#define FLASH_REV "unknown"
#endif
CUresult cuCtxCreate_v2(CUcontext *, unsigned int, CUdevice);
#define CK(call) do { CUresult r_=(call); if (r_!=CUDA_SUCCESS) { \
    const char *s_="?"; cuGetErrorString(r_,&s_); \
    fprintf(stderr,"%s:%d: %s: %s\n",__FILE__,__LINE__,#call,s_); exit(1); \
} } while (0)

typedef struct { uint32_t w[32]; } desc_t;
static const char *root;
static const char *frozen_root;
static const char *weights_revision;
static unsigned lengths[32], nlengths;
static uint32_t rng=4242;
static void fail(const char *why) { fprintf(stderr,"qsa_attention_gpu: %s\n",why); exit(2); }
static int parse_unsigned(const char *s, unsigned lo, unsigned hi, unsigned *out) {
    if (!s || !*s) return 1;
    unsigned v=0;
    for (;*s;s++) {
        if (*s<'0' || *s>'9') return 1;
        unsigned digit=(unsigned)(*s-'0');
        if (v>hi/10 || (v==hi/10 && digit>hi%10)) return 1;
        v=v*10+digit;
    }
    if (v<lo) return 1;
    *out=v; return 0;
}
static void *alloc(size_t bytes) {
    void *p=malloc(bytes ? bytes : 1);
    if (!p) fail("host allocation failed");
    return p;
}
static float random_float(void) {
    rng=rng*1664525u+1013904223u;
    return (float)((rng>>8)&65535u)/32768.0f-1.0f;
}
static void path(char *p, size_t cap, const char *leaf) {
    int n=snprintf(p,cap,"%s/%s",root,leaf);
    if (n<0 || (size_t)n>=cap) fail("path too long");
}
static void input_f32(const char *leaf, float *dst, size_t count) {
    char p[4096];
    int n=snprintf(p,sizeof p,"%s/%s",frozen_root,leaf);
    if (n<0 || (size_t)n>=sizeof p) fail("input path too long");
    int fd=open(p,O_RDONLY|O_NOFOLLOW);
    if (fd<0) fail("cannot open frozen projection input");
    struct stat st;
    if (fstat(fd,&st) || !S_ISREG(st.st_mode) || st.st_size!=(off_t)(count*sizeof *dst))
        fail("frozen projection input has wrong file type or size");
    size_t got=0;
    while (got<count*sizeof *dst) {
        ssize_t nr=read(fd,(unsigned char *)dst+got,count*sizeof *dst-got);
        if (nr<0 && errno==EINTR) continue;
        if (nr<=0) fail("frozen projection input read failed");
        got+=(size_t)nr;
    }
    if (close(fd)) fail("cannot close frozen projection input");
    for (size_t i=0;i<count;i++) if (!isfinite(dst[i])) fail("non-finite frozen projection input");
}
static FILE *new_file(const char *leaf) {
    char p[4096]; path(p,sizeof p,leaf);
    FILE *f=fopen(p,"wx");
    if (!f) { perror(p); fail("cannot create output file"); }
    return f;
}
static void close_file(FILE *f) {
    if (fflush(f) || fclose(f)) fail("cannot close output file");
}
static void bytes_file(const char *leaf, const void *p, size_t bytes) {
    FILE *f=new_file(leaf);
    if (bytes && fwrite(p,1,bytes,f)!=bytes) fail("output write failed");
    close_file(f);
}
static void device_file(const char *leaf, CUdeviceptr p, size_t bytes) {
    unsigned char buf[65536];
    FILE *f=new_file(leaf);
    while (bytes) {
        size_t take=bytes<sizeof buf ? bytes : sizeof buf;
        CK(cuMemcpyDtoH(buf,p,take));
        if (fwrite(buf,1,take,f)!=take) fail("device output write failed");
        bytes-=take; p+=take;
    }
    close_file(f);
}
static void normalized_key_file(CUdeviceptr tiled, unsigned total) {
    float tile[32*512], row[512];
    FILE *f=new_file("normalized_k.f32");
    for (unsigned base=0;base<total;base+=32) {
        CK(cuMemcpyDtoH(tile,tiled+(size_t)(base/32)*sizeof tile,sizeof tile));
        unsigned count=total-base<32 ? total-base : 32;
        for (unsigned pos=0;pos<count;pos++) {
            for (unsigned kv=0;kv<2;kv++) for (unsigned d=0;d<256;d++)
                row[kv*256+d]=tile[(kv*64+d/4)*128+pos*4+d%4];
            if (fwrite(row,1,sizeof row,f)!=sizeof row) fail("normalized K write failed");
        }
    }
    close_file(f);
}
static CUdeviceptr device_alloc(size_t bytes) {
    CUdeviceptr p;
    CK(cuMemAlloc(&p,bytes ? bytes : 4));
    CK(cuMemsetD8(p,0,bytes ? bytes : 4));
    return p;
}
static CUdeviceptr upload(const void *p, size_t bytes) {
    CUdeviceptr d=device_alloc(bytes);
    CK(cuMemcpyHtoD(d,p,bytes));
    return d;
}
static void op(desc_t *p, unsigned code, unsigned units) {
    memset(p,0,sizeof *p); p->w[0]=code; p->w[1]=units;
}
static void ptr(desc_t *p, unsigned at, CUdeviceptr value) {
    unsigned i=2+(at-0x140)/4;
    p->w[i]=(uint32_t)value; p->w[i+1]=(uint32_t)(value>>32);
}
static void u32(desc_t *p, unsigned at, uint32_t value) { p->w[2+(at-0x140)/4]=value; }
static void f32(desc_t *p, unsigned at, float value) {
    uint32_t bits; memcpy(&bits,&value,4); u32(p,at,bits);
}
static void token(desc_t *p, unsigned at, unsigned base) {
    u32(p,at,base); p->w[31]|=1u<<(2+(at-0x140)/4);
}
static void parse_lengths(const char *arg, unsigned total) {
    char *s=strdup(arg), *save=NULL;
    if (!s) fail("length allocation failed");
    if (!*s || *s==',' || s[strlen(s)-1]==',' || strstr(s,",,")) fail("invalid snapshot list");
    for (char *v=strtok_r(s,",",&save); v; v=strtok_r(NULL,",",&save)) {
        unsigned n;
        if (nlengths==32 || parse_unsigned(v,1,total,&n)) fail("invalid snapshot length");
        for (unsigned i=0;i<nlengths;i++) if (lengths[i]==n) fail("duplicate snapshot length");
        lengths[nlengths++]=n;
    }
    free(s);
    if (!nlengths) fail("empty snapshot list");
}
static int wanted(unsigned n) {
    for (unsigned i=0;i<nlengths;i++) if (lengths[i]==n) return 1;
    return 0;
}
static void snapshot_file(unsigned n, const char *name, CUdeviceptr p, size_t bytes) {
    char leaf[128];
    int r=snprintf(leaf,sizeof leaf,"n%u/%s",n,name);
    if (r<0 || (size_t)r>=sizeof leaf) fail("snapshot name too long");
    device_file(leaf,p,bytes);
}
int main(int argc, char **argv) {
    unsigned total,budget;
    if (argc!=5 || parse_unsigned(argv[1],1,262144,&total) ||
        parse_unsigned(argv[2],1,4096,&budget)) {
        fprintf(stderr,"usage: qsa_attention_gpu T BUDGET SNAPSHOT_LENGTHS NEW_DIRECTORY\n");
        return 2;
    }
    root=argv[4]; parse_lengths(argv[3],total);
    frozen_root=getenv("ISSUE22_FROZEN_INPUT_DIR");
    if (frozen_root && !*frozen_root) frozen_root=NULL;
    weights_revision=getenv("ISSUE22_WEIGHT_REVISION");
    if (frozen_root && (!weights_revision || strlen(weights_revision)!=40))
        fail("frozen projection requires full ISSUE22_WEIGHT_REVISION");
    unsigned endian=1;
    if (*(unsigned char *)&endian!=1 || sizeof(float)!=4) fail("little-endian FP32 host required");
    struct stat st;
    if (!stat(root,&st) || errno!=ENOENT) fail("output directory must not exist");
    /* Validate every argument before touching the CUDA driver. */
    CK(cuInit(0));
    CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction fn;
    CK(cuDeviceGet(&dev,0)); CK(cuCtxCreate_v2(&ctx,CU_CTX_SCHED_SPIN,dev));
    char pci_bus_id[32]; int driver_version;
    CK(cuDeviceGetPCIBusId(pci_bus_id,sizeof pci_bus_id,dev));
    CK(cuDriverGetVersion(&driver_version));
    CK(cuModuleLoad(&mod,getenv("DIE_CUBIN") ? getenv("DIE_CUBIN") : "die.cubin"));
    CK(cuModuleGetFunction(&fn,mod,"die"));
    if (mkdir(root,0700)) fail("cannot create output directory");
    const float eps=1e-6f;
    float qn[256],kn[256],qg[12288];
    size_t raw_count=(size_t)total*512, rope_count=(size_t)total*32;
    float *k=alloc(raw_count*4), *v=alloc(raw_count*4);
    float *cs=alloc(rope_count*4), *sn=alloc(rope_count*4);
    if (frozen_root) {
        input_f32("qg.f32",qg,12288);
        input_f32("qn.f32",qn,256);
        input_f32("kn.f32",kn,256);
        input_f32("raw_k.f32",k,raw_count);
        input_f32("raw_v.f32",v,raw_count);
    } else {
        for (unsigned i=0;i<256;i++) { qn[i]=0.5f+0.5f*random_float(); kn[i]=0.5f+0.5f*random_float(); }
        for (unsigned i=0;i<12288;i++) qg[i]=2.0f*random_float();
        for (unsigned t=0;t<total;t++) {
            for (unsigned d=0;d<512;d++) {
                size_t i=(size_t)t*512+d;
                k[i]=2.0f*random_float(); v[i]=random_float();
                if (t%7==3) v[i]*=1e-4f;
                if (t%11==5 && d>=256) v[i]=0.0f;
                if (t%13==6) v[i]*=3000.0f;
            }
        }
    }
    for (unsigned t=0;t<total;t++) {
        for (unsigned d=0;d<32;d++) {
            double angle=(double)t*pow(10000000.0,-(double)d/32.0);
            cs[(size_t)t*32+d]=(float)cos(angle);
            sn[(size_t)t*32+d]=(float)sin(angle);
        }
    }
    CUdeviceptr dqg=upload(qg,sizeof qg), dqn=upload(qn,sizeof qn), dkn=upload(kn,sizeof kn);
    CUdeviceptr dk=upload(k,raw_count*4), dv=upload(v,raw_count*4);
    CUdeviceptr dcs=upload(cs,rope_count*4), dsn=upload(sn,rope_count*4);
    free(k); free(v); free(cs); free(sn);
    device_file("raw_k.f32",dk,raw_count*4); device_file("raw_v.f32",dv,raw_count*4);
    device_file("qn.f32",dqn,sizeof qn); device_file("kn.f32",dkn,sizeof kn);
    device_file("rope_cs.f32",dcs,rope_count*4); device_file("rope_sn.f32",dsn,rope_count*4);
    unsigned cpos=(4*(budget+1)+2048+31)/32*32;
    /* K is tiled in groups of 32 tokens, 2 KV heads and 256 channels. */
    /* Gather loads the four-token tail tile before masking invalid lanes. */
    unsigned cache_positions=(total+127)/128*128;
    CUdeviceptr dqo=device_alloc(6144*4);
    CUdeviceptr dkq=device_alloc((size_t)cache_positions*512), dvq=device_alloc((size_t)cache_positions*512);
    CUdeviceptr dks=device_alloc((size_t)cache_positions*32), dvs=device_alloc((size_t)cache_positions*32);
    CUdeviceptr dqo_f32=device_alloc(6144*4);
    CUdeviceptr dk_f32=device_alloc((size_t)cache_positions*512*4);
    CUdeviceptr dv_f32=device_alloc((size_t)cache_positions*512*4);
    CUdeviceptr dkc=device_alloc((size_t)cpos*512*4), dvc=device_alloc((size_t)cpos*512*4);
    CUdeviceptr dsel=device_alloc((size_t)(budget+8)*4);
    CUdeviceptr dpm=device_alloc(24*8*4), dpd=device_alloc(24*8*4);
    CUdeviceptr dpo=device_alloc(24*8*256*4), daout=device_alloc(6144*4);
    CUdeviceptr dprog=device_alloc(sizeof(desc_t)*5), dctl=device_alloc(8192);
    desc_t prog[5];
    op(prog,OP_ATTNQKQ,26);
    ptr(prog,0x140,dqg); ptr(prog,0x158,dqn); ptr(prog,0x160,dkn);
    ptr(prog,0x168,dcs); ptr(prog,0x170,dsn); ptr(prog,0x178,dqo);
    ptr(prog,0x180,dkq); ptr(prog,0x188,dvq); token(prog,0x190,0);
    f32(prog,0x194,eps); ptr(prog,0x198,dks); ptr(prog,0x1a0,dvs);
    op(prog+1,OP_ATTNQK,26);
    ptr(prog+1,0x140,dqg); ptr(prog+1,0x158,dqn); ptr(prog+1,0x160,dkn);
    ptr(prog+1,0x168,dcs); ptr(prog+1,0x170,dsn); ptr(prog+1,0x178,dqo_f32);
    ptr(prog+1,0x180,dk_f32); ptr(prog+1,0x188,dv_f32);
    token(prog+1,0x190,0); f32(prog+1,0x194,eps);
    op(prog+2,OP_KVGATHER,budget+1);
    ptr(prog+2,0x140,dkq); ptr(prog+2,0x148,dks); ptr(prog+2,0x150,dvq);
    ptr(prog+2,0x158,dvs); ptr(prog+2,0x160,dkc); ptr(prog+2,0x168,dvc);
    ptr(prog+2,0x170,dsel); token(prog+2,0x178,0); u32(prog+2,0x17c,budget);
    op(prog+3,OP_ATTNPARTN,32);
    ptr(prog+3,0x140,dqo); ptr(prog+3,0x148,dkc); ptr(prog+3,0x150,dvc);
    ptr(prog+3,0x158,dpm); ptr(prog+3,0x160,dpd); ptr(prog+3,0x168,dpo);
    token(prog+3,0x170,1); u32(prog+3,0x174,budget);
    op(prog+4,OP_ATTNMERGE,24);
    ptr(prog+4,0x140,dpm); ptr(prog+4,0x148,dpd); ptr(prog+4,0x150,dpo);
    ptr(prog+4,0x158,dqg); ptr(prog+4,0x160,daout);
    uint32_t *ids=alloc((size_t)budget*4);
    unsigned saved=0;
    for (unsigned t=0;t<total;t++) {
        unsigned n=t+1, capture=wanted(n), blocks=n/4, selected=blocks<budget ? blocks : budget;
        ptr(prog,0x148,dk+(size_t)t*512*4); ptr(prog,0x150,dv+(size_t)t*512*4);
        ptr(prog+1,0x148,dk+(size_t)t*512*4); ptr(prog+1,0x150,dv+(size_t)t*512*4);
        if (capture) {
            for (unsigned i=0;i<selected;i++) ids[i]=blocks>budget ? (uint32_t)((uint64_t)i*blocks/budget) : i;
            if (selected) CK(cuMemcpyHtoD(dsel,ids,(size_t)selected*4));
        }
        unsigned nph=capture ? 5 : 2, ntok=n;
        CK(cuMemcpyHtoD(dprog,prog,sizeof(desc_t)*nph));
        CK(cuMemcpyHtoD(dctl+8,&t,4));
        void *params[]={&dprog,&nph,&dctl,&ntok};
        CK(cuLaunchKernel(fn,16,1,1,256,1,1,0,0,params,NULL));
        CK(cuCtxSynchronize());
        if (!capture) continue;
        float query_int8[6144],query_f32[6144];
        CK(cuMemcpyDtoH(query_int8,dqo,sizeof query_int8));
        CK(cuMemcpyDtoH(query_f32,dqo_f32,sizeof query_f32));
        if (memcmp(query_int8,query_f32,sizeof query_int8))
            fail("FP32 and INT8 append normalized queries differ");
        char leaf[128],directory[4096];
        snprintf(leaf,sizeof leaf,"n%u",n); path(directory,sizeof directory,leaf);
        if (mkdir(directory,0700)) fail("cannot create snapshot directory");
        snapshot_file(n,"qg.f32",dqg,sizeof qg);
        snapshot_file(n,"qo.f32",dqo,6144*4); snapshot_file(n,"aout.f32",daout,6144*4);
        snapshot_file(n,"kq.i8",dkq,(size_t)n*512); snapshot_file(n,"vq.i8",dvq,(size_t)n*512);
        snapshot_file(n,"ks.f16",dks,(size_t)n*32); snapshot_file(n,"vs.f16",dvs,(size_t)n*32);
        snprintf(leaf,sizeof leaf,"n%u/selected_ids.u32",n);
        bytes_file(leaf,ids,(size_t)selected*4);
        snprintf(leaf,sizeof leaf,"n%u/snapshot.json",n);
        FILE *f=new_file(leaf);
        if (fprintf(f,"{\"schema\":\"flash-sass.qsa-attention-snapshot.v1\","
            "\"length\":%u,\"position\":%u,\"budget\":%u,\"complete_blocks\":%u,"
            "\"tail\":%u,\"attended_keys\":%u,\"sparse_exclusion\":%s,"
            "\"evidence_kind\":\"synthetic_gpu_fixture\",\"selection_source\":\"supplied_fixture_ids\","
            "\"production_sparse_selector_executed\":false,\"full_model\":false,"
            "\"normalized_query_append_modes_byte_identical\":true,"
            "\"program_opcodes\":[%u,%u,%u,%u,%u],\"quality_validation\":false",
            n,t,budget,blocks,n%4,4*selected+n%4,blocks>budget ? "true" : "false",
            OP_ATTNQKQ,OP_ATTNQK,OP_KVGATHER,OP_ATTNPARTN,OP_ATTNMERGE)<0) fail("metadata write failed");
        if (frozen_root && fprintf(f,",\"weights_origin\":\"original_checkpoint_safetensors_layer3\","
            "\"weights_revision\":\"%s\",\"activation_source\":\"synthetic_hidden_states\"",
            weights_revision)<0) fail("metadata write failed");
        if (fputs("}\n",f)==EOF) fail("metadata write failed");
        close_file(f); saved++;
        printf("captured N=%u: %u/%u fixed complete blocks, tail=%u; GPU attention, %s\n",
               n,selected,blocks,n%4,frozen_root ? "original layer-3 QKV weights" : "synthetic projections");
        fflush(stdout);
    }
    free(ids);
    if (saved!=nlengths) fail("not every requested snapshot completed");
    normalized_key_file(dk_f32,total);
    FILE *f=new_file("complete.json.tmp");
    if (fprintf(f,"{\"schema\":\"flash-sass.qsa-attention-capture.v1\","
        "\"positions\":%u,\"budget\":%u,\"epsilon\":%.17g,\"byte_order\":\"little\","
        "\"device_pci_bus_id\":\"%s\",\"cuda_driver_version\":%d,"
        "\"teacher_forced\":false,\"input_boundary\":\"post_projection_before_norm_rope_and_kv_quantization\","
        "\"gamma\":\"supplied_multiplicative_f32\",\"rope_tables\":\"supplied_f32\","
        "\"rope_base\":10000000,\"rope_table_generator\":\"fp64_libm_cast_f32\","
        "\"evidence_kind\":\"synthetic_gpu_fixture\",\"selection_source\":\"supplied_fixture_ids\","
        "\"production_sparse_selector_executed\":false,\"full_model\":false,\"seed\":4242,"
        "\"generator\":\"fixed_projection_lcg_v1\",\"build_revision\":\"%s\","
        "\"kv_dtype\":\"int8_with_fp16_group32_scales\",\"upstream_activation_quantization_isolated\":false,"
        "\"normalized_k_available\":true,\"prequant_k_source\":\"separate_gpu_fp32_append\","
        "\"independent_checkpoint_evidence\":false,\"quality_validation\":false,\"snapshot_lengths\":[",
        total,budget,(double)eps,pci_bus_id,driver_version,FLASH_REV)<0) fail("completion write failed");
    for (unsigned i=0;i<nlengths;i++) if (fprintf(f,"%s%u",i ? "," : "",lengths[i])<0) fail("completion write failed");
    if (fputs("]",f)==EOF) fail("completion write failed");
    if (frozen_root && fprintf(f,",\"weights_origin\":\"original_checkpoint_safetensors_layer3\","
        "\"weights_revision\":\"%s\",\"activation_source\":\"synthetic_hidden_states\"",
        weights_revision)<0) fail("completion write failed");
    if (fputs("}\n",f)==EOF) fail("completion write failed");
    close_file(f);
    char pending[4096],complete[4096];
    path(pending,sizeof pending,"complete.json.tmp"); path(complete,sizeof complete,"complete.json");
    if (link(pending,complete) || unlink(pending)) fail("cannot publish completion marker");
    CK(cuCtxDestroy(ctx));
    return 0;
}
