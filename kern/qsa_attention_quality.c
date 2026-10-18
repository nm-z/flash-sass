/* Independent higher-precision replay of frozen, fixed-selection attention.
 *
 * Equations: Transformers 8498ed425fa01bbee7d086efdf1a6634c2962aed,
 * modeling_qwen4_exp.py blob 946f5fd6e9ba21347654b1b2ca196e86b0f751b3:
 * RMSNorm 143..163; partial RoPE 630..661; GQA/attention/gate 773..890.
 * No local instruction twin is linked. FP64 scalar arithmetic does not
 * reproduce framework FP32/BF16 casts, GPU reductions, or checkpoint inference.
 * The Perl wrapper validates metadata, hashes and optional numerical limits.
 */
#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

enum { H=24, KV=2, D=256, W=512, PAIRS=32, MAXMODES=7, PATHCAP=4096 };
#define MAX_TOKENS ((size_t)262144)
#define MEMORY_LIMIT ((size_t)512*1024*1024)
_Static_assert(CHAR_BIT==8 && sizeof(float)==4 && sizeof(double)==8,
               "requires byte-addressed binary32/binary64");
_Static_assert(FLT_RADIX==2 && FLT_MANT_DIG==24 && FLT_MAX_EXP==128 &&
               DBL_MANT_DIG==53 && DBL_MAX_EXP==1024, "requires IEEE formats");
static size_t allocated;
static _Noreturn void fail(const char *fmt, ...) {
    va_list ap;
    fputs("qsa_attention_quality: ",stderr);
    va_start(ap,fmt); vfprintf(stderr,fmt,ap); va_end(ap);
    fputc('\n',stderr); exit(2);
}
static void *memory(size_t count, size_t width) {
    if (count && width>SIZE_MAX/count) fail("allocation size overflow");
    size_t bytes=count*width;
    if (!bytes) bytes=1;
    if (bytes>MEMORY_LIMIT-allocated) fail("512 MiB cumulative allocation limit exceeded");
    void *p=calloc(1,bytes);
    if (!p) fail("allocation of %zu bytes failed",bytes);
    allocated+=bytes; return p;
}
static size_t integer(const char *s, size_t lo, size_t hi) {
    if (!*s) fail("empty integer");
    for (const char *p=s;*p;p++) if (*p<'0'||*p>'9') fail("invalid integer: %s",s);
    errno=0; char *end;
    uintmax_t n=strtoumax(s,&end,10);
    if (errno || *end || n<lo || n>hi) fail("integer outside [%zu,%zu]: %s",lo,hi,s);
    return (size_t)n;
}
static void path(char out[PATHCAP], const char *dir, const char *name) {
    int n=snprintf(out,PATHCAP,"%s/%s",dir,name);
    if (n<0 || n>=PATHCAP) fail("path too long");
}
static FILE *input(const char *p) {
    FILE *f=fopen(p,"rb");
    if (!f) fail("cannot open %s: %s",p,strerror(errno));
    return f;
}
static void read_bytes(FILE *f, void *p, size_t n, const char *name) {
    if (n && fread(p,1,n,f)!=n) fail("short file or read error: %s",name);
}
static void end_input(FILE *f, const char *name) {
    if (fgetc(f)!=EOF || ferror(f)) fail("file exceeds declared shape or read failed: %s",name);
    if (fclose(f)) fail("cannot close %s",name);
}
static uint32_t u32(const unsigned char *p) {
    return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;
}
static double f32(const unsigned char *p) {
    uint32_t bits=u32(p); float f; memcpy(&f,&bits,4); return f;
}
static bool half(uint16_t bits, double *out) {
    unsigned e=(bits>>10)&31, m=bits&1023;
    if (e==31) return false;
    double v=e ? ldexp((double)(1024+m),(int)e-25) : ldexp((double)m,-24);
    *out=bits&0x8000 ? -v : v; return true;
}
/* Validate all rows while retaining only the fixed selected-token rows.
 * Also retain the query-position RoPE row if that complete block was excluded. */
static double *load_f32(const char *dir, const char *name, size_t rows,
                        size_t width, const int32_t *map, size_t keep,
                        size_t extra_row, double *extra) {
    char p[PATHCAP]; path(p,dir,name);
    if (!rows || rows>MAX_TOKENS || !width || width>W || (!map && keep!=rows))
        fail("invalid shape for %s",p);
    unsigned char buf[W*4];
    double *a=memory(keep*width,sizeof *a);
    FILE *f=input(p);
    for (size_t row=0;row<rows;row++) {
        int32_t dest=map ? map[row] : (int32_t)row;
        if (dest<-1 || (dest>=0 && (size_t)dest>=keep)) fail("invalid row map");
        read_bytes(f,buf,width*4,p);
        for (size_t d=0;d<width;d++) {
            double v=f32(buf+d*4);
            if (!isfinite(v)) fail("nonfinite f32: %s row %zu channel %zu",p,row,d);
            if (dest>=0) a[(size_t)dest*width+d]=v;
            if (extra && row==extra_row) extra[d]=v;
        }
    }
    end_input(f,p); return a;
}
static double *dequant(const char *dir, const char *qn, const char *sn,
                       size_t n, const int32_t *map, size_t keep) {
    char qp[PATHCAP],sp[PATHCAP]; path(qp,dir,qn); path(sp,dir,sn);
    FILE *qf=input(qp), *sf=input(sp);
    double *a=memory(keep*W,sizeof *a);
    unsigned char q[W],s[32];
    for (size_t t=0;t<n;t++) {
        read_bytes(qf,q,sizeof q,qp); read_bytes(sf,s,sizeof s,sp);
        double scales[16];
        for (unsigned g=0;g<16;g++) {
            uint16_t b=(uint16_t)(s[g*2]|(unsigned)s[g*2+1]<<8);
            if (!half(b,scales+g) || scales[g]<0) fail("invalid f16 scale: %s row %zu group %u",sp,t,g);
        }
        if (map[t]>=0) for (unsigned d=0;d<W;d++) {
            int value=q[d]<128 ? q[d] : (int)q[d]-256;
            a[(size_t)map[t]*W+d]=value*scales[d/32];
        }
    }
    end_input(qf,qp); end_input(sf,sp); return a;
}
static uint32_t *selection(const char *dir, size_t n, size_t budget, size_t *count) {
    size_t blocks=n/4; *count=blocks<budget ? blocks : budget;
    uint32_t *ids=memory(*count,sizeof *ids);
    unsigned char *seen=memory(blocks,1);
    char p[PATHCAP]; path(p,dir,"selected_ids.u32"); FILE *f=input(p);
    for (size_t i=0;i<*count;i++) {
        unsigned char b[4]; read_bytes(f,b,4,p); ids[i]=u32(b);
        if (ids[i]>=blocks) fail("selected block outside completed prefix");
        if (seen[ids[i]]++) fail("duplicate selected block");
        if (blocks<=budget && ids[i]!=i) fail("implicit dense IDs must be ascending complete blocks");
    }
    end_input(f,p); free(seen); return ids;
}
static void attention_set(size_t n, size_t total, const uint32_t *ids, size_t count,
                           uint32_t **tokens, int32_t **mapping, size_t *keep) {
    *keep=count*4+n%4;
    if (!*keep || *keep>n || n>total || total>MAX_TOKENS) fail("invalid attended set");
    *tokens=memory(*keep,sizeof **tokens); *mapping=memory(total,sizeof **mapping);
    for (size_t t=0;t<total;t++) (*mapping)[t]=-1;
    size_t at=0;
    for (size_t b=0;b<count;b++) {
        if (ids[b]>=n/4) fail("selected block outside prefix");
        for (size_t d=0;d<4;d++) {
            size_t t=ids[b]*4+d;
            if ((*mapping)[t]!=-1) fail("selected tokens overlap");
            (*tokens)[at]=(uint32_t)t; (*mapping)[t]=(int32_t)at++;
        }
    }
    for (size_t t=n/4*4;t<n;t++) {
        (*tokens)[at]=(uint32_t)t; (*mapping)[t]=(int32_t)at++;
    }
    if (at!=*keep) fail("attended set length mismatch");
}
static void normalize(double *v, const double *gamma, const double *cs,
                       const double *sn, double epsilon) {
    double ss=0;
    for (unsigned d=0;d<D;d++) ss+=v[d]*v[d];
    double scale=1/sqrt(ss/D+epsilon);
    if (!isfinite(scale) || scale<=0) fail("invalid RMSNorm denominator");
    for (unsigned d=0;d<D;d++) {
        v[d]=(v[d]*scale)*gamma[d];
        if (!isfinite(v[d])) fail("nonfinite normalized input");
    }
    for (unsigned d=0;d<PAIRS;d++) {
        double a=v[d],b=v[d+PAIRS];
        v[d]=a*cs[d]-b*sn[d]; v[d+PAIRS]=b*cs[d]+a*sn[d];
        if (!isfinite(v[d]) || !isfinite(v[d+PAIRS])) fail("nonfinite RoPE result");
    }
}
static double sigmoid(double x) {
    if (x>=0) return 1/(1+exp(-x));
    double e=exp(x); return e/(1+e);
}
static void attention(size_t n, const double *q, const double *k,
                      const double *v, const double *qg, double *scores,
                      double *out) {
    if (!n) fail("empty attention set");
    for (unsigned h=0;h<H;h++) {
        unsigned kv=h/12;
        double maximum=-DBL_MAX,den=0,acc[D]={0};
        for (size_t t=0;t<n;t++) {
            double dot=0; const double *key=k+t*W+kv*D;
            for (unsigned d=0;d<D;d++) dot+=q[h*D+d]*key[d];
            scores[t]=dot/16;
            if (!isfinite(scores[t])) fail("nonfinite attention score");
            if (scores[t]>maximum) maximum=scores[t];
        }
        for (size_t t=0;t<n;t++) { scores[t]=exp(scores[t]-maximum); den+=scores[t]; }
        if (!isfinite(den) || den<=0) fail("invalid softmax denominator");
        for (size_t t=0;t<n;t++) for (unsigned d=0;d<D;d++)
            acc[d]+=scores[t]*v[t*W+kv*D+d];
        for (unsigned d=0;d<D;d++) {
            out[h*D+d]=(acc[d]/den)*sigmoid(qg[h*W+D+d]);
            if (!isfinite(out[h*D+d])) fail("nonfinite attention output");
        }
    }
}
typedef struct {
    double max_abs,mean_abs,rmse,error_l2,reference_l2,observed_l2,relative_l2,cosine;
    bool relative_l2_finite,reference_zero,observed_zero,cosine_defined;
} metric;
static metric measure(const double *observed, const double *reference, size_t n) {
    metric r={0}; double sum_abs=0;
    for (size_t i=0;i<n;i++) {
        double e=observed[i]-reference[i],a=fabs(e);
        if (!isfinite(e)) fail("nonfinite metric input");
        if (a>r.max_abs) r.max_abs=a;
        sum_abs+=a; r.error_l2=hypot(r.error_l2,e);
        r.reference_l2=hypot(r.reference_l2,reference[i]);
        r.observed_l2=hypot(r.observed_l2,observed[i]);
    }
    r.mean_abs=sum_abs/n; r.rmse=r.error_l2/sqrt((double)n);
    r.reference_zero=r.reference_l2==0; r.observed_zero=r.observed_l2==0;
    r.relative_l2=r.reference_zero ? 0 : r.error_l2/r.reference_l2;
    r.relative_l2_finite=r.reference_zero ? r.error_l2==0 : isfinite(r.relative_l2);
    r.cosine_defined=!r.reference_zero && !r.observed_zero;
    if (r.cosine_defined) {
        for (size_t i=0;i<n;i++) r.cosine+=(observed[i]/r.observed_l2)*(reference[i]/r.reference_l2);
        if (r.cosine>1) r.cosine=1;
        if (r.cosine< -1) r.cosine=-1;
    }
    if (!isfinite(r.mean_abs) || !isfinite(r.rmse) || !isfinite(r.error_l2) ||
        !isfinite(r.reference_l2) || !isfinite(r.observed_l2) || !isfinite(r.cosine))
        fail("nonfinite metric");
    return r;
}
static void metrics(FILE *f, const metric *r) {
    fprintf(f,"{\"max_abs\":%.17g,\"mean_abs\":%.17g,\"rmse\":%.17g,"
        "\"error_l2\":%.17g,\"reference_l2\":%.17g,\"observed_l2\":%.17g,\"relative_l2\":",
        r->max_abs,r->mean_abs,r->rmse,r->error_l2,r->reference_l2,r->observed_l2);
    if (r->relative_l2_finite) fprintf(f,"%.17g",r->relative_l2); else fputs("null",f);
    fprintf(f,",\"relative_l2_finite\":%s,\"reference_zero\":%s,\"observed_zero\":%s,\"cosine\":",
        r->relative_l2_finite?"true":"false",r->reference_zero?"true":"false",r->observed_zero?"true":"false");
    if (r->cosine_defined) fprintf(f,"%.17g",r->cosine); else fputs("null",f);
    fprintf(f,",\"cosine_defined\":%s}",r->cosine_defined?"true":"false");
}
static void uint_array(FILE *f, const uint32_t *a, size_t n) {
    fputc('[',f);
    for (size_t i=0;i<n;i++) fprintf(f,"%s%" PRIu32,i?",":"",a[i]);
    fputc(']',f);
}
static void report(const char *p, size_t n, size_t total, size_t budget, double eps,
                   const uint32_t *ids, size_t count, const uint32_t *tokens,
                   size_t keep, const metric *m, const metric *qm,
                   const double *out, bool normalized_k_available) {
    const char *names[MAXMODES]={
        "exported_q_dequantized_kv",
        "exported_q_raw_k_normalized_rope_dequantized_v",
        "exported_q_dequantized_k_raw_v",
        "exported_q_raw_k_normalized_rope_raw_v",
        "reconstructed_q_dequantized_kv",
        "exported_q_gpu_f32_k_dequantized_v",
        "exported_q_gpu_f32_k_raw_v"};
    const char *interpretations[MAXMODES]={
        "arithmetic_control_against_supplied_gated_output",
        "k_storage_plus_k_normalization_and_rope_arithmetic_sensitivity",
        "v_storage_sensitivity",
        "combined_k_storage_normalization_rope_and_v_storage_sensitivity",
        "q_normalization_and_rope_arithmetic_sensitivity",
        "k_append_normalization_rope_and_storage_path_sensitivity",
        "k_append_path_and_v_storage_combined_sensitivity"};
    FILE *f=fopen(p,"wbx");
    if (!f) fail("cannot exclusively create %s: %s",p,strerror(errno));
    fprintf(f,"{\"schema\":\"flash-sass.qsa-attention-reference.v1\",\"complete\":true,"
        "\"gpu\":false,\"quality\":false,\"higher_precision_math_replay\":true,"
        "\"framework_dtype_equivalence\":false,\"framework_casts_reproduced\":false,"
        "\"production_sparse_dispatch_verified\":false,\"numerical_acceptance_performed\":false,"
        "\"capture_execution_provenance\":\"not_validated_by_this_binary\","
        "\"selection_source\":\"provided_selected_ids\",\"all_modes_same_selected_ids\":true,"
        "\"all_modes_same_attended_token_order\":true,"
        "\"gamma_convention\":\"supplied_already_converted_multiplicative\","
        "\"softmax_arithmetic\":\"stable_binary64_scalar\","
        "\"gate_arithmetic\":\"stable_binary64_sigmoid_after_attention\","
        "\"length\":%zu,\"total_positions\":%zu,\"budget\":%zu,\"epsilon\":%.17g,"
        "\"complete_blocks\":%zu,\"selected_blocks\":%zu,\"excluded_complete_blocks\":%zu,"
        "\"tail\":%zu,\"attended_keys\":%zu,\"heads\":24,\"kv_heads\":2,\"head_dim\":256,"
        "\"gqa_group_size\":12,\"qk_scale\":0.0625,\"rope_rotated_dimensions\":64,"
        "\"rope_pair_offset\":32,\"cumulative_allocated_bytes\":%zu,"
        "\"normalized_k_available\":%s,"
        "\"pure_k_storage_isolated\":false,"
        "\"fp32_k_append_note\":\"separate_append_may_include_k_normalization_rope_differences\","
        "\"relative_l2_zero_reference_rule\":\"zero_when_both_zero_else_null_meaning_positive_infinity\","
        "\"selected_ids\":",n,total,budget,eps,n/4,count,n/4-count,n%4,keep,allocated,
        normalized_k_available?"true":"false");
    uint_array(f,ids,count);
    fputs(",\"attended_token_ids\":",f); uint_array(f,tokens,keep);
    fputs(",\"arithmetic_vs_gpu\":",f); metrics(f,m);
    fputs(",\"query_vs_gpu\":",f); metrics(f,qm);
    fputs(",\"modes\":[",f);
    unsigned modes=normalized_k_available ? MAXMODES : 5;
    for (unsigned mode=0;mode<modes;mode++) {
        fprintf(f,"%s{\"name\":\"%s\",\"interpretation\":\"%s\",\"reference\":\"%s\","
            "\"same_selected_ids\":true,\"metrics\":",mode?",":"",names[mode],interpretations[mode],
            mode?names[0]:"supplied_aout");
        metrics(f,m+mode);
        fputs(",\"output_shape\":[24,256],\"output_f64_row_major\":[",f);
        for (unsigned d=0;d<H*D;d++) fprintf(f,"%s%.17g",d?",":"",out[mode*H*D+d]);
        fputs("]}",f);
    }
    fputs("]}\n",f);
    if (ferror(f) || fclose(f)) fail("cannot finish report %s",p);
}
static int replay(const char *root, size_t n, size_t total, size_t budget,
                  double epsilon, const char *output) {
    char snap[PATHCAP],leaf[64],optional[PATHCAP];
    snprintf(leaf,sizeof leaf,"n%zu",n); path(snap,root,leaf);
    size_t count,keep; uint32_t *ids=selection(snap,n,budget,&count),*tokens; int32_t *map;
    attention_set(n,total,ids,count,&tokens,&map,&keep);
    double *kn=load_f32(root,"kn.f32",1,D,NULL,1,0,NULL);
    double *qn=load_f32(root,"qn.f32",1,D,NULL,1,0,NULL);
    double *rk=load_f32(root,"raw_k.f32",total,W,map,keep,0,NULL);
    double *rv=load_f32(root,"raw_v.f32",total,W,map,keep,0,NULL);
    double qc[PAIRS],qs[PAIRS];
    double *cs=load_f32(root,"rope_cs.f32",total,PAIRS,map,keep,n-1,qc);
    double *sn=load_f32(root,"rope_sn.f32",total,PAIRS,map,keep,n-1,qs);
    double *dk=dequant(snap,"kq.i8","ks.f16",n,map,keep);
    double *dv=dequant(snap,"vq.i8","vs.f16",n,map,keep);
    double *qg=load_f32(snap,"qg.f32",H,W,NULL,H,0,NULL);
    double *qo=load_f32(snap,"qo.f32",H,D,NULL,H,0,NULL);
    double *aout=load_f32(snap,"aout.f32",H,D,NULL,H,0,NULL);
    path(optional,root,"normalized_k.f32");
    struct stat st; bool has_k=stat(optional,&st)==0;
    if (!has_k && errno!=ENOENT) fail("cannot inspect optional normalized K");
    double *nk=has_k ? load_f32(root,"normalized_k.f32",total,W,map,keep,0,NULL) : NULL;
    double *rq=memory(H*D,sizeof *rq),*out=memory(MAXMODES*H*D,sizeof *out);
    double *scratch=memory(keep,sizeof *scratch);
    for (size_t t=0;t<keep;t++) for (unsigned kv=0;kv<KV;kv++)
        normalize(rk+t*W+kv*D,kn,cs+t*PAIRS,sn+t*PAIRS,epsilon);
    for (unsigned h=0;h<H;h++) {
        memcpy(rq+h*D,qg+h*W,D*sizeof *rq); normalize(rq+h*D,qn,qc,qs,epsilon);
    }
    metric qm=measure(rq,qo,H*D),m[MAXMODES];
    attention(keep,qo,dk,dv,qg,scratch,out);
    attention(keep,qo,rk,dv,qg,scratch,out+H*D);
    attention(keep,qo,dk,rv,qg,scratch,out+2*H*D);
    attention(keep,qo,rk,rv,qg,scratch,out+3*H*D);
    attention(keep,rq,dk,dv,qg,scratch,out+4*H*D);
    if (has_k) {
        attention(keep,qo,nk,dv,qg,scratch,out+5*H*D);
        attention(keep,qo,nk,rv,qg,scratch,out+6*H*D);
    }
    m[0]=measure(out,aout,H*D);
    for (unsigned i=1;i<(has_k?MAXMODES:5);i++) m[i]=measure(out+i*H*D,out,H*D);
    report(output,n,total,budget,epsilon,ids,count,tokens,keep,m,&qm,out,has_k);
    free(ids); free(tokens); free(map); free(kn); free(qn); free(rk); free(rv);
    free(cs); free(sn); free(dk); free(dv); free(qg); free(qo); free(aout);
    free(nk); free(rq); free(out); free(scratch);
    return 0;
}
static void close_enough(double a, double b, double tolerance, const char *name) {
    if (!isfinite(a) || fabs(a-b)>tolerance) fail("selftest %s: %.17g != %.17g",name,a,b);
}
static int selftest(void) {
    double decoded;
    const uint16_t halves[]={0x3c00,0xbc00,0x0001,0x7bff};
    const double expected[]={1,-1,0x1p-24,65504};
    for (unsigned i=0;i<4;i++) {
        if (!half(halves[i],&decoded)) fail("finite half rejected");
        close_enough(decoded,expected[i],0,"half conversion");
    }
    if (half(0x7c00,&decoded)||half(0x7e00,&decoded)) fail("nonfinite half accepted");
    double vec[D]={0},gamma[D],cs[PAIRS],sn[PAIRS]={0};
    for (unsigned d=0;d<D;d++) gamma[d]=1;
    for (unsigned d=0;d<PAIRS;d++) cs[d]=1;
    vec[0]=3; vec[32]=4; cs[0]=0; sn[0]=1;
    normalize(vec,gamma,cs,sn,1e-6);
    close_enough(vec[0],-4/sqrt(25.0/256+1e-6),1e-13,"RMS RoPE left");
    close_enough(vec[32],3/sqrt(25.0/256+1e-6),1e-13,"RMS RoPE right");
    double *q=memory(H*D,8),*qg=memory(H*W,8),*k=memory(7*W,8);
    double *v=memory(7*W,8),*out=memory(H*D,8),scores[7];
    for (unsigned t=0;t<2;t++) for (unsigned kv=0;kv<KV;kv++)
        for (unsigned d=0;d<D;d++) v[t*W+kv*D+d]=(kv?10:2)+4*t;
    attention(2,q,k,v,qg,scores,out);
    for (unsigned h=0;h<H;h++) for (unsigned d=0;d<D;d++)
        close_enough(out[h*D+d],h<12?2:6,0,"uniform GQA");
    for (unsigned h=0;h<H;h++) { q[h*D]=1; qg[h*W+D+1]=1000; qg[h*W+D+2]=-1000; }
    for (unsigned kv=0;kv<KV;kv++) { k[kv*D]=-16000; k[W+kv*D]=16000; }
    attention(2,q,k,v,qg,scores,out);
    for (unsigned h=0;h<H;h++) {
        close_enough(out[h*D],h<12?3:7,0,"stable softmax");
        close_enough(out[h*D+1],h<12?6:14,0,"positive sigmoid");
        close_enough(out[h*D+2],0,0,"negative sigmoid");
    }
    uint32_t selected[]={1},*tokens; int32_t *map; size_t keep;
    attention_set(11,11,selected,1,&tokens,&map,&keep);
    if (keep!=7) fail("tail count");
    for (unsigned i=0;i<7;i++) if (tokens[i]!=i+4 || map[i+4]!=(int32_t)i) fail("tail order");
    for (unsigned i=0;i<4;i++) if (map[i]!=-1) fail("excluded block included");
    memset(q,0,H*D*8); memset(qg,0,H*W*8); memset(k,0,7*W*8);
    for (size_t t=0;t<keep;t++) for (unsigned d=0;d<W;d++) v[t*W+d]=tokens[t];
    attention(keep,q,k,v,qg,scores,out);
    for (unsigned d=0;d<H*D;d++) close_enough(out[d],3.5,0,"tail attention");
    double zero[2]={0},nonzero[2]={1,0};
    metric a=measure(zero,zero,2),b=measure(nonzero,zero,2);
    if (!a.relative_l2_finite || a.relative_l2!=0 || b.relative_l2_finite) fail("zero reference rule");
    free(q); free(qg); free(k); free(v); free(out); free(tokens); free(map);
    puts("{\"schema\":\"flash-sass.qsa-attention-selftest.v1\",\"complete\":true,"
         "\"analytical_test_groups\":6,\"zero_reference_metric_check\":true,\"gpu\":false,\"quality\":false}");
    return 0;
}
int main(int argc, char **argv) {
    if (argc==2 && !strcmp(argv[1],"--selftest")) return selftest();
    if (argc!=8 || strcmp(argv[1],"--replay")) {
        fputs("usage: qsa_attention_quality --selftest | --replay ROOT N TOTAL BUDGET EPS NEWJSON\n",stderr);
        return 2;
    }
    if (!*argv[2] || !*argv[7]) fail("empty path");
    size_t n=integer(argv[3],1,MAX_TOKENS), total=integer(argv[4],n,MAX_TOKENS);
    size_t budget=integer(argv[5],1,4096);
    errno=0; char *end; double eps=strtod(argv[6],&end);
    if (errno || end==argv[6] || *end || !isfinite(eps) || eps<0) fail("invalid epsilon");
    return replay(argv[2],n,total,budget,eps,argv[7]);
}
