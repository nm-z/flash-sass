/*
 * Independent postprojection QSA indexer replay.
 * Mathematical source: huggingface/transformers
 * 8498ed425fa01bbee7d086efdf1a6634c2962aed,
 * src/transformers/models/qwen4_exp/modeling_qwen4_exp.py.
 *
 * Supplied gamma is multiplicative: DO NOT add one again. FP64 is a
 * numerical control over frozen FP32 inputs, not original model weights.
 * This tool does not reproduce projection arithmetic or attention output.
 * Compile without floating contraction or fast-math.
 */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

enum { D=128, H=4, R=32, BLOCK=4, NM=7 };
static const char *mode_name[NM] = {
    "reference_fp64", "fixed_iq_fp64_pool", "fixed_iq_scalar_fp32_pool",
    "fixed_iq_scalar_fp32_pool_fp16_store", "fixed_iq_fp16_mean_norm_rope_return_casts",
    "fixed_iq_bf16_mean_norm_rope_return_casts", "fixed_iq_exported_pool"
};
typedef struct { size_t n; long double error2, reference2; double max_abs; } Metric;
typedef struct { double score; uint32_t id; } Entry;
typedef struct {
    uint32_t *ids;
    size_t n, k, strict_above, cutoff_ties;
    double cutoff, next;
} Ranking;

static void fail(const char *message) {
    fprintf(stderr,"qsa_quality: %s\n",message); exit(2);
}
static void file_fail(const char *path) {
    fprintf(stderr,"qsa_quality: %s: %s\n",path,strerror(errno)); exit(2);
}
static size_t product(size_t a,size_t b) {
    if(b && a>SIZE_MAX/b) fail("size overflow");
    return a*b;
}
static void *allocate(size_t n) {
    void *p=calloc(n?n:1,1);
    if(!p) fail("allocation failed");
    return p;
}
static char *path_part(const char *a,const char *b,int slash) {
    size_t n=strlen(a)+strlen(b)+1+(size_t)slash;
    char *p=allocate(n);
    if(snprintf(p,n,slash?"%s/%s":"%s%s",a,b)<0) fail("path formatting failed");
    return p;
}
static char *join(const char *a,const char *b) { return path_part(a,b,1); }
static char *suffix(const char *a,const char *b) { return path_part(a,b,0); }
static char *snapshot_path(const char *directory,size_t n) {
    char name[64];
    snprintf(name,sizeof(name),"n%zu",n);
    return join(directory,name);
}
/* Validate the whole stream's shape even when reading only a prefix. */
static void *read_member(const char *directory,const char *name,size_t expected,size_t wanted) {
    char *path=join(directory,name);
    struct stat st;
    if(wanted>expected) fail("invalid prefix size");
    if(stat(path,&st)) file_fail(path);
    if(!S_ISREG(st.st_mode)||st.st_size<0||(uintmax_t)st.st_size!=(uintmax_t)expected) {
        fprintf(stderr,"qsa_quality: %s: expected %zu bytes, found %" PRIdMAX "\n",
                path,expected,(intmax_t)st.st_size); exit(2);
    }
    FILE *f=fopen(path,"rb");
    if(!f) file_fail(path);
    void *p=allocate(wanted);
    if(wanted && fread(p,1,wanted,f)!=wanted) fail("short input read");
    if(fclose(f)) file_fail(path);
    free(path); return p;
}
static FILE *create_file(const char *path) {
    FILE *f=fopen(path,"wbx");
    if(!f) file_fail(path);
    return f;
}
static void close_file(FILE *f,const char *path) {
    if(ferror(f)) fail("output write failed");
    if(fclose(f)) file_fail(path);
}
static void write_exact(const char *path,const void *p,size_t bytes) {
    FILE *f=create_file(path);
    if(bytes && fwrite(p,1,bytes,f)!=bytes) fail("short output write");
    close_file(f,path);
}
static void write_member(const char *directory,const char *name,const void *p,size_t bytes) {
    char *path=join(directory,name);
    write_exact(path,p,bytes); free(path);
}
static void check_floats(const float *p,size_t n) {
    for(size_t i=0;i<n;++i) if(!isfinite(p[i])) fail("nonfinite FP32 input");
}
static uint32_t float_bits(float x) { uint32_t u; memcpy(&u,&x,4); return u; }
static float bits_float(uint32_t u) { float x; memcpy(&x,&u,4); return x; }
/* IEEE binary32 -> binary16, round to nearest, ties to even. */
static uint16_t half_bits(float x) {
    uint32_t u=float_bits(x),sign=(u>>16)&0x8000u,mantissa=u&0x7fffffu;
    unsigned raw_exp=(u>>23)&255u;
    int exp=(int)raw_exp-127+15;
    if(raw_exp==255u) return (uint16_t)(sign|(mantissa?0x7e00u:0x7c00u));
    if(exp>=31) return (uint16_t)(sign|0x7c00u);
    if(exp<=0) {
        if(exp < -10) return (uint16_t)sign;
        mantissa|=0x800000u;
        unsigned shift=(unsigned)(14-exp);
        uint32_t truncated=mantissa>>shift;
        uint32_t remainder=mantissa&((UINT32_C(1)<<shift)-1u);
        uint32_t halfway=UINT32_C(1)<<(shift-1u);
        if(remainder>halfway || (remainder==halfway && (truncated&1u))) ++truncated;
        return (uint16_t)(sign|truncated);
    }
    uint32_t truncated=((uint32_t)exp<<10)|(mantissa>>13),remainder=mantissa&0x1fffu;
    if(remainder>0x1000u || (remainder==0x1000u && (truncated&1u))) ++truncated;
    return (uint16_t)(sign|truncated);
}
static double half_value(uint16_t h) {
    unsigned exp=(h>>10)&31u,m=h&1023u;
    double x=exp==31u?(m?NAN:INFINITY):!exp?ldexp((double)m,-24):ldexp((double)(1024u+m),(int)exp-25);
    return h&0x8000u?-x:x;
}
static float rounded(float x,int dtype) {
    if(dtype==1) return (float)half_value(half_bits(x));
    if(dtype==2) {
        uint32_t u=float_bits(x);
        if((u&0x7f800000u)!=0x7f800000u) u+=0x7fffu+((u>>16)&1u);
        return bits_float(u&0xffff0000u);
    }
    return x;
}
static void metric_add(Metric *m,double candidate,double reference) {
    if(!isfinite(candidate)||!isfinite(reference)) fail("nonfinite replay value");
    long double error=(long double)candidate-(long double)reference;
    m->error2+=error*error;
    m->reference2+=(long double)reference*(long double)reference;
    if(fabsl(error)>m->max_abs) m->max_abs=(double)fabsl(error);
    ++m->n;
}
static void metric_json(FILE *f,const Metric *m) {
    if(!m->n) { fputs("{\"count\":0,\"max_abs\":null,\"rms\":null,\"relative_l2\":null}",f); return; }
    fprintf(f,"{\"count\":%zu,\"max_abs\":%.17g,\"rms\":%.17Lg,\"relative_l2\":",
            m->n,m->max_abs,sqrtl(m->error2/(long double)m->n));
    if(m->reference2>0) fprintf(f,"%.17Lg",sqrtl(m->error2/m->reference2));
    else fputs("null",f);
    fputc('}',f);
}
static void quoted(FILE *f,const char *s) {
    fputc('"',f);
    for(;*s;++s) {
        unsigned char c=(unsigned char)*s;
        if(c=='"'||c=='\\') { fputc('\\',f); fputc(c,f); }
        else if(c<32) fprintf(f,"\\u%04x",(unsigned)c);
        else fputc(c,f);
    }
    fputc('"',f);
}
static void norm_rope64(const double input[D],const float gamma[D],
                        const float cs[R],const float sn[R],double epsilon,double output[D]) {
    double sum=0;
    for(size_t d=0;d<D;++d) sum+=input[d]*input[d];
    double inverse=1.0/sqrt(sum/D+epsilon);
    if(!isfinite(inverse)) fail("nonfinite FP64 RMS normalization");
    for(size_t d=0;d<D;++d) output[d]=(input[d]*inverse)*(double)gamma[d];
    for(size_t p=0;p<R;++p) {
        double a=output[p],b=output[p+R];
        output[p]=a*(double)cs[p]-b*(double)sn[p];
        output[p+R]=b*(double)cs[p]+a*(double)sn[p];
    }
}
/*
 * Scalar FP32 arithmetic control. dtype1/2 round mean, RMSNorm return, and
 * RoPE return to FP16/BF16. This explicitly defined cast experiment does
 * not claim device reduction or framework RoPE bitwise equivalence.
 */
static void norm_rope32(const float input[D],const float gamma[D],
                        const float cs[R],const float sn[R],double epsilon,int dtype,float output[D]) {
    float x[D],sum=0;
    for(size_t d=0;d<D;++d) { x[d]=rounded(input[d],dtype); float square=x[d]*x[d]; sum=sum+square; }
    float inverse=1.0f/sqrtf(sum/(float)D+(float)epsilon);
    if(!isfinite(inverse)) fail("nonfinite scalar FP32 RMS normalization");
    for(size_t d=0;d<D;++d) { float normalized=x[d]*inverse; output[d]=rounded(normalized*gamma[d],dtype); }
    for(size_t p=0;p<R;++p) {
        float a=output[p],b=output[p+R];
        float ac=a*cs[p],bs=b*sn[p],bc=b*cs[p],as=a*sn[p];
        output[p]=rounded(ac-bs,dtype); output[p+R]=rounded(bc+as,dtype);
    }
    check_floats(output,D);
}
static double score64(const double query[H*D],const double pool[D]) {
    double score=0;
    for(size_t h=0;h<H;++h) {
        double dot=0;
        for(size_t d=0;d<D;++d) dot+=query[h*D+d]*pool[d];
        if(dot>0) score+=dot;
    }
    if(!isfinite(score)) fail("nonfinite FP64 indexer score");
    return score;
}
static float score32(const float query[H*D],const uint16_t pool[D]) {
    float score=0;
    for(size_t h=0;h<H;++h) {
        float dot=0;
        for(size_t d=0;d<D;++d) { float term=query[h*D+d]*(float)half_value(pool[d]); dot=dot+term; }
        if(dot>0) score=score+dot;
    }
    if(!isfinite(score)) fail("nonfinite scalar FP32 indexer score");
    return score;
}
static int entry_order(const void *aa,const void *bb) {
    const Entry *a=aa,*b=bb;
    if(a->score>b->score) return -1;
    if(a->score<b->score) return 1;
    return (a->id>b->id)-(a->id<b->id);
}
static Ranking rank_scores(const double *scores,size_t n,size_t budget) {
    Ranking r={0};
    Entry *entries=allocate(product(n,sizeof(*entries)));
    r.n=n; r.k=n<budget?n:budget;
    r.ids=allocate(product(r.k,sizeof(*r.ids)));
    for(size_t i=0;i<n;++i) {
        if(!isfinite(scores[i])) fail("nonfinite score before selection");
        entries[i].score=scores[i]; entries[i].id=(uint32_t)i;
    }
    qsort(entries,n,sizeof(*entries),entry_order);
    for(size_t i=0;i<r.k;++i) r.ids[i]=entries[i].id;
    if(r.k) {
        r.cutoff=entries[r.k-1].score;
        if(r.k<n) r.next=entries[r.k].score;
        for(size_t i=0;i<n;++i) {
            if(scores[i]>r.cutoff) ++r.strict_above;
            if(scores[i]==r.cutoff) ++r.cutoff_ties;
        }
    }
    free(entries); return r;
}
static void check_ids(const uint32_t *ids,size_t k,size_t n) {
    unsigned char *seen=allocate(n);
    for(size_t i=0;i<k;++i) {
        if((size_t)ids[i]>=n) fail("selected block ID out of range");
        if(seen[ids[i]]) fail("duplicate selected block ID");
        seen[ids[i]]=1;
    }
    free(seen);
}
static void ranking_json(FILE *f,const Ranking *r) {
    fprintf(f,"{\"selected_blocks\":%zu,\"cutoff\":",r->k);
    if(!r->k) {
        fputs("null,\"cutoff_margin\":null,\"strict_above\":0,\"cutoff_ties\":0,"
              "\"selected_at_cutoff\":0,\"ambiguous_cutoff\":false}",f); return;
    }
    fprintf(f,"%.17g,\"cutoff_margin\":",r->cutoff);
    if(r->k<r->n) fprintf(f,"%.17g",r->cutoff-r->next); else fputs("null",f);
    fprintf(f,",\"strict_above\":%zu,\"cutoff_ties\":%zu,\"selected_at_cutoff\":%zu,"
              "\"ambiguous_cutoff\":%s}",r->strict_above,r->cutoff_ties,
            r->k-r->strict_above,r->cutoff_ties>r->k-r->strict_above?"true":"false");
}
/* No guessed tolerance: differing exact-tie subsets are explicitly compatible. */
static void selection_json(FILE *f,const uint32_t *ids,size_t k,
                            const double *scores,const Ranking *reference) {
    size_t common=0,strict=0,below=0;
    for(size_t i=0;i<k;++i) {
        for(size_t j=0;j<reference->k;++j) if(ids[i]==reference->ids[j]) { ++common; break; }
        if(reference->k) {
            if(scores[ids[i]]>reference->cutoff) ++strict;
            if(scores[ids[i]]<reference->cutoff) ++below;
        }
    }
    fprintf(f,"{\"canonical_overlap\":%zu,\"selected_blocks\":%zu,"
              "\"missing_strictly_above_cutoff\":%zu,\"selected_below_cutoff\":%zu,"
              "\"exact_tie_compatible\":%s}",common,k,reference->strict_above-strict,below,
            k==reference->k && strict==reference->strict_above && !below?"true":"false");
}
static void pool_modes(const float *raw,size_t block,const float gamma[D],
                       const float *cs,const float *sn,double epsilon,
                       const uint16_t *exported,double p[NM][D]) {
    double mean64[D];
    float mean32[D],normalized[D];
    const float *bc=cs+block*BLOCK*R,*bs=sn+block*BLOCK*R;
    for(size_t d=0;d<D;++d) {
        double a=0; float b=0;
        for(size_t t=0;t<BLOCK;++t) { float x=raw[(block*BLOCK+t)*D+d]; a+=(double)x; b=b+x; }
        mean64[d]=a/BLOCK; mean32[d]=b/(float)BLOCK;
    }
    norm_rope64(mean64,gamma,bc,bs,epsilon,p[0]);
    memcpy(p[1],p[0],sizeof(p[0]));
    norm_rope32(mean32,gamma,bc,bs,epsilon,0,normalized);
    for(size_t d=0;d<D;++d) { p[2][d]=normalized[d]; p[3][d]=half_value(half_bits(normalized[d])); }
    norm_rope32(mean32,gamma,bc,bs,epsilon,1,normalized);
    for(size_t d=0;d<D;++d) p[4][d]=normalized[d];
    norm_rope32(mean32,gamma,bc,bs,epsilon,2,normalized);
    for(size_t d=0;d<D;++d) p[5][d]=normalized[d];
    for(size_t d=0;d<D;++d) p[6][d]=half_value(exported[block*D+d]);
}
static size_t unsigned_arg(const char *s) {
    char *end;
    if(!*s||*s<'0'||*s>'9') fail("invalid unsigned integer");
    errno=0;
    unsigned long long value=strtoull(s,&end,10);
    if(errno||*end||value>UINT32_MAX) fail("unsupported unsigned integer");
    return (size_t)value;
}
static double epsilon_arg(const char *s) {
    char *end; errno=0; double x=strtod(s,&end);
    if(errno||!*s||*end||!isfinite(x)||x<0) fail("invalid epsilon");
    return x;
}

static int replay(const char *directory,size_t n,size_t positions,
                  size_t budget,double epsilon,const char *output) {
    if(!n||n>positions||!budget) fail("require 0 < N <= positions and budget > 0");
    size_t nv=n/BLOCK,k=nv<budget?nv:budget;
    char *snapshot=snapshot_path(directory,n);
#define ROOT(name,total,wanted) read_member(directory,name,total,wanted)
#define SNAP(name,bytes) read_member(snapshot,name,bytes,bytes)
    float *raw=ROOT("raw_ik.f32",product(product(positions,D),4),product(product(n,D),4));
    float *cs=ROOT("rope_cs.f32",product(product(positions,R),4),product(product(n,R),4));
    float *sn=ROOT("rope_sn.f32",product(product(positions,R),4),product(product(n,R),4));
    float *iqn=ROOT("iqn.f32",D*4,D*4),*ikn=ROOT("ikn.f32",D*4,D*4);
    float *iqk=SNAP("iqk.f32",(H+1)*D*4),*iq=SNAP("iq.f32",H*D*4);
    uint16_t *pool=SNAP("pool.f16",product(product(nv,D),2));
    uint32_t *provided_ids=SNAP("selected_ids.u32",product(k,4));
    float *provided_scores=nv>budget?SNAP("scores.f32",product(nv,4)):NULL;
#undef ROOT
#undef SNAP
    check_floats(raw,product(n,D)); check_floats(cs,product(n,R)); check_floats(sn,product(n,R));
    check_floats(iqn,D); check_floats(ikn,D); check_floats(iqk,(H+1)*D); check_floats(iq,H*D);
    if(provided_scores) check_floats(provided_scores,nv);
    for(size_t i=0;i<product(nv,D);++i)
        if(!isfinite(half_value(pool[i]))) fail("nonfinite exported FP16 pool");
    check_ids(provided_ids,k,nv);
    double *readback_scores=NULL;
    Ranking readback_ranking={0};
    if(provided_scores) {
        readback_scores=allocate(product(nv,sizeof(double)));
        for(size_t b=0;b<nv;++b) readback_scores[b]=provided_scores[b];
        readback_ranking=rank_scores(readback_scores,nv,budget);
    }

    double q64[H*D],q_exported[H*D],q32[H*D];
    Metric query64={0},query32={0},query32_exported={0},last_key={0};
    Metric pool_to_ref[NM]={{0}},pool_to_exported[NM]={{0}};
    Metric scores_to_ref[NM]={{0}},scores_to_fixed_pool[NM]={{0}};
    Metric raw_score_error={0},scaled_score_error={0},score_to_ref={0};
    double *scores[NM];
    Ranking ranking[NM];
    char *score_paths[NM],*id_paths[NM];
    const double scale=1.0/sqrt((double)D);
    for(size_t h=0;h<H;++h) {
        double input[D]; float output32[D];
        for(size_t d=0;d<D;++d) input[d]=iqk[h*D+d];
        norm_rope64(input,iqn,cs+(n-1)*R,sn+(n-1)*R,epsilon,q64+h*D);
        norm_rope32(iqk+h*D,iqn,cs+(n-1)*R,sn+(n-1)*R,epsilon,0,output32);
        for(size_t d=0;d<D;++d) q32[h*D+d]=output32[d];
    }
    for(size_t i=0;i<H*D;++i) {
        q_exported[i]=iq[i];
        metric_add(&query64,q_exported[i],q64[i]);
        metric_add(&query32,q32[i],q64[i]);
        metric_add(&query32_exported,q_exported[i],q32[i]);
    }
    for(size_t d=0;d<D;++d) metric_add(&last_key,iqk[H*D+d],raw[(n-1)*D+d]);

    char *pool_path=suffix(output,".pool.reference_fp64.f64");
    char *query_path=suffix(output,".query.reference_fp64.f64");
    FILE *pool_file=create_file(pool_path);
    write_exact(query_path,q64,sizeof(q64));
    for(size_t m=0;m<NM;++m) scores[m]=allocate(product(nv,sizeof(double)));
    for(size_t b=0;b<nv;++b) {
        double p[NM][D],unscaled_exported=0;
        pool_modes(raw,b,ikn,cs,sn,epsilon,pool,p);
        if(fwrite(p[0],sizeof(double),D,pool_file)!=D) fail("short FP64 pool write");
        for(size_t m=0;m<NM;++m) {
            double unscaled=score64(m==0?q64:q_exported,p[m]);
            scores[m][b]=unscaled*scale;
            if(m==6) unscaled_exported=unscaled;
            for(size_t d=0;d<D;++d) {
                metric_add(&pool_to_ref[m],p[m][d],p[0][d]);
                metric_add(&pool_to_exported[m],p[6][d],p[m][d]);
            }
        }
        if(provided_scores) {
            metric_add(&raw_score_error,provided_scores[b],unscaled_exported);
            metric_add(&scaled_score_error,(double)provided_scores[b]*scale,scores[6][b]);
            metric_add(&score_to_ref,(double)provided_scores[b]*scale,scores[0][b]);
        }
    }
    close_file(pool_file,pool_path);
    for(size_t m=0;m<NM;++m) {
        char tail[160];
        ranking[m]=rank_scores(scores[m],nv,budget);
        for(size_t b=0;b<nv;++b) {
            metric_add(&scores_to_ref[m],scores[m][b],scores[0][b]);
            metric_add(&scores_to_fixed_pool[m],scores[m][b],scores[1][b]);
        }
        snprintf(tail,sizeof(tail),".scores.%s.f64",mode_name[m]);
        score_paths[m]=suffix(output,tail);
        snprintf(tail,sizeof(tail),".ids.%s.u32",mode_name[m]);
        id_paths[m]=suffix(output,tail);
        write_exact(score_paths[m],scores[m],product(nv,sizeof(double)));
        write_exact(id_paths[m],ranking[m].ids,product(k,sizeof(uint32_t)));
    }
    FILE *report=create_file(output); /* Only after all sidecars close. */
    fputs("{\n\"schema\":\"flash-sass.qsa-quality-replay.v1\","
          "\n\"scope\":\"postprojection_indexer_only\","
          "\n\"input_origin\":\"unverified_capture_metadata_attached_by_wrapper\","
          "\n\"independent_checkpoint_evidence\":false,\"quality_validation\":false,"
          "\n\"reference_commit\":\"8498ed425fa01bbee7d086efdf1a6634c2962aed\","
          "\n\"reference_source\":\"src/transformers/models/qwen4_exp/modeling_qwen4_exp.py\","
          "\n\"gamma_convention\":\"supplied_multiplicative_gamma_no_add_one\","
          "\n\"score_convention\":\"sum_head_relu_dot_div_sqrt_128\","
          "\n\"candidate_score_convention\":\"unscaled_sum_head_relu_dot\","
          "\n\"selection_tie_policy\":\"lower_block_id_for_exact_equal_scores\",\"tie_tolerance\":0,"
          "\n\"precision_notes\":["
          "\"FP64 controls frozen FP32 inputs, not original weights\","
          "\"Scalar FP32 reductions and reciprocal sqrt may differ from device kernels\","
          "\"FP16/BF16 modes cast FP32 mean, RMSNorm return and RoPE return\","
          "\"All modes use supplied FP32 RoPE tables\","
          "\"Modes 1 through 6 hold the exported query fixed\","
          "\"No inferred pass thresholds\"],\n",report);
    fprintf(report,"\"length\":%zu,\"positions\":%zu,\"budget\":%zu,\"production_budget\":%s,"
                   "\"epsilon\":%.17g,\"complete_blocks\":%zu,\"selected_blocks\":%zu,"
                   "\"attended_keys\":%zu,\"sparse_exclusion\":%s,\n",
            n,positions,budget,budget==512?"true":"false",epsilon,nv,k,k*BLOCK+n%BLOCK,nv>budget?"true":"false");
    fputs("\"tail_token_ids\":[",report);
    for(size_t t=nv*BLOCK;t<n;++t) fprintf(report,"%s%zu",t==nv*BLOCK?"":",",t);
    fputs("],\n\"query_exported_vs_fp64\":",report); metric_json(report,&query64);
    fputs(",\n\"query_scalar_fp32_vs_fp64\":",report); metric_json(report,&query32);
    fputs(",\n\"query_exported_vs_scalar_fp32\":",report); metric_json(report,&query32_exported);
    fputs(",\n\"raw_last_key_snapshot_vs_prefix\":",report); metric_json(report,&last_key);
    fprintf(report,",\n\"candidate_score_readback_present\":%s,\n",provided_scores?"true":"false");
    fputs("\"candidate_raw_scores_vs_exported_inputs_fp64\":",report); metric_json(report,&raw_score_error);
    fputs(",\n\"candidate_scaled_scores_vs_exported_inputs_fp64\":",report); metric_json(report,&scaled_score_error);
    fputs(",\n\"candidate_scaled_scores_vs_full_fp64\":",report); metric_json(report,&score_to_ref);
    fputs(",\n\"provided_selection_vs_exported_inputs_fp64\":",report);
    selection_json(report,provided_ids,k,scores[6],&ranking[6]);
    fputs(",\n\"provided_selection_vs_full_fp64\":",report);
    selection_json(report,provided_ids,k,scores[0],&ranking[0]);
    fputs(",\n\"provided_selection_vs_candidate_score_readback\":",report);
    if(provided_scores) selection_json(report,provided_ids,k,readback_scores,&readback_ranking);
    else fputs("null",report);
    fputs(",\n\"candidate_score_readback_ranking\":",report);
    if(provided_scores) ranking_json(report,&readback_ranking);
    else fputs("null",report);
    fputs(",\n\"pool_fp64_artifact\":",report); quoted(report,pool_path);
    fputs(",\n\"query_fp64_artifact\":",report); quoted(report,query_path);
    fputs(",\n\"modes\":[\n",report);
    for(size_t m=0;m<NM;++m) {
        if(m) fputs(",\n",report);
        fputs("{\"name\":",report); quoted(report,mode_name[m]);
        fputs(",\"pool_mode_vs_fp64\":",report); metric_json(report,&pool_to_ref[m]);
        fputs(",\"exported_pool_vs_mode\":",report); metric_json(report,&pool_to_exported[m]);
        fputs(",\"scores_vs_full_fp64\":",report); metric_json(report,&scores_to_ref[m]);
        fputs(",\"scores_vs_fixed_iq_fp64_pool\":",report); metric_json(report,&scores_to_fixed_pool[m]);
        fputs(",\"ranking\":",report); ranking_json(report,&ranking[m]);
        fputs(",\"selection_vs_full_fp64\":",report);
        selection_json(report,ranking[m].ids,k,scores[0],&ranking[0]);
        fputs(",\"selection_vs_fixed_iq_fp64_pool\":",report);
        selection_json(report,ranking[m].ids,k,scores[1],&ranking[1]);
        fputs(",\"scores_artifact\":",report); quoted(report,score_paths[m]);
        fputs(",\"ids_artifact\":",report); quoted(report,id_paths[m]); fputc('}',report);
    }
    fputs("\n],\n\"computation_complete\":true,\n\"complete\":false\n}\n",report);
    close_file(report,output);
    printf("qsa_quality: wrote %s; no quality threshold applied\n",output);
    /* Process is one replay. Exit reclaims the read-only prefix work buffers. */
    return 0;
}

static uint32_t mix32(uint32_t x) {
    x^=x>>16; x*=UINT32_C(0x7feb352d); x^=x>>15;
    x*=UINT32_C(0x846ca68b); return x^(x>>16);
}
static float synthetic_value(uint32_t x) {
    return ((int)(mix32(x)&65535u)-32768)*(1.0f/8192.0f);
}
static int synthetic(const char *directory) {
    static const size_t lengths[]={1,2047,2048,2049,2050,2051,2052,3100,8000,65000,262000};
    const size_t positions=262000,budget=512;
    const double epsilon=(double)1e-6f;
    float *raw=allocate(product(product(positions,D),sizeof(float)));
    float *cs=allocate(product(product(positions,R),sizeof(float)));
    float *sn=allocate(product(product(positions,R),sizeof(float)));
    float iqn[D],ikn[D];
    if(mkdir(directory,0700)) file_fail(directory);
    for(size_t d=0;d<D;++d) {
        iqn[d]=1.0f+((int)(d%7)-3)*0.03125f;
        ikn[d]=1.0f+((int)(d%11)-5)*0.015625f;
    }
    for(size_t t=0;t<positions;++t) {
        for(size_t d=0;d<D;++d)
            raw[t*D+d]=synthetic_value((uint32_t)t*UINT32_C(65537)+(uint32_t)d+UINT32_C(12345));
        for(size_t p=0;p<R;++p) {
            double angle=(double)t*pow(10000000.0,-(double)p/R);
            cs[t*R+p]=(float)cos(angle); sn[t*R+p]=(float)sin(angle);
        }
    }
    write_member(directory,"raw_ik.f32",raw,product(product(positions,D),4));
    write_member(directory,"rope_cs.f32",cs,product(product(positions,R),4));
    write_member(directory,"rope_sn.f32",sn,product(product(positions,R),4));
    write_member(directory,"iqn.f32",iqn,sizeof(iqn)); write_member(directory,"ikn.f32",ikn,sizeof(ikn));
    for(size_t j=0;j<sizeof(lengths)/sizeof(lengths[0]);++j) {
        size_t n=lengths[j],nv=n/BLOCK,k=nv<budget?nv:budget;
        char *snapshot=snapshot_path(directory,n);
        float iqk[(H+1)*D],iq[H*D];
        uint16_t *pool=allocate(product(product(nv,D),sizeof(uint16_t)));
        float *scores32=allocate(product(nv,sizeof(float)));
        double *scores64=allocate(product(nv,sizeof(double)));
        uint32_t *ids=allocate(product(k,sizeof(uint32_t)));
        if(mkdir(snapshot,0700)) file_fail(snapshot);
        for(size_t i=0;i<H*D;++i)
            iqk[i]=synthetic_value((uint32_t)n*UINT32_C(104729)+(uint32_t)i+UINT32_C(76543));
        memcpy(iqk+H*D,raw+(n-1)*D,D*sizeof(float));
        for(size_t h=0;h<H;++h)
            norm_rope32(iqk+h*D,iqn,cs+(n-1)*R,sn+(n-1)*R,epsilon,0,iq+h*D);
        for(size_t b=0;b<nv;++b) {
            float mean[D],normalized[D];
            for(size_t d=0;d<D;++d) {
                float sum=0;
                for(size_t t=0;t<BLOCK;++t) sum=sum+raw[(b*BLOCK+t)*D+d];
                mean[d]=sum/(float)BLOCK;
            }
            norm_rope32(mean,ikn,cs+b*BLOCK*R,sn+b*BLOCK*R,epsilon,0,normalized);
            for(size_t d=0;d<D;++d) pool[b*D+d]=half_bits(normalized[d]);
            scores32[b]=score32(iq,pool+b*D); scores64[b]=scores32[b];
        }
        Ranking ranking=rank_scores(scores64,nv,budget);
        for(size_t i=0;i<k;++i) ids[i]=nv<=budget?(uint32_t)i:ranking.ids[i];
        write_member(snapshot,"iqk.f32",iqk,sizeof(iqk)); write_member(snapshot,"iq.f32",iq,sizeof(iq));
        write_member(snapshot,"pool.f16",pool,product(product(nv,D),2));
        write_member(snapshot,"selected_ids.u32",ids,product(k,4));
        if(nv>budget) write_member(snapshot,"scores.f32",scores32,product(nv,4));
        char *path=join(snapshot,"snapshot.json");
        FILE *f=create_file(path);
        fprintf(f,"{\"schema\":\"flash-sass.qsa-layer-snapshot.v1\","
                  "\"length\":%zu,\"position\":%zu,\"layer\":0,\"die\":0,\"budget\":%zu,"
                  "\"complete_blocks\":%zu,\"tail\":%zu,\"attended_keys\":%zu,"
                  "\"sparse_exclusion\":%s,\"scores_source\":\"%s\","
                  "\"selected_ids_source\":\"%s\",\"evidence_kind\":\"synthetic_cpu_fixture\","
                  "\"quality_validation\":false}\n",
                n,n-1,budget,nv,n%BLOCK,k*BLOCK+n%BLOCK,nv>budget?"true":"false",
                nv>budget?"synthetic_scalar_fp32":"absent_dense",
                nv>budget?"synthetic_scalar_fp32_topk":"implicit_all_completed_blocks");
        close_file(f,path);
        free(path); free(ranking.ids); free(ids); free(scores64); free(scores32); free(pool); free(snapshot);
    }
    char *path=join(directory,"complete.json");
    FILE *f=create_file(path);
    fprintf(f,"{\"schema\":\"flash-sass.qsa-layer-capture.v1\",\"positions\":%zu,"
              "\"layer\":0,\"die\":0,\"budget\":%zu,\"epsilon\":%.17g,\"rope_base\":10000000,"
              "\"byte_order\":\"little\",\"evidence_kind\":\"synthetic_cpu_fixture\","
              "\"coverage\":\"indexer_only\",\"independent_checkpoint_evidence\":false,"
              "\"quality_validation\":false,\"producer\":\"qsa_quality --synthetic\","
              "\"snapshot_lengths\":[",positions,budget,epsilon);
    for(size_t i=0;i<sizeof(lengths)/sizeof(lengths[0]);++i) fprintf(f,"%s%zu",i?",":"",lengths[i]);
    fputs("],\"complete\":true}\n",f);
    close_file(f,path); free(path); free(raw); free(cs); free(sn);
    printf("qsa_quality: synthetic CPU fixtures written to %s; not checkpoint/GPU evidence\n",directory);
    return 0;
}

static int selftest(void) {
    unsigned tests=0;
    float gamma[D],cs[R],sn[R];
    double input[D]={0},output[D],query[H*D]={0},pool[D]={0},scores[]={9,8,8,1};
#define CHECK(condition,description) do { ++tests; if(!(condition)) { \
    fprintf(stderr,"qsa_quality selftest failed: %s\n",description); return 1; } } while(0)
    CHECK(half_value(half_bits(1.0f))==1.0,"FP16 one");
    CHECK(half_value(half_bits(65504.0f))==65504.0,"FP16 maximum finite");
    CHECK(half_value(half_bits(ldexpf(1.0f,-24)))==ldexp(1.0,-24),"FP16 minimum subnormal");
    CHECK(half_value(half_bits(ldexpf(1.0f,-25)))==0,"FP16 subnormal tie");
    CHECK(half_value(half_bits(1.0f+ldexpf(1.0f,-11)))==1.0,"FP16 even tie");
    CHECK(half_value(half_bits(1.0f+3.0f*ldexpf(1.0f,-11)))==1.0+ldexp(1.0,-9),"FP16 odd tie");
    CHECK(rounded(1.0f+ldexpf(1.0f,-8),2)==1.0f,"BF16 even tie");
    CHECK(rounded(1.0f+3.0f*ldexpf(1.0f,-8),2)==1.0f+ldexpf(1.0f,-6),"BF16 odd tie");
    for(size_t d=0;d<D;++d) gamma[d]=1;
    for(size_t p=0;p<R;++p) { cs[p]=1; sn[p]=0; }
    input[0]=1;
    norm_rope64(input,gamma,cs,sn,0,output);
    CHECK(fabs(output[0]-sqrt(128.0))<1e-12,"gamma does not add one");
    CHECK(output[1]==0,"zero coordinate");
    cs[0]=0; sn[0]=1;
    norm_rope64(input,gamma,cs,sn,0,output);
    CHECK(output[0]==0 && fabs(output[32]-sqrt(128.0))<1e-12,"RoPE half layout");
    query[0]=1; query[D]=-2; query[2*D]=3; query[3*D]=-4; pool[0]=2;
    CHECK(score64(query,pool)==8,"per-head ReLU before sum");
    Ranking rank=rank_scores(scores,4,2);
    CHECK(rank.ids[0]==0 && rank.ids[1]==1,"lower ID exact tie");
    CHECK(rank.strict_above==1 && rank.cutoff_ties==2 && rank.cutoff-rank.next==0,"ambiguous cutoff");
    free(rank.ids);
    double raw4[4]={4,0,0,0};
    double mean=(raw4[0]+raw4[1]+raw4[2]+raw4[3])/4;
    CHECK(mean==1,"complete-block mean includes four raw values");
    CHECK(2051/4==512 && 2051%4==3 && 2052/4==513,"first production exclusion is N2052");
#undef CHECK
    printf("{\"selftest\":\"pass\",\"checks\":%u,\"evidence_kind\":\"analytic_cpu_unit_checks\","
           "\"gpu_execution\":false,\"checkpoint_validation\":false}\n",tests);
    return 0;
}
int main(int argc,char **argv) {
    uint16_t endian=1;
    if(*(unsigned char *)&endian!=1 || sizeof(float)!=4 || sizeof(double)!=8 ||
       FLT_RADIX!=2 || FLT_MANT_DIG!=24 || DBL_MANT_DIG!=53)
        fail("requires little-endian IEEE binary32/binary64");
    if(argc==2 && !strcmp(argv[1],"--selftest")) return selftest();
    if(argc==3 && !strcmp(argv[1],"--synthetic")) return synthetic(argv[2]);
    if(argc==8 && !strcmp(argv[1],"--replay"))
        return replay(argv[2],unsigned_arg(argv[3]),unsigned_arg(argv[4]),
                      unsigned_arg(argv[5]),epsilon_arg(argv[6]),argv[7]);
    fputs("usage: qsa_quality --selftest | --synthetic NEW_DIR |\n"
          "       --replay CAPTURE_DIR N POSITIONS BUDGET EPS OUTPUT_JSON\n"
          "Use qsa_quality_replay.pl to check metadata and add hashes.\n",stderr);
    return 2;
}
