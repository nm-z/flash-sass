#define _POSIX_C_SOURCE 200809L
#include "qsa_capture.h"
#include "attn.h"
#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
    int dir;
    FILE *catalog;
    unsigned files;
    qsa_capture_read read;
    void *arg;
    char *error;
    size_t error_size;
} sink;
static int fail(sink *s, const char *fmt, ...) {
    if (s->error && s->error_size) {
        va_list ap; va_start(ap, fmt);
        vsnprintf(s->error, s->error_size, fmt, ap); va_end(ap);
    }
    return -1;
}
static void quoted(FILE *f, const char *text) {
    fputc('"', f);
    for (const unsigned char *p = (const unsigned char *)text; *p; ++p) {
        if (*p == '"' || *p == '\\') fprintf(f, "\\%c", *p);
        else if (*p < 32 || *p >= 127) fprintf(f, "\\u%04x", *p);
        else fputc(*p, f);
    }
    fputc('"', f);
}
static FILE *create(sink *s, const char *name) {
    int fd = openat(s->dir, name, O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW, 0600);
    if (fd < 0) { fail(s, "create %s: %s", name, strerror(errno)); return NULL; }
    FILE *f = fdopen(fd, "wb");
    if (!f) { close(fd); unlinkat(s->dir, name, 0); fail(s, "fdopen %s failed", name); }
    return f;
}
static int save(sink *s, const char *name, const char *dtype, const char *layout,
                const void *data, size_t bytes, int finite_f32) {
    if (finite_f32) {
        if (bytes % 4) return fail(s, "unaligned f32 buffer %s", name);
        const float *v = data;
        for (size_t i = 0; i < bytes/4; ++i)
            if (!isfinite(v[i])) return fail(s, "nonfinite %s element %zu", name, i);
    }
    FILE *f = create(s, name);
    if (!f) return -1;
    int bad = bytes && fwrite(data, 1, bytes, f) != bytes;
    if (fclose(f)) bad = 1;
    if (bad) { unlinkat(s->dir, name, 0); return fail(s, "write %s failed", name); }
    fputs("{\"file\":", s->catalog); quoted(s->catalog, name);
    fputs(",\"dtype\":", s->catalog); quoted(s->catalog, dtype);
    fputs(",\"layout\":", s->catalog); quoted(s->catalog, layout);
    fprintf(s->catalog, ",\"bytes\":%zu}\n", bytes);
    if (ferror(s->catalog)) return fail(s, "tensor catalog write failed");
    ++s->files; return 0;
}
static int get(sink *s, unsigned die, uint64_t address, void *out, size_t bytes) {
    if (!bytes) return 0;
    if (!address || address > UINT64_MAX - bytes || s->read(s->arg, die, address, out, bytes))
        return fail(s, "device readback failed: die %u, %zu bytes", die, bytes);
    return 0;
}
static int device_file(sink *s, const qsa_capture_layer *l, const char *name,
                       const char *dtype, const char *layout, uint64_t ptr, size_t bytes) {
    void *h = malloc(bytes ? bytes : 1);
    if (!h) return fail(s, "allocation for %s failed", name);
    char file[96]; snprintf(file, sizeof file, "layer-%02u-%s", l->layer, name);
    int rc = get(s, l->die, ptr, h, bytes);
    if (!rc) rc = save(s, file, dtype, layout, h, bytes, !strcmp(dtype, "f32"));
    free(h); return rc;
}
static int layer_files(sink *s, const qsa_capture_plan *p, const qsa_capture_layer *l) {
    struct tensor { const char *name, *dtype, *layout; uint64_t ptr; size_t bytes; } t[] = {
        {"qg.f32","f32","[24][q256,gate256]",l->qg,12288*4},
        {"k.f32","f32","[2][256]",l->k,512*4},
        {"v.f32","f32","[2][256]",l->v,512*4},
        {"qo.f32","f32","[24][256]",l->qo,6144*4},
        {"iqk.f32","f32","[four q heads x128, raw k128]",l->iqk,640*4},
        {"iq.f32","f32","[4][128]",l->iq,512*4},
        {"pm.f32","f32","[24][8]",l->pm,192*4},
        {"pd.f32","f32","[24][8]",l->pd,192*4},
        {"po.f32","f32","[24][8][256]",l->po,49152*4},
        {"attention.f32","f32","[24][256], sigmoid-gated",l->aout,6144*4},
        {"raw-key-ring.f32","f32","[4][128], slot=position%4",l->ring,512*4},
        {"qnorm.f32","f32","[256], loaded gamma",l->qnorm,256*4},
        {"knorm.f32","f32","[256], loaded gamma",l->knorm,256*4},
        {"iqnorm.f32","f32","[128], loaded gamma",l->iqnorm,128*4},
        {"iknorm.f32","f32","[128], loaded gamma",l->iknorm,128*4},
        {"current-k.i8","i8","[2][256]",l->kq+(uint64_t)(p->visible-1)*512,512},
        {"current-v.i8","i8","[2][256]",l->vq+(uint64_t)(p->visible-1)*512,512},
        {"current-ks.f16","f16","[2][8]",l->ks+(uint64_t)(p->visible-1)*32,32},
        {"current-vs.f16","f16","[2][8]",l->vs+(uint64_t)(p->visible-1)*32,32}
    };
    for (size_t i = 0; i < sizeof t/sizeof *t; ++i)
        if (device_file(s,l,t[i].name,t[i].dtype,t[i].layout,t[i].ptr,t[i].bytes)) return -1;
    unsigned complete = p->visible/4, selected = complete < p->nblk ? complete : p->nblk;
    unsigned keys = 4*selected + p->visible%4;
    uint32_t *ids = malloc((size_t)(selected+1)*4);
    size_t packed_pool_bytes = (size_t)((complete+31)/32)*8192;
    uint16_t *packed = malloc(packed_pool_bytes ? packed_pool_bytes : 1);
    uint16_t *pool = malloc(complete ? (size_t)complete*256 : 1);
    size_t kc_bytes = ATTN_KC_FLOATS(keys)*4;
    float *kc = malloc(kc_bytes), *linear_k = malloc((size_t)keys*512*4);
    if (!ids || !packed || !pool || !kc || !linear_k) {
        free(ids); free(packed); free(pool); free(kc); free(linear_k);
        return fail(s,"layer capture allocation failed");
    }
    int rc = -1;
    if (complete > p->nblk) {
        if (get(s,l->die,l->sel,ids,(size_t)selected*4)) goto done;
        for (unsigned i=0;i<selected;++i) if (ids[i]>=complete || (i && ids[i]<=ids[i-1])) {
            fail(s,"invalid observed selected block IDs at layer %u",l->layer); goto done;
        }
        if (device_file(s,l,"scores.f32","f32","[complete_blocks], unscaled production FP32",l->sc,(size_t)complete*4)) goto done;
    } else {
        /* The dense fast path does not write score/selection scratch. Never
         * present old bytes as measured scores or device-observed IDs. */
        for (unsigned i=0;i<selected;++i) ids[i]=i;
    }
    char file[96]; snprintf(file,sizeof file,"layer-%02u-selected.u32",l->layer);
    if (save(s,file,"u32",complete>p->nblk ? "observed block IDs, ascending" : "implicit dense block IDs, host enumeration",ids,(size_t)selected*4,0)) goto done;
    if (get(s,l->die,l->pkt,packed,packed_pool_bytes)) goto done;
    for (unsigned b=0;b<complete;++b) for (unsigned d=0;d<128;++d) {
        size_t index=(size_t)(b>>5)*4096+(size_t)(d>>2)*128+(b&31)*4+(d&3);
        uint16_t bits=packed[index];
        if ((bits&0x7c00u)==0x7c00u) { fail(s,"nonfinite pooled key at layer %u",l->layer); goto done; }
        pool[(size_t)b*128+d]=bits;
    }
    snprintf(file,sizeof file,"layer-%02u-pool.f16",l->layer);
    if (save(s,file,"f16","[complete_blocks][128], padding excluded",pool,(size_t)complete*256,0)) goto done;
    if (get(s,l->die,l->kc,kc,kc_bytes)) goto done;
    for (unsigned j=0;j<keys;++j) for (int kv=0;kv<2;++kv) for (int d=0;d<256;++d)
        linear_k[(size_t)j*512+kv*256+d]=kc[attn_kidx(j,kv,d)];
    snprintf(file,sizeof file,"layer-%02u-selected-k.f32",l->layer);
    if (save(s,file,"f32","[attended_keys][2][256], compact gather order",linear_k,(size_t)keys*512*4,1)) goto done;
    if (device_file(s,l,"selected-v.f32","f32","[attended_keys][2][256], compact gather order",l->vc,(size_t)keys*512*4)) goto done;
    rc=0;
done:
    free(ids); free(packed); free(pool); free(kc); free(linear_k); return rc;
}
int qsa_capture_write(const qsa_capture_plan *p, qsa_capture_read readback,
                      void *arg, const char *new_dir, char *error, size_t error_size) {
    sink s={.dir=-1,.read=readback,.arg=arg,.error=error,.error_size=error_size};
    uint32_t one=1;
    if (error && error_size) error[0]=0;
    if (!p || !readback || !new_dir || !*new_dir || !p->ids || !p->revision ||
        !p->visible || p->visible>QSA_CAPTURE_MAX_VISIBLE || p->nblk!=512 ||
        !p->dies || p->dies>QSA_CAPTURE_MAX_DIES || *(unsigned char*)&one!=1 ||
        sizeof(float)!=4 || FLT_RADIX!=2 || FLT_MANT_DIG!=24)
        return fail(&s,"invalid full-model capture plan or host arithmetic");
    for (unsigned i=0;i<p->visible;++i) if (p->ids[i]>=QSA_CAPTURE_VOCAB)
        return fail(&s,"input token outside vocabulary at %u",i);
    unsigned offset=0;
    for (unsigned d=0;d<p->dies;++d) {
        if (p->die[d].offset!=offset || !p->die[d].count ||
            p->die[d].count>QSA_CAPTURE_VOCAB-offset || !p->die[d].logits || !p->die[d].ids)
            return fail(&s,"logit slices do not cover the full vocabulary exactly");
        offset+=p->die[d].count;
    }
    if (offset!=QSA_CAPTURE_VOCAB) return fail(&s,"incomplete vocabulary coverage");
    for (unsigned i=0;i<QSA_CAPTURE_LAYERS;++i) {
        const qsa_capture_layer *l=p->layer+i;
        if (l->layer!=4*i+3 || l->die>=p->dies) return fail(&s,"invalid QSA layer ownership");
#define VALID_PTR(field) if (!l->field) return fail(&s,"missing QSA buffer " #field);
        QSA_CAPTURE_SCRATCH_FIELDS(VALID_PTR)
        VALID_PTR(pkt) VALID_PTR(ring) VALID_PTR(kq) VALID_PTR(vq) VALID_PTR(ks) VALID_PTR(vs)
        VALID_PTR(qnorm) VALID_PTR(knorm) VALID_PTR(iqnorm) VALID_PTR(iknorm)
#undef VALID_PTR
        if (l->kq>UINT64_MAX-(uint64_t)p->visible*512 || l->vq>UINT64_MAX-(uint64_t)p->visible*512 ||
            l->ks>UINT64_MAX-(uint64_t)p->visible*32 || l->vs>UINT64_MAX-(uint64_t)p->visible*32)
            return fail(&s,"cache address overflow");
    }
    if (mkdir(new_dir,0700)) return fail(&s,"create new directory: %s",strerror(errno));
    s.dir=open(new_dir,O_RDONLY|O_DIRECTORY|O_NOFOLLOW);
    if (s.dir<0) return fail(&s,"open capture directory: %s",strerror(errno));
    s.catalog=create(&s,"tensors.jsonl");
    float *logits=malloc(4u*QSA_CAPTURE_VOCAB);
    uint32_t *device_ids=malloc((size_t)p->visible*4);
    int rc=-1;
    if (!s.catalog || !logits || !device_ids) { fail(&s,"capture initialization failed"); goto done; }
    for (unsigned d=0;d<p->dies;++d) {
        if (get(&s,d,p->die[d].ids,device_ids,(size_t)p->visible*4)) goto done;
        if (memcmp(device_ids,p->ids,(size_t)p->visible*4)) {
            fail(&s,"device %u consumed different teacher-forced IDs",d); goto done;
        }
        if (get(&s,d,p->die[d].logits,logits+p->die[d].offset,(size_t)p->die[d].count*4)) goto done;
    }
    if (save(&s,"prefix_ids.u32","u32","[visible], exact consumed prefix",p->ids,(size_t)p->visible*4,0) ||
        save(&s,"row_ids.u32","u32","[1], final input ID",p->ids+p->visible-1,4,0) ||
        save(&s,"logits.f32","f32","[1][248320], final position, full vocabulary",logits,4u*QSA_CAPTURE_VOCAB,1)) goto done;
    for (unsigned i=0;i<QSA_CAPTURE_LAYERS;++i) if (layer_files(&s,p,p->layer+i)) goto done;
    { FILE *catalog=s.catalog; s.catalog=NULL;
      int bad=ferror(catalog); if (fclose(catalog)) bad=1;
      if (bad) { fail(&s,"tensor catalog close failed"); goto done; } }
    FILE *manifest=create(&s,"capture.json");
    if (!manifest) goto done;
    fputs("{\n\"schema\":\"flash-sass.candidate-forward-capture.v1\",\n"
          "\"status\":\"CAPTURED_NOT_VALIDATED\",\n\"source_revision\":",manifest);
    quoted(manifest,p->revision);
    fprintf(manifest,",\n\"cubin_fnv1a64\":\"%016" PRIx64 "\",\n"
            "\"visible\":%u,\"logit_position\":%u,\"logit_rows\":1,\"vocab\":%u,\n"
            "\"model_layers\":48,\"qsa_layers\":12,\"nblk\":512,\"dies\":%u,\n"
            "\"kv_host\":%s,\"ple_disk\":%s,\"teacher_forced\":true,\n"
            "\"original_checkpoint_lineage_verified\":false,\"independent_checkpoint_acceptance\":false,\n"
            "\"diagnostic_scratch_isolated\":true,\"capture_disabled_parity_measured\":false,\n"
            "\"byte_order\":\"little\",\"tensor_catalog\":\"tensors.jsonl\",\"files\":%u,\n"
            "\"complete_blocks\":%u,\"tail\":%u,\"selected_blocks\":%u,\n"
            "\"selection_mode\":\"%s\",\"score_scaling\":\"unscaled FP32\",\n"
            "\"per_layer\":[\n",
            p->cubin_fnv,p->visible,p->visible-1,QSA_CAPTURE_VOCAB,p->dies,
            p->kv_host?"true":"false",p->ple_disk?"true":"false",s.files,
            p->visible/4,p->visible%4,p->visible/4<512?p->visible/4:512,
            p->visible/4>512?"observed_device_top512":"implicit_dense_no_score_readback");
    for (unsigned i=0;i<QSA_CAPTURE_LAYERS;++i)
        fprintf(manifest,"%s{\"layer\":%u,\"die\":%u}",i?",\n":"",p->layer[i].layer,p->layer[i].die);
    fputs("\n],\"complete\":true\n}\n",manifest);
    int bad=ferror(manifest); if (fclose(manifest)) bad=1;
    if (bad) { unlinkat(s.dir,"capture.json",0); fail(&s,"capture manifest write failed"); goto done; }
    rc=0;
done:
    if (s.catalog && fclose(s.catalog)) { fail(&s,"tensor catalog close failed"); rc=-1; }
    if (rc) unlinkat(s.dir,"capture.json",0);
    if (close(s.dir)) { fail(&s,"capture directory close failed"); rc=-1; }
    free(logits); free(device_ids); return rc;
}
