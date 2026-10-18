#define _POSIX_C_SOURCE 200809L
#include "qsa_capture.h"
#include "attn.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define ZERO UINT64_C(0x100000000)
#define IDS UINT64_C(0x200000000)
#define LOGITS UINT64_C(0x300000000)
#define SCORES UINT64_C(0x400000000)
#define SELECT UINT64_C(0x500000000)
#define POOL UINT64_C(0x600000000)
#define KC UINT64_C(0x700000000)
#define VC UINT64_C(0x800000000)
static unsigned checks;
#define CHECK(expr) do { ++checks; if (!(expr)) { fprintf(stderr,"line %d: %s failed\n",__LINE__,#expr); exit(1); } } while (0)
typedef struct {
    uint32_t ids[2052], select[512];
    float scores[513], *kc;
    uint16_t pool[17*4096];
    unsigned visible, read_score, read_selection;
    int id_mismatch, nan_logit, read_error;
} fixture;
static int readback(void *arg, unsigned die, uint64_t address, void *out, size_t bytes) {
    fixture *f=arg;
    if (die>1 || f->read_error) return -1;
    switch (address) {
        case IDS:
            if (bytes!=(size_t)f->visible*4) return -1;
            memcpy(out,f->ids,bytes);
            if (f->id_mismatch && die==1) ((uint32_t *)out)[0]=2;
            return 0;
        case LOGITS:
            if (bytes>4u*QSA_CAPTURE_VOCAB || bytes%4) return -1;
            for (size_t i=0;i<bytes/4;i++) ((float *)out)[i]=1.0f;
            if (f->nan_logit) ((float *)out)[0]=NAN;
            return 0;
        case SCORES:
            ++f->read_score;
            if (f->visible!=2052 || bytes!=sizeof f->scores) return -1;
            memcpy(out,f->scores,bytes); return 0;
        case SELECT:
            ++f->read_selection;
            if (f->visible!=2052 || bytes!=sizeof f->select) return -1;
            memcpy(out,f->select,bytes); return 0;
        case POOL:
            if (bytes>sizeof f->pool) return -1;
            memcpy(out,f->pool,bytes); return 0;
        case KC:
            if (bytes>ATTN_KC_FLOATS(2048)*4) return -1;
            memcpy(out,f->kc,bytes); return 0;
        case VC:
            if (bytes>2048*512*4) return -1;
            memset(out,0,bytes); return 0;
        default:
            if (address<ZERO || address>ZERO+2052*512 || bytes>512*1024) return -1;
            memset(out,0,bytes); return 0;
    }
}
static qsa_capture_plan plan_for(fixture *f, unsigned visible) {
    f->visible=visible;
    qsa_capture_plan p={.visible=visible,.nblk=512,.dies=2,.ids=f->ids,.revision="synthetic-not-a-checkpoint"};
    p.die[0]=(qsa_capture_die){.offset=0,.count=QSA_CAPTURE_VOCAB/2,.logits=LOGITS,.ids=IDS};
    p.die[1]=(qsa_capture_die){.offset=QSA_CAPTURE_VOCAB/2,.count=QSA_CAPTURE_VOCAB/2,.logits=LOGITS,.ids=IDS};
    for (unsigned i=0;i<QSA_CAPTURE_LAYERS;i++) {
        qsa_capture_layer *l=&p.layer[i]; l->layer=4*i+3; l->die=i%2;
#define SET_ZERO(field) l->field=ZERO;
        QSA_CAPTURE_SCRATCH_FIELDS(SET_ZERO)
#undef SET_ZERO
        l->sc=SCORES; l->sel=SELECT; l->pkt=POOL; l->kc=KC; l->vc=VC;
        l->ring=l->kq=l->vq=l->ks=l->vs=ZERO;
        l->qnorm=l->knorm=l->iqnorm=l->iknorm=ZERO;
    }
    return p;
}
static void filename(char *out, size_t n, const char *root, const char *leaf) {
    int size=snprintf(out,n,"%s/%s",root,leaf); CHECK(size>0 && (size_t)size<n);
}
static off_t file_size(const char *root, const char *leaf) {
    char path[4096]; filename(path,sizeof path,root,leaf); struct stat st;
    return stat(path,&st) ? (off_t)-1 : st.st_size;
}
static void read_file(const char *root, const char *leaf, void *buf, size_t n) {
    char path[4096]; filename(path,sizeof path,root,leaf); FILE *f=fopen(path,"rb");
    CHECK(f!=NULL); CHECK(fread(buf,1,n,f)==n); CHECK(fgetc(f)==EOF); CHECK(!fclose(f));
}
static int run(fixture *f, const qsa_capture_plan *p, const char *root, const char *leaf, char *dir, char *err) {
    filename(dir,4096,root,leaf);
    return qsa_capture_write(p,readback,f,dir,err,256);
}
int main(int argc,char **argv) {
    if (argc!=2) { fprintf(stderr,"usage: %s NEW_DIR\n",argv[0]); return 2; }
    CHECK(!mkdir(argv[1],0700));
    fixture *f=calloc(1,sizeof *f); CHECK(f!=NULL);
    f->kc=calloc(ATTN_KC_FLOATS(2048),4); CHECK(f->kc!=NULL);
    for (unsigned i=0;i<2052;i++) f->ids[i]=1;
    for (unsigned i=0;i<513;i++) f->scores[i]=(float)i;
    for (unsigned i=0;i<512;i++) f->select[i]=i+1;
    for (unsigned b=0;b<513;b++) for (unsigned d=0;d<128;d++) {
        size_t at=(size_t)(b>>5)*4096+(size_t)(d>>2)*128+(b&31)*4+(d&3);
        f->pool[at]=(uint16_t)(0x3c00+(b%64)*8+d%8);
    }
    for (unsigned j=0;j<2048;j++) for (int kv=0;kv<2;kv++) for (int d=0;d<256;d++)
        f->kc[attn_kidx(j,kv,d)]=(float)(1000*j+256*(unsigned)kv+(unsigned)d);
    char dir[4096],err[256];
    qsa_capture_plan p=plan_for(f,4);
    CHECK(!run(f,&p,argv[1],"dense",dir,err));
    CHECK(!f->read_score && !f->read_selection);
    CHECK(file_size(dir,"capture.json")>0);
    CHECK(file_size(dir,"logits.f32")==4u*QSA_CAPTURE_VOCAB);
    CHECK(file_size(dir,"prefix_ids.u32")==16);
    CHECK(file_size(dir,"row_ids.u32")==4);
    CHECK(file_size(dir,"layer-03-scores.f32")==-1);
    uint32_t selected; read_file(dir,"layer-03-selected.u32",&selected,4); CHECK(selected==0);
    uint16_t pooled[128]; read_file(dir,"layer-03-pool.f16",pooled,sizeof pooled);
    CHECK(pooled[7]==0x3c07 && pooled[8]==0x3c00);
    float compact[4*512]; read_file(dir,"layer-03-selected-k.f32",compact,sizeof compact);
    CHECK(compact[0]==0 && compact[513]==1001 && compact[2047]==3511);
    CHECK(qsa_capture_write(&p,readback,f,dir,err,sizeof err)==-1);
    CHECK(file_size(dir,"capture.json")>0); /* old evidence survives refusal */
    p=plan_for(f,2052);
    CHECK(!run(f,&p,argv[1],"sparse",dir,err));
    CHECK(f->read_score==12 && f->read_selection==12);
    CHECK(file_size(dir,"layer-47-scores.f32")==513*4);
    CHECK(file_size(dir,"layer-47-pool.f16")==513*256);
    CHECK(file_size(dir,"layer-47-selected-k.f32")==2048*512*4);
    CHECK(file_size(dir,"layer-47-selected-v.f32")==2048*512*4);
    uint32_t sparse_ids[512]; read_file(dir,"layer-47-selected.u32",sparse_ids,sizeof sparse_ids);
    CHECK(sparse_ids[0]==1 && sparse_ids[511]==512);
    uint16_t *sparse_pool=malloc(513*256); CHECK(sparse_pool!=NULL);
    read_file(dir,"layer-47-pool.f16",sparse_pool,513*256);
    CHECK(sparse_pool[32*128+7]==0x3d07 && sparse_pool[512*128+7]==0x3c07); free(sparse_pool);
    f->select[1]=f->select[0];
    CHECK(run(f,&p,argv[1],"duplicate-selection",dir,err)==-1);
    CHECK(strstr(err,"selected block IDs")!=NULL && file_size(dir,"capture.json")==-1);
    f->select[1]=2;
    p=plan_for(f,1); f->id_mismatch=1;
    CHECK(run(f,&p,argv[1],"different-prefix",dir,err)==-1);
    CHECK(strstr(err,"different teacher-forced IDs")!=NULL && file_size(dir,"capture.json")==-1);
    f->id_mismatch=0; f->nan_logit=1;
    CHECK(run(f,&p,argv[1],"nonfinite-logit",dir,err)==-1);
    CHECK(strstr(err,"nonfinite")!=NULL && file_size(dir,"capture.json")==-1);
    f->nan_logit=0; f->read_error=1;
    CHECK(run(f,&p,argv[1],"failed-read",dir,err)==-1);
    CHECK(file_size(dir,"capture.json")==-1); f->read_error=0;
    p.die[1].offset++;
    CHECK(run(f,&p,argv[1],"gap",dir,err)==-1); CHECK(access(dir,F_OK)!=0);
    p=plan_for(f,1); p.layer[0].layer=7;
    CHECK(run(f,&p,argv[1],"missing-layer",dir,err)==-1); CHECK(access(dir,F_OK)!=0);
    p=plan_for(f,1); p.layer[0].kq=UINT64_MAX;
    CHECK(run(f,&p,argv[1],"address-overflow",dir,err)==-1); CHECK(access(dir,F_OK)!=0);
    p=plan_for(f,1); p.nblk=1;
    CHECK(run(f,&p,argv[1],"wrong-budget",dir,err)==-1); CHECK(access(dir,F_OK)!=0);
    p=plan_for(f,1); f->ids[0]=QSA_CAPTURE_VOCAB;
    CHECK(run(f,&p,argv[1],"invalid-token",dir,err)==-1); CHECK(access(dir,F_OK)!=0);
    free(f->kc); free(f);
    printf("PASS_SYNTHETIC_CAPTURE_IO: %u assertions; dense/no-readback, sparse, pool/K detiling, full vocabulary, per-die IDs, failure closure and no overwrites\n",checks);
    return 0;
}
