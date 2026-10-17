#ifndef FLASH_MODEL_CAPTURE_H
#define FLASH_MODEL_CAPTURE_H
#include <errno.h>
#include <sys/stat.h>
#ifndef FLASH_REV
#define FLASH_REV "unknown"
#endif
static FILE *model_capture_logits;
static int model_capture_enabled(void){const char *p=getenv("MODEL_CAPTURE");return p&&*p;}
static void model_capture_check(int b0,int b1,int embed,int head,int loop) {
    if(!model_capture_enabled())return;
    if(!model_qsa||b0!=0||b1!=48||!embed||!head||loop||!getenv("MODEL_IDS")||!getenv("MODEL_PLE")||getenv("MODEL_NPH")||getenv("MODEL_INJECT")) {
        fprintf(stderr,"MODEL_CAPTURE needs full blocks 0..48, embed+head, per-token QSA, complete MODEL_IDS, real MODEL_PLE, and no truncation/injection\n");exit(2);
    }
}
static FILE *model_capture_open(const char *leaf) {
    char path[1200];int n=snprintf(path,sizeof path,"%s/%s",getenv("MODEL_CAPTURE"),leaf);if(n<0||(size_t)n>=sizeof path){fprintf(stderr,"capture path too long\n");exit(2);}FILE *f=fopen(path,"wx");if(!f){perror(path);exit(1);}return f;
}
static void model_capture_record(unsigned t,unsigned T,const uint32_t *ids,const dn_bufs_t *buf,split_t sp[MAXD][MAXB]) {
    if(!model_capture_enabled())return;
    if(!t){uint32_t endian=1;if(*(const unsigned char *)&endian!=1){fprintf(stderr,"capture needs little-endian host\n");exit(2);}
        unsigned char *covered=calloc(248320,1);if(!covered){fprintf(stderr,"capture coverage allocation failed\n");exit(1);}
        for(int d=0;d<nd;d++){int start=sp[d][0].o0,count=sp[d][0].on;
            if(start<0||count<0||start>248320||count>248320-start){fprintf(stderr,"capture head slice out of bounds on die %d\n",d);exit(2);}
            for(int j=start;j<start+count;j++)if(covered[j]++){fprintf(stderr,"capture head slices overlap at %d\n",j);exit(2);}
        }
        for(int j=0;j<248320;j++)if(!covered[j]){fprintf(stderr,"capture head slices miss vocabulary ID %d\n",j);exit(2);}free(covered);
        if(mkdir(getenv("MODEL_CAPTURE"),0700)){perror(getenv("MODEL_CAPTURE"));exit(2);}
        FILE *f=model_capture_open("ids.u32");if(fwrite(ids,4,T,f)!=T||fclose(f)){fprintf(stderr,"capture ids write failed\n");exit(1);}
        f=model_capture_open("capture_info.json");
        if(fprintf(f,"{\n  \"schema\":\"flash-sass.raw-model-capture.v1\",\n  \"teacher_forced\":true,\n  \"positions\":%u,\n  \"vocab_size\":248320,\n  \"byte_order\":\"little\",\n  \"blocks\":[0,48],\n  \"attention_mode\":\"qsa\",\n  \"FLASH_NBLK\":%u,\n  \"FLASH_KV_HOST\":%d,\n  \"dies\":%d,\n  \"build_revision\":\"%s\",\n  \"logits_source\":\"actual GPU per-die head row slices\",\n  \"checkpoint_provenance\":null,\n  \"reference_provenance\":null,\n  \"quality_validation\":false\n}\n",T,model_budget,model_host,nd,FLASH_REV)<0||fclose(f)){fprintf(stderr,"capture metadata write failed\n");exit(1);}
        model_capture_logits=model_capture_open("logits.f32");
    }
    float *row=malloc(4*248320);if(!row){fprintf(stderr,"capture logit allocation failed\n");exit(1);}
    for(int d=0;d<nd;d++){CK(cuCtxSetCurrent(ctx[d]));CK(cuMemcpyDtoH(row+sp[d][0].o0,buf[d].hlo,4u*(unsigned)sp[d][0].on));}
    if(!model_capture_logits||fwrite(row,4,248320,model_capture_logits)!=248320||fflush(model_capture_logits)){fprintf(stderr,"capture logit write failed at t=%u\n",t);exit(1);}free(row);
    if(t+1==T){if(fclose(model_capture_logits)){fprintf(stderr,"capture logit close failed\n");exit(1);}model_capture_logits=NULL;printf("captured %u teacher-forced GPU logit rows; checkpoint/reference provenance still required\n",T);}
}
#endif
