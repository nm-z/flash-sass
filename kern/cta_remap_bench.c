/* Test-only engine library for issue #12. The normal engine implementation,
 * weights, programs and callbacks are shared by both arms. Only the die image
 * changes. This library is for a validator-owned flashcore, never a live server.
 *
 * stdin command:
 * ctabench OUT OLD NEW TRACE_OLD TRACE_NEW PROMPT_POSITIONS DECODE_INTERVALS PAIRS
 *
 * Prompt IDs are the fixed token 32. A snapshot immediately before the last
 * prompt position is built once. Each timed response restores that snapshot.
 * The first callback is excluded: host decode time is (last-first)/intervals.
 * Traced responses retain the last two positions, giving a complete last-token
 * device interval on each die independently; clocks from different dies are
 * never subtracted or added. Trace mode and mapping order alternate in balanced
 * pairs; all four cells use the same construction capacity and allocations.
 */
#define _GNU_SOURCE
#define engine_session cta_normal_engine_session
#include "engine.c"
#undef engine_session
#include <errno.h>
#include <inttypes.h>
#include <sys/stat.h>
#include "cta_trace.h"

typedef cta_trace_record cta_bench_record;

typedef struct {
    int first, count, expected, bad;
    uint32_t *ids;
    double *arrival;
} cta_callbacks;
static int cta_token(void *arg, int position, uint32_t id) {
    cta_callbacks *c=arg;
    if (position != c->first+c->count || c->count >= c->expected) {c->bad=1; return 1;}
    c->ids[c->count]=id;
    c->arrival[c->count++]=now();
    return 0;
}
static unsigned cta_nonempty(unsigned token, unsigned cta, int old) {
    unsigned n=token+1, nv=n/4;
    if (nv>nblk_qsa) n=4*nblk_qsa+n%4;
    unsigned ck=((n+8191)/8192)*1024, count=0;
    for(unsigned unit=cta;unit<32;unit+=16) {
        unsigned chunk=old ? unit&7 : unit>>2;
        count+=chunk*ck<n;
    }
    return count;
}
static int cta_load(const char *path) {
    CUmodule next[MAXD]={0}; CUfunction function[MAXD];
    for(int d=0;d<nd;d++) {
        CK(cuCtxSetCurrent(ctx[d]));
        CUresult rc=cuModuleLoad(&next[d],path);
        if (!rc) rc=cuModuleGetFunction(&function[d],next[d],"die");
        if(rc) {
            for(int e=0;e<=d;e++) if(next[e]) {cuCtxSetCurrent(ctx[e]); cuModuleUnload(next[e]);}
            return -1;
        }
    }
    for(int d=0;d<nd;d++) {
        CK(cuCtxSetCurrent(ctx[d])); CK(cuModuleUnload(mod[d]));
        mod[d]=next[d]; fn[d]=function[d];
    }
    snprintf(cubin_path,sizeof cubin_path,"%s",path); cubin_fnv=file_fnv(path);
    return 0;
}
static int cta_category(unsigned opcode) {
    if(opcode==OP_ATTNPARTN || opcode==OP_ATTNMERGE) return 0;
    if(opcode==OP_KVGATHER) return 1;
    if(opcode==OP_IDXQK || opcode==OP_IDXSCORE || opcode==OP_IDXSEL) return 2;
    return 3;
}
static int cta_trace_dump(const char *path, CUdeviceptr *trace, unsigned first,
                          unsigned nonce, int old, uint64_t *previous,
                          double device_us[MAXD], double categories[MAXD][8]) {
    FILE *f=fopen(path,"wx"); if(!f) return -1;
    fputs("die,token,phase,cta,opcode,op,wait_start_ns,wait_end_ns,body_start_ns,body_end_ns,phase_end_ns,nonempty_units,nonce\n",f);
    int bad=0;
    for(int d=0;d<nd && !bad;d++) {
        size_t phases=(size_t)prof_nph[d], count=2*phases*16;
        cta_bench_record *records=malloc(count*sizeof *records);
        if(!records){bad=1;break;}
        CK(cuCtxSetCurrent(ctx[d])); CK(cuMemcpyDtoH(records,trace[d],count*sizeof *records));
        uint64_t maximum=previous[d];
        for(unsigned t=0;t<2;t++) for(unsigned p=0;p<phases;p++) for(unsigned c=0;c<16;c++) {
            const cta_bench_record *r=&records[((size_t)t*phases+p)*16+c];
            unsigned op=prof_op[d][p];
            unsigned expected=op==OP_ATTNPARTN ? cta_nonempty(first+t,c,old) : 0;
            if(!cta_trace_valid(r,first+t,p,c,op,nonce,previous[d],expected)) bad=1;
            if(p && records[((size_t)t*phases+p-1)*16+c].end>r->wait_start) bad=1;
            if(t && !p && records[(phases-1)*16+c].end>r->wait_start) bad=1;
            if(r->end>maximum) maximum=r->end;
            fprintf(f,"%d,%u,%u,%u,%u,%s,%"PRIu64",%"PRIu64",%"PRIu64",%"PRIu64",%"PRIu64",%u,%u\n",
                    d,r->token,r->phase,r->cta,r->opcode,die_opname[op],
                    r->wait_start,r->wait_end,r->body_start,r->body_end,r->end,r->units,r->nonce);
        }
        if(!bad) {
            uint64_t begin=records[(phases-1)*16].end;
            uint64_t end=records[(2*phases-1)*16].end, pieces=0;
            if(end<=begin) bad=1;
            else {
                device_us[d]=(end-begin)*.001;
                for(unsigned p=0;p<phases;p++) {
                    const cta_bench_record *r=&records[(phases+p)*16];
                    categories[d][cta_category(r->opcode)]+=(r->body_end-r->body_start)*.001;
                    categories[d][4]+=(r->wait_end-r->wait_start)*.001;
                    categories[d][5]+=(r->body_start-r->wait_end)*.001;
                    categories[d][6]+=(r->end-r->body_end)*.001;
                    pieces+=r->end-r->wait_start;
                }
                if(pieces>end-begin) bad=1;
                else categories[d][7]=(end-begin-pieces)*.001;
            }
        }
        previous[d]=maximum; free(records);
    }
    if(ferror(f)) bad=1;
    if(fclose(f)) bad=1;
    if(bad) fprintf(stderr,"CTA trace failed metadata, work, freshness, or reconciliation: %s\n",path);
    return bad ? -1 : 0;
}
static int cta_measure(FILE *out, const char *dir, const char *image[4],
                       unsigned prompt, unsigned decode, unsigned pairs) {
    if(nd!=6 || nlayers_run!=48 || nblk_qsa!=512 ||
       (core->ple_compact && !ple_disk) || prompt<2 || decode<2 || decode>1024 ||
       pairs<2 || pairs>100 || (pairs&1) || prompt>(unsigned)MAXT ||
       decode>(unsigned)MAXT-prompt) {
        fputs("error: require six dies, all 48 layers, real PLE, NBLK=512, prompt>=2, decode 2..1024, even pairs 2..100, and sufficient capacity\n",out);
        return -1;
    }
    for(int i=0;i<4;i++) if(!file_fnv(image[i])) {fprintf(out,"error: cannot read image %d\n",i); return -1;}
    if(mkdir(dir,0775)) {fprintf(out,"error: cannot create new output directory: %s\n",strerror(errno));return -1;}
    uint32_t *prompt_ids=malloc(prompt*sizeof *prompt_ids), *canonical=malloc((decode+1)*4);
    cta_callbacks cb={.first=(int)prompt-1,.expected=(int)decode+1};
    cb.ids=malloc((decode+1)*4); cb.arrival=malloc((decode+1)*sizeof *cb.arrival);
    if(!prompt_ids || !canonical || !cb.ids || !cb.arrival) return -1;
    for(unsigned i=0;i<prompt;i++) prompt_ids[i]=32;
    uint32_t stop[2]={UINT32_MAX,UINT32_MAX}; sampling greedy={0};
    engine_cb callbacks={cta_token,NULL,&cb}; run_stats stats;
    char path[2048]; int bad=0; CUdeviceptr trace[MAXD]={0};
    uint64_t previous[MAXD]={0}; unsigned nonce=0;
    snap_s=0;
    if(cta_load(image[0])) {bad=1;goto done;}
    int rc=engine_run(prompt_ids,(int)prompt,(int)decode,stop,&greedy,(int)prompt-1,NULL,0,&callbacks,&stats);
    if(rc || cb.bad || cb.count!=cb.expected || stats.ntok!=(int)(prompt+decode) || stats.snapshot!=(int)prompt-1 || ple_short_reads) {
        fprintf(out,"error: prefill/snapshot/parity baseline failed rc=%d\n",rc);bad=1;goto done;
    }
    memcpy(canonical,cb.ids,(decode+1)*4);
    snprintf(path,sizeof path,"%s/config.txt",dir);
    FILE *config=fopen(path,"wx");
    if(!config){bad=1;goto done;}
    fprintf(config,"revision=%s\ndies=%d\nlayers=%d\nnblk=%u\nkv_host=%d\nple_disk=%d\nple_offset=%lld\ncore_capacity=%d\nengine_capacity=%d\nimage_capacity=%u\nprompt=%u\nprompt_repeated_token=32\ndecode_intervals=%u\npairs=%u\n",
            FLASH_REV,nd,nlayers_run,nblk_qsa,kv_host,ple_disk,(long long)ple_off,core->maxt,MAXT,img_rows_max,prompt,decode,pairs);
    fprintf(config,"split=%s\nattn_map=%s\ntwo_position=%d\nprefill_chunk=%d\n",getenv("SPLIT_SHARES")?getenv("SPLIT_SHARES"):"default",getenv("ATTN_MAP")?getenv("ATTN_MAP"):"default",two_position,prefill_chunk);
    for(int i=0;i<4;i++) fprintf(config,"image%d=%s\nimage%d_fnv=%016llx\n",i,image[i],i,(unsigned long long)file_fnv(image[i]));
    for(int d=0;d<nd;d++) fprintf(config,"die%d_phases=%d\n",d,prof_nph[d]);
    if(fclose(config)){bad=1;goto done;}
    /* Equal allocations in all four cells; no allocation is inside a timed
     * response. Alternate trace-stage order as well as old/new mapping order. */
    for(int d=0;d<nd;d++) {
        CK(cuCtxSetCurrent(ctx[d]));
        CK(cuMemAlloc(&trace[d],(size_t)2*prof_nph[d]*16*64));
    }
    for(int pair=-2;pair<(int)pairs && !bad;pair++) for(int stage_turn=0;stage_turn<2 && !bad;stage_turn++) {
        int stage=(pair&1)?stage_turn:1-stage_turn;
        for(int turn=0;turn<2 && !bad;turn++) {
            int arm=(pair&1)?turn:1-turn;
            if(cta_load(image[stage*2+arm])) {bad=1;break;}
            ++nonce;
            for(int d=0;d<nd;d++) {
                CK(cuCtxSetCurrent(ctx[d]));
                uint32_t meta[6]={0,0,prompt+decode-2,0,16,nonce};
                if(stage) {
                    CK(cuMemsetD8(trace[d],0,(size_t)2*prof_nph[d]*16*64));
                    meta[0]=(uint32_t)trace[d];meta[1]=(uint32_t)(trace[d]>>32);meta[3]=2;
                }
                CK(cuMemcpyHtoD(dCTL[d]+0xd0,meta,sizeof meta));
            }
            cb.count=cb.bad=0;
            rc=engine_run(prompt_ids,(int)prompt,(int)decode,stop,&greedy,0,NULL,0,&callbacks,&stats);
            if(rc || cb.bad || cb.count!=cb.expected || stats.ntok!=(int)(prompt+decode) ||
               stats.resumed!=(int)prompt-1 || stats.ended || ple_short_reads ||
               memcmp(canonical,cb.ids,(decode+1)*4)) {
                fprintf(out,"error: matched decode failed stage=%d pair=%d arm=%d rc=%d\n",stage,pair,arm,rc);bad=1;break;
            }
            for(unsigned i=1;i<=decode;i++) if(cb.arrival[i]<cb.arrival[i-1]) bad=1;
            double host_us=(cb.arrival[decode]-cb.arrival[0])*1e6/decode;
            if(!isfinite(host_us) || host_us<=0) {bad=1;break;}
            double device_us[MAXD]={0}, cats[MAXD][8]={{0}};
            const char *stagename=stage?"traced":"untraced", *armname=arm?"new":"old";
            if(stage) {
                snprintf(path,sizeof path,"%s/%s-%s-pair%d.trace.csv",dir,stagename,armname,pair);
                if(cta_trace_dump(path,trace,prompt+decode-2,nonce,!arm,previous,device_us,cats)){bad=1;break;}
            }
            snprintf(path,sizeof path,"%s/%s-%s-pair%d.json",dir,stagename,armname,pair);
            FILE *f=fopen(path,"wx"); if(!f){bad=1;break;}
            fprintf(f,"{\"stage\":\"%s\",\"arm\":\"%s\",\"pair\":%d,\"warmup\":%s,\"order\":%d,\"stage_order\":%d,\"token_parity\":true,\"resumed\":%d,\"decode_intervals\":%u,\"host_decode_us\":%.6f,\"run_ms\":%.6f,\"build_ms\":%.6f,\"token_ids\":[",
                    stagename,armname,pair,pair<0?"true":"false",turn,stage_turn,stats.resumed,decode,host_us,stats.run_ms,stats.build_ms);
            for(unsigned i=0;i<=decode;i++) fprintf(f,"%s%u",i?",":"",cb.ids[i]);
            fputs("],\"arrival_since_first_us\":[",f);
            for(unsigned i=0;i<=decode;i++) fprintf(f,"%s%.6f",i?",":"",(cb.arrival[i]-cb.arrival[0])*1e6);
            fputs("],\"dies\":[",f);
            if(stage) for(int d=0;d<nd;d++) {
                fprintf(f,"%s{\"die\":%d,\"device_decode_us\":%.6f,\"attention_body_us\":%.6f,\"gather_body_us\":%.6f,\"indexer_body_us\":%.6f,\"other_body_us\":%.6f,\"exchange_wait_us\":%.6f,\"dispatch_us\":%.6f,\"grid_wait_us\":%.6f,\"transition_us\":%.6f}",
                        d?",":"",d,device_us[d],cats[d][0],cats[d][1],cats[d][2],cats[d][3],cats[d][4],cats[d][5],cats[d][6],cats[d][7]);
            }
            fputs("]}\n",f); if(fclose(f)){bad=1;break;}
            fprintf(out,"sample %s %s pair=%d host_decode_us=%.6f parity=PASS\n",stagename,armname,pair,host_us);fflush(out);
        }
    }
done:
    for(int d=0;d<nd;d++) if(trace[d]) {
        CK(cuCtxSetCurrent(ctx[d])); uint32_t meta[6]={0};
        CK(cuMemcpyHtoD(dCTL[d]+0xd0,meta,sizeof meta)); CK(cuMemFree(trace[d]));
    }
    free(prompt_ids);free(canonical);free(cb.ids);free(cb.arrival);
    snprintf(path,sizeof path,"%s/status.txt",dir);
    FILE *status=fopen(path,"w");if(status){fputs(bad?"FAIL\n":"PASS\n",status);if(fclose(status))bad=1;}else bad=1;
    return bad?-1:0;
}
int engine_session(FILE *in, FILE *out) {
    char *line=NULL;size_t capacity=0;int result=0;
    while(getline(&line,&capacity,in)>0) {
        if(!strncmp(line,"ctabench ",9)) {
            char dir[512], images[4][512]; unsigned prompt,decode,pairs;char extra;
            int n=sscanf(line+9,"%511s %511s %511s %511s %511s %u %u %u %c",
                         dir,images[0],images[1],images[2],images[3],&prompt,&decode,&pairs,&extra);
            const char *paths[4]={images[0],images[1],images[2],images[3]};
            int rc=n==8?cta_measure(out,dir,paths,prompt,decode,pairs):-1;
            fprintf(out,"ctabench result=%d\n",rc);fflush(out);
            if(rc){result=1;break;}
        } else {
            /* Keep the normal reload/config/quit protocol in the validation
             * library. This one-command stream does not carry chat settings. */
            FILE *one=fmemopen(line,strlen(line),"r");
            if(!one){result=1;break;}
            int rc=cta_normal_engine_session(one,out);fclose(one);
            if(rc){result=rc;break;}
        }
    }
    free(line);return result;
}
