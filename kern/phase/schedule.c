/* Offline production schedule compiler. No CUDA library is linked: symbolic
 * addresses preserve alias relationships while resident metadata supplies types.
 * This emits descriptors and counts; it does not execute model arithmetic. */
#define FLASH_EXACT_BATCH 1
#include "../engine.c"
static CUdeviceptr next_symbol=0x10000000000ull;
static CUdeviceptr symbol(void) { next_symbol+=0x1000000;return next_symbol; }
static void metadata(hold_arena *a,const char *path) {
    FILE *f=fopen(path,"r");if(!f){perror(path);exit(2);}
    char *line=NULL;size_t cap=0;
    if(getline(&line,&cap,f)<0 || sscanf(line,"arena %zu die %d",&a->bytes,&a->die)!=2)exit(2);
    a->base=symbol();
    hold_tensor t;
    while(getline(&line,&cap,f)>0) {
        if(sscanf(line,"%95s %zu %zu %u %d %d %d %d",t.name,&t.off,&t.bytes,&t.type,&t.rows,&t.K,&t.stride,&t.experts)!=8)exit(2);
        a->t=realloc(a->t,(a->n+1)*sizeof *a->t);a->t[a->n++]=t;
    }
    free(line);fclose(f);
}
static unsigned dtype(int b,const char *name) {
    const hold_tensor *t=hold_find(&ar[0],tn(b,name));return t?t->type:0;
}
#include "schedule_symbols.inc"
int main(int argc,char **argv) {
    if(argc!=2){fprintf(stderr,"usage: phase_schedule HOLDER_METADATA_DIRECTORY\n");return 2;}
    core_t local={0};core=&local;nd=6;nlayers_run=48;nblk_qsa=512;MAXT=32768;embed_die=0;
    const int baseline=getenv("FLASH_SCHEDULE_BASELINE") && atoi(getenv("FLASH_SCHEDULE_BASELINE"));
    producer_quant=group_matrices=!baseline;two_position=0;embedding_tables[0]=symbol();
    setenv("FLASH_PRODUCER_QUANT",baseline?"0":"1",1);setenv("FLASH_FUSED_GR",baseline?"0":"1",1);
    setenv("FLASH_ATTN_BALANCE",baseline?"0":"1",1);
    setenv("SPLIT_SHARES","112,97,81,116,97,97",1);
    setenv("ATTN_MAP","1,1,1,1,4,4,4,4,5,5,5,5",1);
    dw=calloc(MAXD,sizeof *dw);sp=calloc(MAXD,sizeof *sp);
#define STATES(x) x=calloc(MAXD,sizeof *x)
    STATES(dst_dn);STATES(dst_conv);STATES(dst_kq);STATES(dst_ks);
    STATES(dst_vq);STATES(dst_vs);STATES(dst_pkt);STATES(dst_ring);
#undef STATES
    for(int d=0;d<nd;d++) {
        char path[1024];snprintf(path,sizeof path,"%s/hold-h%d.manifest",argv[1],d);metadata(&ar[d],path);
        symbols(d);
        xc[d]=(xchg_t){.inbox=symbol(),.flags=symbol(),.xtab=symbol(),.xgen=symbol()};
        producer_tables[0][d][0]=symbol();producer_tables[0][d][1]=symbol();
        for(int b=0;b<48;b++) {
            split_of(&sp[d][b],d,nd,b);weight_symbols(d,b);
            dst_dn[d][b]=symbol();dst_conv[d][b]=symbol();
            if(BLK_IS_ATTN(b)) {
                int owner=runtime_attn_owner(b),rank=d-(d>owner);
                dw[d][b].attn_q_split=1;
                if(d!=owner) {
                    dw[d][b].attn_q_first=(384u*(unsigned)rank/5)*32;
                    dw[d][b].attn_q_rows=(384u*(unsigned)(rank+1)/5)*32-dw[d][b].attn_q_first;
                }
                if(d==owner) {
                    dst_kq[d][b]=symbol();dst_ks[d][b]=symbol();dst_vq[d][b]=symbol();
                    dst_vs[d][b]=symbol();dst_pkt[d][b]=symbol();dst_ring[d][b]=symbol();
                }
            }
        }
        uint32_t stop[2]={~0u,~0u};prog_t p;
        model_program(&p,d,0,64,stop,0,1,1,1,0);
        unsigned count[128]={0},phases=0,inner=0,grid=0,send=0,wait=0;
        for(int i=0;i<p.n;) {
            unsigned width=phase_descriptors(p.p+i);phase_t *a=p.p+i;
            if(a->w[0]>=128 || !width || width>(unsigned)(p.n-i))return 2;
            count[a->w[0]]++;phases++;
            grid+=!(a->w[30]&4u);send+=!!(a->w[30]&1u);wait+=!!(a->w[30]&2u);
            if(a->w[30]&PHASE_CHAIN_FLAG && a->w[30]&PHASE_GROUP_FLAG)
                for(unsigned k=0;k+1<width;k++)inner+=!!(a[k].w[30]&PHASE_CHAIN_BAR_FLAG);
            if(getenv("FLASH_SCHEDULE_DESCRIPTORS")) {
                printf("descriptor die=%d phase=%u index=%d opcode=%u units=%u width=%u flags=%u layer=%d\n",d,phases-1,i,a->w[0],a->w[1],width,a->w[30],prof_layer[d][i]);
            }
            if(getenv("FLASH_SCHEDULE_WAIT_AUDIT")) {
                printf("wait_phase die=%d phase=%u index=%d name=%s units=%u width=%u flags=%u layer=%d grid=%u send=%u exchange_wait=%u\n",
                       d,phases-1,i,profile_op_name(a->w[0]),a->w[1],width,a->w[30],prof_layer[d][i],
                       !(a->w[30]&4u),!!(a->w[30]&1u),!!(a->w[30]&2u));
                for(unsigned j=0;j<width;j++)
                    printf("wait_body die=%d phase=%u member=%u name=%s units=%u flags=%u\n",
                           d,phases-1,j,profile_op_name(a[j].w[0]),a[j].w[1],a[j].w[30]);
            }
            i+=(int)width;
        }
        printf("schedule die=%d layers=48 head=1 mtp=0 fp8=0 descriptors=%d phases=%u inner_grid_barriers=%u\n",d,p.n,phases,inner);
        printf("wait_summary die=%d baseline=%d grid_boundaries=%u exchange_publications=%u exchange_waits=%u publication_system_fence_threads=%u\n",d,baseline,grid,send,wait,16u*send);
        for(unsigned o=0;o<128;o++)if(count[o])printf("op die=%d opcode=%u name=%s phases=%u\n",d,o,profile_op_name(o),count[o]);
        prog_free(&p);
    }
    return 0;
}
