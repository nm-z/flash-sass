/* Gather benchmark: finite distinguishable cache, exact preflight, raw events.
 * CLI: qsa_gather_bench nv tail iterations pattern host csv [seed]
 * Host is strict 0/1. Unique-address footprints are not hardware traffic.
 * CUDA events exclude selection upload and include the kernel framework.
 * QSA_PROFILE_FIRST=1 brackets the first timed launch for nvprof range capture. */
#define main qsa_component_main
#include "qsa_test.c"
#undef main
#include <cudaProfiler.h>
#ifndef FLASH_REV
#define FLASH_REV "unknown"
#endif
#define GB 512u
static uint32_t gh(uint32_t x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
static uint64_t gf(uint64_t h,const void *p,size_t n){const unsigned char *s=p;for(size_t i=0;i<n;i++){h^=s[i];h*=0x100000001b3ull;}return h;}
static int gu(const char *s,unsigned lo,unsigned hi,unsigned *out){if(!s||!*s)return 1;unsigned v=0;for(;*s;s++){if(*s<'0'||*s>'9')return 1;unsigned d=(unsigned)(*s-'0');if(v>hi/10||(v==hi/10&&d>hi%10))return 1;v=10*v+d;}if(v<lo)return 1;*out=v;return 0;}
static int gc_u(const void *a,const void *b){uint32_t x=*(const uint32_t *)a,y=*(const uint32_t *)b;return x<y?-1:x>y;}
static int gc_d(const void *a,const void *b){double x=*(const double *)a,y=*(const double *)b;return x<y?-1:x>y;}
static int gv(const uint32_t *s,unsigned k,unsigned nv){for(unsigned i=0;i<k;i++)if(s[i]>=nv||(i&&s[i-1]>=s[i]))return 1;return 0;}
static uint32_t *gtables(unsigned nv,unsigned k,unsigned it,const char *pattern,unsigned seed){
    uint32_t *all=malloc((size_t)it*k*4),*scratch=malloc((size_t)nv*4);if(!all||!scratch){fprintf(stderr,"selection allocation failed\n");exit(1);}
    if(!strcmp(pattern,"captured")||!strcmp(pattern,"captured-token")){
        const char *path=getenv("GATHER_IDS");FILE *f=path?fopen(path,"rb"):NULL;if(!f){fprintf(stderr,"captured needs GATHER_IDS\n");exit(2);}if(fseek(f,0,SEEK_END)){perror(path);exit(2);}long bytes=ftell(f);rewind(f);
        if(bytes!=(long)(k*4)&&bytes!=(long)((size_t)it*k*4)){fprintf(stderr,"GATHER_IDS needs one or iterations rows\n");exit(2);}if(fread(all,1,(size_t)bytes,f)!=(size_t)bytes){fprintf(stderr,"short GATHER_IDS\n");exit(2);}fclose(f);if(bytes==(long)(k*4))for(unsigned i=1;i<it;i++)memcpy(all+(size_t)i*k,all,k*4);
    }else for(unsigned i=0;i<it;i++){
        uint32_t *row=all+(size_t)i*k;
        if(k==nv){for(unsigned j=0;j<k;j++)row[j]=j;}
        else if(!strcmp(pattern,"clustered")){unsigned start=(nv-k)/2;for(unsigned j=0;j<k;j++)row[j]=start+j;}
        else if(!strcmp(pattern,"even")){for(unsigned j=0;j<k;j++)row[j]=(uint32_t)((uint64_t)j*nv/k);}
        else{
            unsigned seq=!strcmp(pattern,"changing")||!strcmp(pattern,"changing-token")?i:0;uint32_t rng=gh(seed+seq*977u);for(unsigned j=0;j<nv;j++)scratch[j]=j;
            for(unsigned j=0;j<k;j++){rng=rng*1664525u+1013904223u;unsigned pick=j+rng%(nv-j);uint32_t v=scratch[j];scratch[j]=scratch[pick];scratch[pick]=v;row[j]=scratch[j];}qsort(row,k,4,gc_u);
            if(seq&&i&&!memcmp(row,row-k,k*4)){unsigned missing=0;for(unsigned j=0;j<k&&row[j]==missing;j++)missing++;row[k-1]=missing;qsort(row,k,4,gc_u);}
        }
    }
    for(unsigned i=0;i<it;i++)if(gv(all+(size_t)i*k,k,nv)){fprintf(stderr,"invalid/nonunique selection row %u\n",i);exit(2);}free(scratch);return all;
}
static void glaunch(unsigned t,CUdeviceptr selected,const uint32_t *ids,unsigned k,CUevent begin,CUevent end,double *us){
    CK(cuMemcpyHtoD(selected,ids,k*4));CK(cuMemcpyHtoD(dCTL+8,&t,4));unsigned nph=1,ntok=t+1;void *args[]={&dPROG,&nph,&dCTL,&ntok};CK(cuEventRecord(begin,0));CK(cuLaunchKernel(fn,16,1,1,256,1,1,0,0,args,NULL));CK(cuEventRecord(end,0));CK(cuEventSynchronize(end));float ms;CK(cuEventElapsedTime(&ms,begin,end));*us=1000.*ms;
}
int main(int argc,char **argv){
    if(argc<7||argc>8){fprintf(stderr,"usage: qsa_gather_bench nv tail iters clustered|even|random|changing|changing-token|captured|captured-token|warm host0|1 csv [seed]\n");return 2;}
    unsigned nv,tail,it,host,seed=4242;if(gu(argv[1],1,65536,&nv)||gu(argv[2],0,3,&tail)||gu(argv[3],1,10000,&it)||gu(argv[5],0,1,&host)||(argc==8&&gu(argv[7],0,0xffffffffu,&seed))||4*nv+tail>262144){fprintf(stderr,"invalid benchmark dimensions/strict host boolean\n");return 2;}
    const char *pattern=argv[4];if(strcmp(pattern,"clustered")&&strcmp(pattern,"even")&&strcmp(pattern,"random")&&strcmp(pattern,"changing")&&strcmp(pattern,"changing-token")&&strcmp(pattern,"captured")&&strcmp(pattern,"captured-token")&&strcmp(pattern,"warm")){fprintf(stderr,"unknown pattern\n");return 2;}
    const char *profile_env=getenv("QSA_PROFILE_FIRST");int profile_first=0;
    if(profile_env){if(!strcmp(profile_env,"1"))profile_first=1;else if(strcmp(profile_env,"0")){fprintf(stderr,"QSA_PROFILE_FIRST must be exactly 0 or 1\n");return 2;}}
    unsigned k=nv<GB?nv:GB,visible=4*nv+tail,t=visible-1;
    int position_changes=!strcmp(pattern,"changing-token")||!strcmp(pattern,"captured-token");
    unsigned position_end=t+(position_changes?it-1:0);
    if(position_end>=262144u){fprintf(stderr,"per-token position sequence exceeds supported range\n");return 2;}
    unsigned logical_first=qsa_nkeys(t,GB),logical_last=qsa_nkeys(position_end,GB);
    unsigned logical_min=0xffffffffu,logical_max=0,tail_mask=0;
    for(unsigned i=0;i<it;i++){unsigned pos=t+(position_changes?i:0),n=qsa_nkeys(pos,GB);if(n<logical_min)logical_min=n;if(n>logical_max)logical_max=n;tail_mask|=1u<<(pos&3u);}
    unsigned physical_min=4*(k+(logical_min>4*k?1:0));
    unsigned slots=4*(k+(logical_max>4*k?1:0));
    unsigned npos=(position_end+128)/128*128,cpos=(4*(GB+1)+31)/32*32;
    if(!npos)npos=128;
    uint32_t *tables=gtables(nv,k,it,pattern,seed);uint64_t table_hash=gf(0xcbf29ce484222325ull,tables,(size_t)it*k*4);unsigned changes=0;for(unsigned i=1;i<it;i++)changes+=memcmp(tables+(size_t)i*k,tables+(size_t)(i-1)*k,k*4)!=0;
    if((!strcmp(pattern,"changing")||!strcmp(pattern,"changing-token"))&&nv>GB&&changes!=it-1){fprintf(stderr,"changing pattern did not change every selection\n");return 3;}
    FILE *csv=fopen(argv[6],"wx");if(!csv){perror(argv[6]);return 2;}
    int8_t *kq=malloc((size_t)npos*512),*vq=malloc((size_t)npos*512);uint16_t *ks=malloc((size_t)npos*32),*vs=malloc((size_t)npos*32);
    size_t kcf=ATTN_KC_FLOATS(cpos),vcf=(size_t)cpos*512;float *kc=calloc(kcf,4),*vc=calloc(vcf,4),*kg=malloc(4*kcf),*vg=malloc(4*(size_t)slots*512);double *times=malloc(it*sizeof *times),*sorted=malloc(it*sizeof *sorted);uint64_t *signatures=malloc(it*sizeof *signatures);
    if(!kq||!vq||!ks||!vs||!kc||!vc||!kg||!vg||!times||!sorted||!signatures){fprintf(stderr,"benchmark allocation failed\n");return 1;}
    for(unsigned p=0;p<npos;p++){for(unsigned d=0;d<512;d++){size_t j=(size_t)p*512+d;kq[j]=(int8_t)((int)(gh(seed+(p/4)*65537u+(p&3)*193u+d*17u)%255)-127);vq[j]=(int8_t)((int)(gh(seed+97u+(p/4)*131071u+(p&3)*389u+d*29u)%255)-127);}for(unsigned g=0;g<16;g++){size_t j=(size_t)p*16+g;ks[j]=qsa_f2h((1+(gh(p*31u+g+seed)&31u))/512.f);vs[j]=qsa_f2h((1+(gh(p*47u+g+seed+7)&31u))/1024.f);}}
    uint64_t cache_hash=0xcbf29ce484222325ull;cache_hash=gf(cache_hash,kq,(size_t)npos*512);cache_hash=gf(cache_hash,ks,(size_t)npos*32);cache_hash=gf(cache_hash,vq,(size_t)npos*512);cache_hash=gf(cache_hash,vs,(size_t)npos*32);
    dev_init();CUdeviceptr dk=host?halloc((size_t)npos*512):dalloc((size_t)npos*512),dv=host?halloc((size_t)npos*512):dalloc((size_t)npos*512),dks=host?halloc((size_t)npos*32):dalloc((size_t)npos*32),dvs=host?halloc((size_t)npos*32):dalloc((size_t)npos*32),dsel=dalloc(4*(GB+8)),dkc=dalloc(4*kcf),dvc=dalloc(4*vcf);
    CK(cuMemcpyHtoD(dk,kq,(size_t)npos*512));CK(cuMemcpyHtoD(dv,vq,(size_t)npos*512));CK(cuMemcpyHtoD(dks,ks,(size_t)npos*32));CK(cuMemcpyHtoD(dvs,vs,(size_t)npos*32));
    desc_t p;d_init(&p,OP_KVGATHER,GB+1);d_ptr(&p,0x140,dk);d_ptr(&p,0x148,dks);d_ptr(&p,0x150,dv);d_ptr(&p,0x158,dvs);d_ptr(&p,0x160,dkc);d_ptr(&p,0x168,dvc);d_ptr(&p,0x170,dsel);d_tok(&p,0x178,0);d_u32(&p,0x17c,GB);
    unsigned signature_changes=0;
    for(unsigned i=0;i<it;i++){
        uint32_t *ids=tables+(size_t)i*k;unsigned pos=t+(position_changes?i:0);unsigned expected=qsa_nkeys(pos,GB);
        CK(cuMemcpyHtoD(dsel,ids,k*4));unsigned n=qsa_gather_gpu(kq,ks,vq,vs,ids,pos,GB,kc,vc);
        if(n!=expected){fprintf(stderr,"logical key count mismatch at position %u: got %u expected %u\n",pos,n,expected);return 3;}
        unsigned active_slots=4*(k+(n>4*k?1:0));run_token(&p,1,pos);CK(cuMemcpyDtoH(kg,dkc,4*kcf));CK(cuMemcpyDtoH(vg,dvc,4*(size_t)active_slots*512));
        uint64_t sig=0xcbf29ce484222325ull;unsigned bad=0;
        for(unsigned j=0;j<active_slots;j++)for(int h=0;h<2;h++)for(int d=0;d<256;d++){size_t ki=attn_kidx(j,h,d),vi=(size_t)j*512+h*256+d;if(!isfinite(kg[ki])||!isfinite(vg[vi])||memcmp(kg+ki,kc+ki,4)||memcmp(vg+vi,vc+vi,4))bad++;sig=gf(sig,kg+ki,4);sig=gf(sig,vg+vi,4);}
        if(bad){fprintf(stderr,"preflight output mismatch row %u: %u values\n",i,bad);return 3;}signatures[i]=sig;if(i)signature_changes+=sig!=signatures[i-1];
    }
    if((!strcmp(pattern,"changing")||!strcmp(pattern,"changing-token"))&&nv>GB&&signature_changes!=it-1){fprintf(stderr,"changing selection lacks distinguishable output signatures\n");return 3;}
    CK(cuMemcpyHtoD(dPROG,&p,sizeof p));CUevent begin,end;CK(cuEventCreate(&begin,CU_EVENT_DEFAULT));CK(cuEventCreate(&end,CU_EVENT_DEFAULT));double dummy;
    for(unsigned i=0;i<10;i++)glaunch(t,dsel,tables,k,begin,end,&dummy);
    if(fprintf(csv,"iteration,position,tail_residue,logical_keys,physical_gathered_rows,source_unique_address_bytes,destination_unique_address_bytes,gpu_launch_us,selection_fnv1a64,verified_output_fnv1a64\n")<0){fprintf(stderr,"CSV write failed\n");return 1;}
    double sum=0,min=1e100,max=0;
    for(unsigned i=0;i<it;i++){
        uint32_t *ids=tables+(size_t)i*k;unsigned pos=t+(position_changes?i:0);unsigned keys=qsa_nkeys(pos,GB);
        if(profile_first&&i==0)CK(cuProfilerStart());
        glaunch(pos,dsel,ids,k,begin,end,times+i);
        if(profile_first&&i==0)CK(cuProfilerStop());
        sum+=times[i];if(times[i]<min)min=times[i];if(times[i]>max)max=times[i];uint64_t sh=gf(0xcbf29ce484222325ull,ids,k*4);
        unsigned rows=4*(k+(keys>4*k?1:0));
        if(fprintf(csv,"%u,%u,%u,%u,%u,%llu,%llu,%.9f,%016llx,%016llx\n",i,pos,pos&3u,keys,rows,(unsigned long long)rows*1088,(unsigned long long)rows*4096,times[i],(unsigned long long)sh,(unsigned long long)signatures[i])<0){fprintf(stderr,"CSV write failed\n");return 1;}
    }
    if(fclose(csv)){fprintf(stderr,"CSV close failed\n");return 1;}memcpy(sorted,times,it*sizeof *times);qsort(sorted,it,sizeof *sorted,gc_d);double mean=sum/it,ss=0;for(unsigned i=0;i<it;i++){double d=times[i]-mean;ss+=d*d;}
    unsigned q50=(it*50+99)/100-1,q90=(it*90+99)/100-1,q99=(it*99+99)/100-1;
    const char *position_policy=!strcmp(pattern,"changing-token")?"one token per launch with seeded changing IDs":!strcmp(pattern,"captured-token")?"one token per captured selection row":!strcmp(pattern,"changing")?"fixed position with changing IDs":!strcmp(pattern,"captured")?"fixed position with captured selection layout":!strcmp(pattern,"warm")?"fixed position with repeated-layout warm condition":"fixed position with repeated synthetic layout";
    printf("{\"source_revision\":\"%s\",\"pattern\":\"%s\",\"nv\":%u,\"tail\":%u,\"visible\":%u,\"position_start\":%u,\"position_end\":%u,\"tail_residue_start\":%u,\"tail_residue_end\":%u,\"tail_residues_mask\":%u,\"position_policy\":\"%s\",\"iterations\":%u,\"host_kv\":%u,\"realized_placement\":\"%s\",\"seed\":%u,\"block_budget\":512,\"logical_keys\":%u,\"logical_keys_first\":%u,\"logical_keys_last\":%u,\"logical_keys_min\":%u,\"logical_keys_max\":%u,\"physical_gathered_rows\":%u,\"physical_gathered_rows_min\":%u,\"physical_gathered_rows_max\":%u,\"physical_gathered_slots\":%u,\"physical_k_head_rows\":%u,\"physical_v_head_rows\":%u,\"source_unique_address_bytes\":%llu,\"source_unique_address_bytes_min\":%llu,\"source_unique_address_bytes_max\":%llu,\"destination_unique_address_bytes\":%llu,\"destination_unique_address_bytes_min\":%llu,\"destination_unique_address_bytes_max\":%llu,\"selection_changes\":%u,\"verified_output_signature_changes\":%u,\"cache_fnv1a64\":\"%016llx\",\"selections_fnv1a64\":\"%016llx\",\"preflight_verified_all\":true,\"selection_upload_timed\":false,\"warmup_launches\":10,\"profile_first_launch_requested\":%s,\"timing_scope\":\"CUDA event interval including persistent kernel framework\",\"quantile_method\":\"nearest_rank\",\"gpu_us_min\":%.9f,\"gpu_us_mean\":%.9f,\"gpu_us_stddev\":%.9f,\"gpu_us_p50\":%.9f,\"gpu_us_p90\":%.9f,\"gpu_us_p99\":%.9f,\"gpu_us_max\":%.9f,\"measured_hardware_transaction_bytes\":null,\"hardware_counter_gate\":\"separate Maxwell profiler required\"}\n",
           FLASH_REV,pattern,nv,tail,visible,t,position_end,t&3u,position_end&3u,tail_mask,position_policy,it,host,host?"pinned-system-memory":"device-vram",seed,
           logical_first,logical_first,logical_last,logical_min,logical_max,slots,physical_min,slots,slots,2*slots,2*slots,
           (unsigned long long)slots*1088,(unsigned long long)physical_min*1088,(unsigned long long)slots*1088,
           (unsigned long long)slots*4096,(unsigned long long)physical_min*4096,(unsigned long long)slots*4096,changes,signature_changes,
           (unsigned long long)cache_hash,(unsigned long long)table_hash,profile_first?"true":"false",
           min,mean,sqrt(ss/it),sorted[q50],sorted[q90],sorted[q99],max);
    CK(cuEventDestroy(begin));CK(cuEventDestroy(end));free(tables);free(kq);free(vq);free(ks);free(vs);free(kc);free(vc);free(kg);free(vg);free(times);free(sorted);free(signatures);return 0;
}
