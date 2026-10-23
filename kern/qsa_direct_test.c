/* Direct-position QSA: synthetic history, current original-input projections.
 * This is component parity evidence, not a checkpoint-quality evaluation. */
#define main qsa_component_main
#include "qsa_test.c"
#undef main
#ifndef FLASH_REV
#define FLASH_REV "unknown"
#endif
typedef struct {float qg[12288],k[512],v[512],iqk[640],qn[256],kn[256],iqn[128],ikn[128];} direct_frame;
static uint32_t direct_hash(uint32_t x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
static float direct_raw(unsigned t,unsigned d,unsigned seed){return ((int)(direct_hash(t*1664525u+d*1013904223u+seed)&65535u)-32768)/32768.f;}
static uint64_t direct_file_hash(const char *path){if(!path)return 0;FILE*f=fopen(path,"rb");if(!f)return 0;uint64_t h=0xcbf29ce484222325ull;unsigned char buf[65536];size_t n;while((n=fread(buf,1,sizeof buf,f)))for(size_t i=0;i<n;i++){h^=buf[i];h*=0x100000001b3ull;}fclose(f);return h;}
static int direct_compare(const char *name,const void *want,CUdeviceptr dev,size_t n,int *shown){void*got=malloc(n?n:1);if(!got)exit(1);if(n)CK(cuMemcpyDtoH(got,dev,n));int bad=cmpb(name,want,got,n,shown);free(got);return bad;}
static void direct_flip(float*x){uint32_t u;memcpy(&u,x,4);u^=1;memcpy(x,&u,4);}
static unsigned direct_param(const desc_t*d,int addr){return d->w[2+(addr-0x140)/4];}
static void direct_assert(const desc_t*p,unsigned nph,unsigned budget){
 unsigned cnt[DIE_NOPS]={0};
 for(unsigned i=0;i<nph;i++){unsigned op=p[i].w[0];if(op<DIE_NOPS)cnt[op]++;int addr=op==OP_IDXSCORE?0x164:op==OP_IDXSEL?0x15c:op==OP_KVGATHER?0x17c:op==OP_ATTNPARTN?0x174:0;if((addr&&direct_param(p+i,addr)!=budget)||(op==OP_KVGATHER&&p[i].w[1]!=budget+1)){fprintf(stderr,"descriptor budget assertion failed\n");exit(3);}}
 if(cnt[OP_ATTNQKQ]!=1||cnt[OP_IDXQK]!=1||cnt[OP_IDXSCORE]!=1||cnt[OP_IDXSEL]!=5||cnt[OP_KVGATHER]!=1||cnt[OP_ATTNPARTN]!=1||cnt[OP_ATTNMERGE]!=1||cnt[OP_ATTNQK]||cnt[OP_ATTNPART]){fprintf(stderr,"QSA chain assertion failed\n");exit(3);}
}
#include "qsa_direct_projection.h"
#include "qsa_direct_fixtures.h"
static int direct_host;
static int test_direct(unsigned visible,unsigned budget,unsigned steps,unsigned seed){
 const float eps=1e-6f;unsigned t0=visible-1,last=visible+steps-1,npos=(last+127u)&~127u,ngrp=npos/128,cpos=(4u*(budget+1u)+2048u+31u)/32u*32u;
 size_t kcf=ATTN_KC_FLOATS(cpos),vcf=(size_t)cpos*512;direct_frame frame,initial;memset(&frame,0,sizeof frame);rs=seed;
 const char*projected=getenv("QSA_PROJECTED");FILE*frames=NULL;
 if(projected){frames=fopen(projected,"rb");if(!frames){perror(projected);return 2;}if(fseek(frames,0,SEEK_END)||ftell(frames)!=(long)(steps*sizeof frame)||fseek(frames,0,SEEK_SET)||fread(&frame,sizeof frame,1,frames)!=1){fprintf(stderr,"QSA_PROJECTED needs %u frames of %zu bytes\n",steps,sizeof frame);fclose(frames);return 2;}}
 else{for(int i=0;i<256;i++){frame.qn[i]=.75f+.125f*frand();frame.kn[i]=.75f+.125f*frand();}for(int i=0;i<128;i++){frame.iqn[i]=.75f+.125f*frand();frame.ikn[i]=.75f+.125f*frand();}}
 direct_projection projection;direct_projection_init(&projection,&frame,steps,seed,projected!=NULL);direct_projection_cpu(&projection,&frame,t0,seed);initial=frame;
 for(size_t j=0;j<sizeof frame/sizeof(float);j++)if(!isfinite(((float*)&frame)[j])){fprintf(stderr,"nonfinite projected frame\n");return 2;}
 int8_t*kq=calloc((size_t)npos*512,1),*vq=calloc((size_t)npos*512,1);uint16_t*ks=calloc((size_t)npos*16,2),*vs=calloc((size_t)npos*16,2),*pool=calloc((size_t)npos/4*128,2),*pkt=calloc((size_t)ngrp*4096,2);
 float*score=calloc((size_t)npos/4+64,4),*kc=calloc(kcf,4),*vc=calloc(vcf,4);uint32_t*sel=calloc(budget+8,4);float ring[512],iq[512],qo[6144],out[6144],pm[192],pd[192],po[49152];
 if(!kq||!vq||!ks||!vs||!pool||!pkt||!score||!kc||!vc||!sel){fprintf(stderr,"prefix allocation failed\n");return 1;}
 direct_seed_kv(t0,npos,seed,&frame,kq,ks,vq,vs);
 for(unsigned b=0;b<t0/4;b++){float raw[512],cs[32],sn[32];for(unsigned j=0;j<4;j++)for(unsigned d=0;d<128;d++)raw[j*128+d]=direct_raw(4*b+j,d,seed);rope_table_gpu(4*b,1e7f,cs,sn);idx_pool_gpu(raw,frame.ikn,eps,cs,sn,pool+(size_t)b*128);for(int d=0;d<128;d++)pkt[pkt_idx(b,d)]=pool[(size_t)b*128+d];}
 direct_seed_ring(t0,seed,ring);direct_history_pattern(getenv("QSA_PATTERN"),t0,budget,&frame,pool,pkt);int host=direct_host;
 CUdeviceptr dKQ=host?halloc((size_t)npos*512):dalloc((size_t)npos*512),dVQ=host?halloc((size_t)npos*512):dalloc((size_t)npos*512),dKS=host?halloc((size_t)npos*32):dalloc((size_t)npos*32),dVS=host?halloc((size_t)npos*32):dalloc((size_t)npos*32);
 CUdeviceptr dPKT=dalloc((size_t)ngrp*8192),dRING=dalloc(2048),dIQK=dalloc(2560),dIQ=dalloc(2048),dSC=dalloc(4*((size_t)npos/4+64)),dHST=dalloc(5120),dSEL=dalloc(4*(budget+8));
 CUdeviceptr dQG=dalloc(4*12288),dK=dalloc(2048),dV=dalloc(2048),dQN=dalloc(1024),dKN=dalloc(1024),dIQN=dalloc(512),dIKN=dalloc(512);
 CUdeviceptr dCS=dalloc(4*32*(size_t)npos),dSN=dalloc(4*32*(size_t)npos),dQO=dalloc(4*6144),dKC=dalloc(4*kcf),dVC=dalloc(4*vcf),dPM=dalloc(sizeof pm),dPD=dalloc(sizeof pd),dPO=dalloc(sizeof po),dOUT=dalloc(sizeof out);
 CK(cuMemcpyHtoD(dKQ,kq,(size_t)npos*512));CK(cuMemcpyHtoD(dVQ,vq,(size_t)npos*512));CK(cuMemcpyHtoD(dKS,ks,(size_t)npos*32));CK(cuMemcpyHtoD(dVS,vs,(size_t)npos*32));CK(cuMemcpyHtoD(dPKT,pkt,(size_t)ngrp*8192));CK(cuMemcpyHtoD(dRING,ring,sizeof ring));
 CK(cuMemcpyHtoD(dQN,frame.qn,sizeof frame.qn));CK(cuMemcpyHtoD(dKN,frame.kn,sizeof frame.kn));CK(cuMemcpyHtoD(dIQN,frame.iqn,sizeof frame.iqn));CK(cuMemcpyHtoD(dIKN,frame.ikn,sizeof frame.ikn));
 for(unsigned t=t0&~3u;t<last;t++){float cs[32],sn[32];rope_table_gpu(t,1e7f,cs,sn);CK(cuMemcpyHtoD(dCS+128*(size_t)t,cs,128));CK(cuMemcpyHtoD(dSN+128*(size_t)t,sn,128));}
 desc_t allprog[64];unsigned projection_n=direct_projection_program(&projection,allprog,dQG,dK,dV,dIQK);desc_t*p=allprog+projection_n;
 d_init(p,OP_ATTNQKQ,26);d_ptr(p,0x140,dQG);d_ptr(p,0x148,dK);d_ptr(p,0x150,dV);d_ptr(p,0x158,dQN);d_ptr(p,0x160,dKN);d_ptr(p,0x168,dCS);d_ptr(p,0x170,dSN);d_ptr(p,0x178,dQO);d_ptr(p,0x180,dKQ);d_ptr(p,0x188,dVQ);d_tok(p,0x190,0);d_f32(p,0x194,eps);d_ptr(p,0x198,dKS);d_ptr(p,0x1a0,dVS);
 d_init(p+1,OP_IDXQK,6);d_ptr(p+1,0x140,dIQK);d_ptr(p+1,0x148,dIQN);d_ptr(p+1,0x150,dIKN);d_ptr(p+1,0x158,dCS);d_ptr(p+1,0x160,dSN);d_ptr(p+1,0x168,dIQ);d_ptr(p+1,0x170,dRING);d_ptr(p+1,0x178,dPKT);d_ptr(p+1,0x180,dHST);d_tok(p+1,0x188,0);d_f32(p+1,0x18c,eps);
 d_init(p+2,OP_IDXSCORE,16);d_ptr(p+2,0x140,dPKT);d_ptr(p+2,0x148,dIQ);d_ptr(p+2,0x150,dSC);d_ptr(p+2,0x158,dHST);d_tok(p+2,0x160,0);d_u32(p+2,0x164,budget);
 for(unsigned j=0;j<5;j++){desc_t*d=p+3+j;d_init(d,OP_IDXSEL,16);d_ptr(d,0x140,dSC);d_ptr(d,0x148,dHST);d_ptr(d,0x150,dSEL);d_tok(d,0x158,0);d_u32(d,0x15c,budget);d_u32(d,0x160,j+2);}
 d_init(p+8,OP_KVGATHER,budget+1);d_ptr(p+8,0x140,dKQ);d_ptr(p+8,0x148,dKS);d_ptr(p+8,0x150,dVQ);d_ptr(p+8,0x158,dVS);d_ptr(p+8,0x160,dKC);d_ptr(p+8,0x168,dVC);d_ptr(p+8,0x170,dSEL);d_tok(p+8,0x178,0);d_u32(p+8,0x17c,budget);
 d_init(p+9,OP_ATTNPARTN,32);d_ptr(p+9,0x140,dQO);d_ptr(p+9,0x148,dKC);d_ptr(p+9,0x150,dVC);d_ptr(p+9,0x158,dPM);d_ptr(p+9,0x160,dPD);d_ptr(p+9,0x168,dPO);d_tok(p+9,0x170,1);d_u32(p+9,0x174,budget);
 d_init(p+10,OP_ATTNMERGE,24);d_ptr(p+10,0x140,dPM);d_ptr(p+10,0x148,dPD);d_ptr(p+10,0x150,dPO);d_ptr(p+10,0x158,dQG);d_ptr(p+10,0x160,dOUT);
 direct_assert(allprog,projection_n+11,budget);if(projection.enabled&&projection_n<2){fprintf(stderr,"projection stages missing\n");return 3;}
 int driver=0;char gpu[128]={0};CK(cuDriverGetVersion(&driver));CK(cuDeviceGetName(gpu,sizeof gpu,0));const char*cubin=getenv("DIE_CUBIN")?getenv("DIE_CUBIN"):"die.cubin";
 printf("direct QSA: SYNTHETIC HISTORY source=%s compiler=%s driver=%d gpu=%s CUDA_VISIBLE_DEVICES=%s cubin=%s fnv1a64=%016llx\n",FLASH_REV,__VERSION__,driver,gpu,getenv("CUDA_VISIBLE_DEVICES")?getenv("CUDA_VISIBLE_DEVICES"):"unset",cubin,(unsigned long long)direct_file_hash(cubin));
 printf("visible=%u steps=%u budget=%u hostKV=%d seed=%u prefix_seeded_once=yes future_sentinels=yes current_mode=%s projected_fnv=%016llx\n",visible,steps,budget,host,seed,projected?"POSTPROJECTION_DIAGNOSTIC":"ORIGINAL_INPUT_PROJECTIONS",(unsigned long long)direct_file_hash(projected));
 int bad=0,shown=0;const char*inject=getenv("QSA_DIRECT_INJECT");
 for(unsigned step=0;step<steps;step++){
  unsigned t=t0+step,nv=(t+1)/4,n=qsa_nkeys(t,budget);float cs[32],sn[32];
  if(step&&frames){if(fread(&frame,sizeof frame,1,frames)!=1)return 2;if(memcmp(frame.qn,initial.qn,sizeof frame.qn)||memcmp(frame.kn,initial.kn,sizeof frame.kn)||memcmp(frame.iqn,initial.iqn,sizeof frame.iqn)||memcmp(frame.ikn,initial.ikn,sizeof frame.ikn)){fprintf(stderr,"norm weights changed\n");return 2;}}
  if(step&&projection.enabled)direct_projection_cpu(&projection,&frame,t,seed);
  for(size_t j=0;j<sizeof frame/sizeof(float);j++)if(!isfinite(((float*)&frame)[j])){fprintf(stderr,"nonfinite frame at step %u\n",step);return 2;}
  if(!step&&direct_replay){direct_replay_io io={dQG,dK,dV,dIQK,dCS,dSN,dKQ,dKS,dVQ,dVS,dPKT,dRING};int rc=direct_replay_prefix(p,t0,seed,&io,kq,ks,vq,vs);if(rc)return rc;}
  if(projection.enabled){CK(cuMemcpyHtoD(projection.z,projection.input,sizeof projection.input));}
  else{CK(cuMemcpyHtoD(dQG,frame.qg,sizeof frame.qg));CK(cuMemcpyHtoD(dK,frame.k,sizeof frame.k));CK(cuMemcpyHtoD(dV,frame.v,sizeof frame.v));CK(cuMemcpyHtoD(dIQK,frame.iqk,sizeof frame.iqk));}
  rope_table_gpu(t,1e7f,cs,sn);for(int h=0;h<24;h++)qk_norm_rope_gpu(frame.qg+512*h,frame.qn,eps,cs,sn,qo+256*h);
  for(int kv=0;kv<2;kv++){float kt[256];qk_norm_rope_gpu(frame.k+256*kv,frame.kn,eps,cs,sn,kt);for(int g=0;g<8;g++){size_t row=(size_t)t*2+kv;kv_quant32_gpu(kt+32*g,kq+row*256+32*g,ks+row*8+g);kv_quant32_gpu(frame.v+256*kv+32*g,vq+row*256+32*g,vs+row*8+g);}}
  for(int h=0;h<4;h++)idx_norm_rope_gpu(frame.iqk+128*h,frame.iqn,eps,cs,sn,iq+128*h);memcpy(ring+(t&3)*128,frame.iqk+512,512);
  if((t&3)==3){unsigned b=t/4;float pc[32],ps[32];rope_table_gpu(t-3,1e7f,pc,ps);idx_pool_gpu(ring,frame.ikn,eps,pc,ps,pool+(size_t)b*128);for(int d=0;d<128;d++)pkt[pkt_idx(b,d)]=pool[(size_t)b*128+d];}
  idx_scores_gpu(iq,pool,nv,score);idx_select_gpu(score,nv,budget,sel);
  if(qsa_gather_gpu(kq,ks,vq,vs,sel,t,budget,kc,vc)!=n)return 3;
  attn_partials_gpu(qo,kc,vc,n,pm,pd,po);attn_merge_gpu(pm,pd,po,frame.qg,out);run_token(allprog,projection_n+11,t);int b=0;
#define DC(name,want,dev,bytes) do{b|=direct_compare(name,want,dev,bytes,&shown);}while(0)
  if(projection.enabled){DC("Q projection",frame.qg,dQG,sizeof frame.qg);DC("K projection",frame.k,dK,sizeof frame.k);DC("V projection",frame.v,dV,sizeof frame.v);DC("index projection",frame.iqk,dIQK,sizeof frame.iqk);}
  DC("QO",qo,dQO,sizeof qo);DC("IQ",iq,dIQ,sizeof iq);DC("ring",ring,dRING,sizeof ring);
  DC("K codes",kq+(size_t)t*512,dKQ+(size_t)t*512,512);DC("V codes",vq+(size_t)t*512,dVQ+(size_t)t*512,512);DC("K scales",ks+(size_t)t*16,dKS+(size_t)t*32,32);DC("V scales",vs+(size_t)t*16,dVS+(size_t)t*32,32);
  if((t&3)==3){size_t off=(size_t)((t/4)>>5)*8192;DC("new pool group",(const char*)pkt+off,dPKT+off,8192);}
  if(nv>budget){if(inject&&!strcmp(inject,"selection"))sel[0]^=1;DC("score bits",score,dSC,4*(size_t)nv);DC("selected IDs",sel,dSEL,4*budget);}
  if(inject&&!strcmp(inject,"gather"))direct_flip(kc+attn_kidx(0,0,0));
  float*kg=malloc(4*kcf),*vg=malloc(4*(size_t)n*512);if(!kg||!vg)return 1;CK(cuMemcpyDtoH(kg,dKC,4*kcf));CK(cuMemcpyDtoH(vg,dVC,4*(size_t)n*512));int gb=0;
  for(unsigned j=0;j<n;j++)for(int kv=0;kv<2;kv++)for(int d=0;d<256;d++){size_t ki=attn_kidx(j,kv,d),vi=(size_t)j*512+kv*256+d;if(memcmp(kg+ki,kc+ki,4)||memcmp(vg+vi,vc+vi,4))gb++;}free(kg);free(vg);if(gb){if(shown++<6)printf("compact K/V mismatch: %d\n",gb);b=1;}
  DC("partial maxima",pm,dPM,sizeof pm);DC("partial denominators",pd,dPD,sizeof pd);DC("partial values",po,dPO,sizeof po);if(inject&&!strcmp(inject,"output"))direct_flip(out);DC("attention output",out,dOUT,sizeof out);
#undef DC
  printf("visible=%u complete=%u tail=%u selected=%u keys=%u exclusion=%s %s\n",t+1,nv,(t+1)&3,nv<budget?nv:budget,n,nv>budget?"yes":"no",b?"FAIL":"EXACT");bad+=b;
 }
 direct_projection_fini(&projection);if(frames)fclose(frames);free(kq);free(vq);free(ks);free(vs);free(pool);free(pkt);free(score);free(kc);free(vc);free(sel);
 printf("direct QSA %s: %u steps, %d mismatching steps; synthetic cache fixture\n",bad?"FAIL":"PASS",steps,bad);return bad?3:0;
}
static int direct_parse(const char*s,unsigned lo,unsigned hi,unsigned*out){if(!s||!*s)return 1;unsigned v=0;for(;*s;s++){if(*s<'0'||*s>'9')return 1;unsigned d=(unsigned)(*s-'0');if(v>hi/10||(v==hi/10&&d>hi%10))return 1;v=v*10+d;}if(v<lo)return 1;*out=v;return 0;}
int main(int argc,char**argv){
 if(argc<3||argc>4){fprintf(stderr,"usage: qsa_direct_test visible NBLK [steps=4]\n");return 2;}
 unsigned v,k,n=4;if(direct_parse(argv[1],1,262144,&v)||direct_parse(argv[2],1,4096,&k)||(argc==4&&direct_parse(argv[3],1,16,&n))||n>262145-v){fprintf(stderr,"invalid visible/NBLK/steps\n");return 2;}
 const char*inj=getenv("QSA_DIRECT_INJECT");if(inj&&strcmp(inj,"selection")&&strcmp(inj,"gather")&&strcmp(inj,"output")){fprintf(stderr,"unknown injection\n");return 2;}if(inj&&!strcmp(inj,"selection")&&(v+n-1)/4<=k){fprintf(stderr,"selection injection needs exclusion\n");return 2;}
 const char*replay=getenv("QSA_REPLAY_PREFIX");if(replay&&strcmp(replay,"0")&&strcmp(replay,"1")){fprintf(stderr,"QSA_REPLAY_PREFIX must be 0 or 1\n");return 2;}direct_replay=replay&&!strcmp(replay,"1");
 const char*pattern=getenv("QSA_PATTERN");if(pattern&&strcmp(pattern,"hash")&&strcmp(pattern,"oldest")&&strcmp(pattern,"newest")&&strcmp(pattern,"clustered")&&strcmp(pattern,"scattered")&&strcmp(pattern,"equal")&&strcmp(pattern,"adjacent"))return 2;if(direct_replay&&(v>65||(pattern&&strcmp(pattern,"hash")))){fprintf(stderr,"prefix replay needs <=64 history tokens and hash\n");return 2;}
 const char*host=getenv("QSA_HOSTKV");if(host&&strcmp(host,"0")&&strcmp(host,"1")){fprintf(stderr,"QSA_HOSTKV must be 0 or 1\n");return 2;}direct_host=host&&!strcmp(host,"1");
 unsigned seed=4242;const char*se=getenv("QSA_DIRECT_SEED");if(se&&direct_parse(se,0,0xffffffffu,&seed))return 2;
 dev_init();return test_direct(v,k,n,seed);
}
