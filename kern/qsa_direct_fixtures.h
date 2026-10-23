#ifndef QSA_DIRECT_FIXTURES_H
#define QSA_DIRECT_FIXTURES_H
static int direct_replay;
static void direct_history_kv(unsigned t,unsigned seed,float*k,float*v){for(unsigned d=0;d<512;d++){k[d]=direct_raw(t,d,seed+41u);v[d]=direct_raw(t,d,seed+83u);}}
static void direct_seed_kv(unsigned t0,unsigned npos,unsigned seed,const direct_frame*f,int8_t*kq,uint16_t*ks,int8_t*vq,uint16_t*vs){
 memset(kq,127,(size_t)npos*512);memset(vq,0x81,(size_t)npos*512);for(size_t i=0;i<(size_t)npos*16;i++)ks[i]=vs[i]=qsa_f2h(4.f);
 for(unsigned t=0;t<t0;t++){float k[512],v[512],cs[32],sn[32];direct_history_kv(t,seed,k,v);rope_table_gpu(t,1e7f,cs,sn);for(int h=0;h<2;h++){float kt[256];qk_norm_rope_gpu(k+256*h,f->kn,1e-6f,cs,sn,kt);for(int g=0;g<8;g++){size_t row=(size_t)t*2+h;kv_quant32_gpu(kt+32*g,kq+row*256+32*g,ks+row*8+g);kv_quant32_gpu(v+256*h+32*g,vq+row*256+32*g,vs+row*8+g);}}}
}
static void direct_seed_ring(unsigned t0,unsigned seed,float ring[512]){memset(ring,0,2048);for(unsigned j=0;j<4&&j<t0;j++){unsigned t=t0-1-j;for(unsigned d=0;d<128;d++)ring[(t&3)*128+d]=direct_raw(t,d,seed);}}
static void direct_history_pattern(const char*name,unsigned t0,unsigned budget,const direct_frame*f,uint16_t*pool,uint16_t*pkt){
 if(!name||!strcmp(name,"hash"))return;unsigned nv=t0/4;if(!nv){fprintf(stderr,"history pattern needs complete block\n");exit(2);}
 float iq[512],cs[32],sn[32];rope_table_gpu(t0,1e7f,cs,sn);for(int h=0;h<4;h++)idx_norm_rope_gpu(f->iqk+128*h,f->iqn,1e-6f,cs,sn,iq+128*h);
 uint16_t base[128]={0},other[128]={0};float score0=0;int d0=-1;
 for(int d=0;d<128&&d0<0;d++)for(int sg=0;sg<2;sg++){memset(base,0,sizeof base);base[d]=qsa_f2h(sg?-1.f:1.f);idx_scores_gpu(iq,base,1,&score0);if(score0>0){d0=d;break;}}
 if(d0<0){fprintf(stderr,"history pattern requires nonzero query\n");exit(2);}memcpy(other,base,sizeof base);
 if(!strcmp(name,"adjacent")){uint32_t u0;memcpy(&u0,&score0,4);int found=0;for(int d=0;d<128&&!found;d++)if(d!=d0)for(unsigned sg=0;sg<2&&!found;sg++)for(unsigned h=1;h<=64;h++){memcpy(other,base,sizeof base);other[d]=(uint16_t)((sg?0x8000u:0u)|h);float v;uint32_t u;idx_scores_gpu(iq,other,1,&v);memcpy(&u,&v,4);if(v>0&&(u==u0+1||u+1==u0)){printf("adjacent fixture score_bits=%08x,%08x\n",u0,u);found=1;break;}}if(!found){fprintf(stderr,"cannot form adjacent-score fixture\n");exit(2);}}
 for(unsigned b=0;b<nv;b++){uint16_t*key=pool+(size_t)b*128;if(!strcmp(name,"adjacent"))memcpy(key,(b&1)?other:base,sizeof base);
 else{int high=0;if(!strcmp(name,"oldest"))high=b<budget;else if(!strcmp(name,"newest"))high=(uint64_t)b+budget>=nv;else if(!strcmp(name,"clustered")){unsigned width=budget<nv?budget:nv,lo=(nv-width)/2;high=b>=lo&&b<lo+width;}else if(!strcmp(name,"scattered"))high=((uint64_t)(b+1)*budget/nv)!=((uint64_t)b*budget/nv);else if(strcmp(name,"equal"))exit(2);memcpy(key,base,sizeof base);if(high)key[d0]=qsa_f2h(qsa_h2f(base[d0])*2.f);}
 for(int d=0;d<128;d++)pkt[pkt_idx(b,d)]=key[d];}
 printf("SYNTHETIC pool override pattern=%s complete_historical=%u\n",name,nv);
}
typedef struct{CUdeviceptr qg,k,v,iqk,cs,sn,kq,ks,vq,vs,pkt,ring;}direct_replay_io;
static int direct_replay_prefix(const desc_t*chain,unsigned t0,unsigned seed,const direct_replay_io*io,const int8_t*kq,const uint16_t*ks,const int8_t*vq,const uint16_t*vs){
 if(!direct_replay||!t0)return 0;if(t0>64)return 2;
 CK(cuMemsetD8(io->kq,0,(size_t)t0*512));CK(cuMemsetD8(io->vq,0,(size_t)t0*512));CK(cuMemsetD8(io->ks,0,(size_t)t0*32));CK(cuMemsetD8(io->vs,0,(size_t)t0*32));CK(cuMemsetD8(io->ring,0,2048));if(t0/4)CK(cuMemsetD8(io->pkt,0,((size_t)t0/128+1)*8192));
 float qg[12288]={0},iqk[640]={0};CK(cuMemcpyHtoD(io->qg,qg,sizeof qg));
 for(unsigned t=0;t<t0;t++){float k[512],v[512],cs[32],sn[32];direct_history_kv(t,seed,k,v);rope_table_gpu(t,1e7f,cs,sn);for(unsigned d=0;d<128;d++)iqk[512+d]=direct_raw(t,d,seed);CK(cuMemcpyHtoD(io->k,k,sizeof k));CK(cuMemcpyHtoD(io->v,v,sizeof v));CK(cuMemcpyHtoD(io->iqk,iqk,sizeof iqk));CK(cuMemcpyHtoD(io->cs+128*(size_t)t,cs,128));CK(cuMemcpyHtoD(io->sn+128*(size_t)t,sn,128));run_token(chain,2,t);}
 int bad=0,shown=0;bad|=direct_compare("sequential prefix K",kq,io->kq,(size_t)t0*512,&shown);bad|=direct_compare("sequential prefix V",vq,io->vq,(size_t)t0*512,&shown);bad|=direct_compare("sequential prefix Ks",ks,io->ks,(size_t)t0*32,&shown);bad|=direct_compare("sequential prefix Vs",vs,io->vs,(size_t)t0*32,&shown);
 printf("direct/sequential seeded prefix: %u tokens %s; current chain uses same direct expected state\n",t0,bad?"FAIL":"EXACT");return bad?3:0;
}
#endif
