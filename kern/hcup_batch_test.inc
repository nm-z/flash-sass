/* Included by die_test.c: the original HCUP fixture plus eight distinct
 * positions, both slots, all six peers, guarded allocations and a scalar GPU
 * control. The CPU-only entry runs before cuInit. */
#define HCUP_CHECK(x) do { if(!(x)){fprintf(stderr,"HCUP plan %s:%d: %s\n",__FILE__,__LINE__,#x);exit(2);} } while(0)
static int hcup_plan_test(void) {
    unsigned cases=0;
    const unsigned widths[]={2,8,16,32};
    for(unsigned v=0;v<4;v++)for(unsigned change=0;change<24;change++) {
        unsigned width=widths[v];prog_t positions[32],out;const prog_t *inputs[32];
        prog_init(&out);
        for(unsigned p=0;p<width;p++) {
            CUdeviceptr b=0x200000000ull+0x100000ull*p;
            xchg_t x={.xtab=b+0x80000,.inbox=b+0xa0000};
            prog_init(positions+p);positions[p].loop=1;positions[p].token_offset=p;
            positions[p].xj=3;positions[p].pending_wait=1;inputs[p]=positions+p;
            ph_hcup(positions+p,&x,0x100000000ull,b+0x1000,b+0x20000,448,b+0x40000,b+0x60000,change==1?320:324,128);
        }
        HCUP_CHECK(!prog_group(&out,inputs,width));
        phase_t *a=out.p,*z=out.p+width-1;int eligible=change<2;
        switch(change) {
        case 2:ph_ptr(z,0x140,0x100100000ull);break;
        case 3:ph_u32(z,0x158,a->w[8]+4);break;
        case 4:z->w[1]=8;break;
        case 5:z->w[31]=1;break;
        case 6:a->w[31]=1;break;
        case 7:z->w[30]^=2;break;
        case 8:ph_u32(z,0x188,320);break;
        case 9:ph_u32(z,0x168,a->w[12]+4);break;
        default:break;
        }
        if(change==10)ph_ptr(z,0x170,(uint64_t)a->w[14]|((uint64_t)a->w[15]<<32));
        if(change==11)ph_ptr(z,0x180,(uint64_t)a->w[4]|((uint64_t)a->w[5]<<32));
        if(change==12)ph_ptr(z,0x150,(uint64_t)a->w[16]|((uint64_t)a->w[17]<<32));
        if(change==13)ph_ptr(z,0x150,(uint64_t)a->w[18]|((uint64_t)a->w[19]<<32));
        if(change==14)ph_ptr(z,0x180,0x100000000ull);
        if(change==15)ph_ptr(z,0x150,(uint64_t)a->w[6]|((uint64_t)a->w[7]<<32));
        if(change==16)ph_ptr(z,0x148,0);
        if(change==17)a->w[8]=0;
        if(change==18)a->w[8]=10244;
        if(change==19)a->w[8]=1793;
        if(change==20)a->w[20]=321;
        if(change==21)a->w[30]|=4;
        if(change==22)a->w[30]&=~1u;
        int xj=out.xj,n=out.n,pending=out.pending_wait;
        int fused=prog_batch_fuse_hcup(&out,change!=23);
        HCUP_CHECK(fused==(eligible && width>=8?(int)(width/8):0));
        HCUP_CHECK(out.xj==xj && out.n==n && out.pending_wait==pending);
        if(fused)for(unsigned p=0;p<width;p+=8) {
            uint32_t flags=out.p[p].w[30];
            HCUP_CHECK(out.p[p].w[0]==OP_BF16HCUP8 && phase_positions(out.p+p)==8);
            HCUP_CHECK(!!(flags&2)==(p==0) && !!(flags&1)==(p+8==width) && !!(flags&4)==(p+8<width));
        }
        phase_t saved[32];memcpy(saved,out.p,width*sizeof *saved);
        HCUP_CHECK(!prog_batch_fuse_hcup(&out,change!=23) && !memcmp(saved,out.p,width*sizeof *saved));
        prog_free(&out);for(unsigned p=0;p<width;p++)prog_free(positions+p);cases++;
    }
    /* Every supported column count: uniform32-column bursts cover each local
     * column exactly once, including empty CTAs and short final warps. */
    for(unsigned cols=1;cols<=2560;cols++) {
        unsigned seen[2560]={0},count=(cols+15)/16;
        for(unsigned cta=0;cta<16;cta++) {
            unsigned c0=cta*count,c1=c0+count;if(c1>cols)c1=cols;if(c0>c1)c0=c1;
            HCUP_CHECK(c1-c0<=160);
            for(unsigned base=0;base<c1-c0;base+=32)for(unsigned lane=0;lane<32;lane++)if(base+lane<c1-c0) {
                seen[c0+base+lane]++;
                HCUP_CHECK(0x3200+7*640+4*(base+lane)+4<=17920);
            }
        }
        for(unsigned col=0;col<cols;col++)HCUP_CHECK(seen[col]==1);
    }
    printf("HCUP B2/B8/B16/B32 eligibility: %u cases and2560 column/burst partitions PASS\n",cases);
    return 0;
}

static int hcup_bits(const char *label,const float *ref,const float *got,size_t n) {
    int bad=0;
    for(size_t i=0;i<n;i++)if(memcmp(ref+i,got+i,4)) {
        if(bad<3)fprintf(stderr,"%s[%zu] got=%a reference=%a\n",label,i,(double)got[i],(double)ref[i]);
        bad++;
    }
    return bad;
}
static float hcup_batch_launch(CUfunction fn,CUdeviceptr dP,CUdeviceptr dCTL,CUdeviceptr dF,
    const phase_t *program,unsigned batch,unsigned die) {
    enum { CTA=16,DIES=6 };uint32_t ctl[64]={0},flags[CTA*DIES];
    for(unsigned i=0;i<CTA*DIES;i++)flags[i]=i/CTA==die?17u:UINT32_MAX;
    for(unsigned peer=0;peer<DIES;peer++){uint64_t row=dF+4*die*CTA;memcpy(ctl+16+2*peer,&row,8);}
    uint64_t own=dF;memcpy(ctl+28,&own,8);ctl[3]=die;ctl[4]=DIES;ctl[8]=17;ctl[9]=batch?8:launch_width;
    CK(cuMemsetD8(dCTL,0,8192));CK(cuMemcpyHtoD(dCTL,ctl,sizeof ctl));CK(cuMemcpyHtoD(dF,flags,sizeof flags));
    CK(cuMemcpyHtoD(dP,program,8*sizeof *program));
    unsigned nph=8,end=batch?8:1;void *args[]={&dP,&nph,&dCTL,&end};
    CUevent begin,finish;float ms;CK(cuEventCreate(&begin,0));CK(cuEventCreate(&finish,0));CK(cuEventRecord(begin,0));
    CK(cuLaunchKernel(fn,CTA,1,1,256,1,1,0,0,args,NULL));CK(cuEventRecord(finish,0));CK(cuEventSynchronize(finish));
    CK(cuEventElapsedTime(&ms,begin,finish));CK(cuEventDestroy(begin));CK(cuEventDestroy(finish));return ms;
}
static int hcup_batch_test(CUfunction fn) {
    enum { P=8,COLS=2560,K=320,DIES=6,CTA=16 };
    if(!getenv("SPLIT_SHARES"))setenv("SPLIT_SHARES","112,97,81,116,97,97",1);
    uint16_t *weights=malloc((size_t)4*COLS*K*2);
    float down[P][324],t[P][324],*xh=malloc(P*4*COLS*4),*up=malloc(P*4*COLS*4),*mixed=malloc(P*COLS*4);
    float *got=malloc(2*XSLOT),*peer_ref=malloc(2*XSLOT),*scalar_peer=malloc((size_t)P*DIES*2*XSLOT);
    float scalar_t[P][324],*scalar_up=malloc(P*4*COLS*4);
    if(!weights||!xh||!up||!mixed||!got||!peer_ref||!scalar_peer||!scalar_up)exit(2);
    for(size_t i=0;i<(size_t)4*COLS*K;i++){float w=.125f*frand();uint32_t bits;memcpy(&bits,&w,4);weights[i]=(uint16_t)(bits>>16);}
    const float edge[]={-352.f,-348.f,-64.f,-4.f,-0.f,0.f,.000244140625f,4.f,64.f,348.f,352.f};
    unsigned cases=0;int failures=0;
    for(unsigned pattern=0;pattern<3;pattern++) {
        for(unsigned p=0;p<P;p++) {
            for(unsigned i=0;i<K;i++)down[p][i]=!pattern?(20.f+p)*frand():pattern==1?
                ((i+p)%3?-1.f:1.f)*(.125f+(float)((i+13*p)%67)):edge[(7*i+i/64+p)%(sizeof edge/sizeof *edge)];
            const uint32_t inject[]={0x7fc12345u+p,0x80000000u,0x3f800001u+p,0xc2800000u};
            memcpy(down[p]+K,inject,sizeof inject);hc_silu_gpu(down[p],t[p]);memcpy(t[p]+K,inject,sizeof inject);
            for(unsigned i=0;i<4*COLS;i++)xh[p*4*COLS+i]=2.f*frand();
            gemv_bf16_ref(weights,4*COLS,K,t[p],up+p*4*COLS);hc_mix_gpu(up+p*4*COLS,xh+p*4*COLS,mixed+p*COLS);
        }
        const unsigned extra_cols[]={1,7,15,17,31,33,137,2559,2560};
        for(unsigned geometry=0;geometry<DIES+sizeof extra_cols/sizeof *extra_cols;geometry++) {
            unsigned die=geometry%DIES,base=geometry<DIES?32*split_bound(80,die,DIES):0;
            unsigned cols=geometry<DIES?32*split_bound(80,die+1,DIES)-base:extra_cols[geometry-DIES];
            HCUP_CHECK(cols && cols<=COLS);size_t wbytes=(size_t)4*cols*K*2,ubytes=4*cols*4;
            uint16_t *local_w=malloc(wbytes);float *local_up=malloc(ubytes);if(!local_w||!local_up)exit(2);
            for(unsigned lane=0;lane<4;lane++)memcpy(local_w+(size_t)lane*cols*K,weights+(size_t)(lane*COLS+base)*K,(size_t)cols*K*2);
            for(unsigned variant=0;variant<4;variant++) {
                unsigned tn=variant&1?324:320,slot=(variant>>1)*XSLOT;
                CUdeviceptr dW=hcup_copy("HCUP8 weights",local_w,wbytes),dCTL=hcup_copy("HCUP8 control",NULL,8192);
                CUdeviceptr dF=hcup_copy("HCUP8 flags",NULL,DIES*CTA*4),dP=hcup_copy("HCUP8 program",NULL,P*sizeof(phase_t));
                CUdeviceptr dD[P],dX[P],dU[P],dT[P],dTAB[P],inbox[P][DIES];uint64_t tables[P][12];
                phase_t scalar[P],batch[P];memset(scalar,0,sizeof scalar);
                for(unsigned p=0;p<P;p++) {
                    dD[p]=hcup_copy("HCUP8 down",down[p],tn*4);dX[p]=hcup_copy("HCUP8 XH",xh+p*4*COLS,4*COLS*4);
                    dU[p]=hcup_copy("HCUP8 up",NULL,ubytes);dT[p]=hcup_copy("HCUP8 T",NULL,tn*4);
                    for(unsigned e=0;e<DIES;e++){inbox[p][e]=hcup_copy("HCUP8 inbox",NULL,2*XSLOT);tables[p][e]=inbox[p][e];tables[p][DIES+e]=dF;}
                    dTAB[p]=hcup_copy("HCUP8 table",tables[p],sizeof tables[p]);
                    phase_t *a=scalar+p;a->w[0]=OP_BF16HCUP;a->w[1]=CTA;a->w[30]=3;
                    ph_ptr(a,0x140,dW);ph_ptr(a,0x148,dD[p]);ph_ptr(a,0x150,dU[p]);ph_u32(a,0x158,4*cols);
                    ph_u32(a,0x168,slot+base*4);ph_ptr(a,0x170,dTAB[p]);ph_ptr(a,0x178,dX[p]+base*4);ph_ptr(a,0x180,dT[p]);ph_u32(a,0x188,tn);
                }
                memcpy(batch,scalar,sizeof batch);batch[0].w[30]|=PHASE_GROUP_FLAG|(8u<<PHASE_WIDTH_SHIFT);
                prog_t plan={.p=batch,.n=P};HCUP_CHECK(prog_batch_fuse_hcup(&plan,1)==1);
                float times[2];int cpu_bad=0,gpu_bad=0,other_bad=0;
                for(unsigned mode=0;mode<2;mode++) {
                    for(unsigned p=0;p<P;p++) {
                        CK(cuMemsetD32(dU[p],0x7fc5a5a5u,4*cols));CK(cuMemsetD32(dT[p],0x7fc5a5a5u,tn));
                        for(unsigned e=0;e<DIES;e++)CK(cuMemsetD32(inbox[p][e],0x7fc5a5a5u,2*XSLOT/4));
                    }
                    times[mode]=hcup_batch_launch(fn,dP,dCTL,dF,mode?batch:scalar,mode,die);
                    for(unsigned p=0;p<P;p++) {
                        CK(cuMemcpyDtoH(got,dT[p],tn*4));cpu_bad+=hcup_bits("HCUP8 T CPU",t[p],got,tn);
                        if(mode)gpu_bad+=hcup_bits("HCUP8 T scalarGPU",scalar_t[p],got,tn);else memcpy(scalar_t[p],got,tn*4);
                        for(unsigned lane=0;lane<4;lane++)memcpy(local_up+lane*cols,up+(p*4+lane)*COLS+base,cols*4);
                        CK(cuMemcpyDtoH(got,dU[p],ubytes));cpu_bad+=hcup_bits("HCUP8 up CPU",local_up,got,4*cols);
                        if(mode)gpu_bad+=hcup_bits("HCUP8 up scalarGPU",scalar_up+p*4*COLS,got,4*cols);else memcpy(scalar_up+p*4*COLS,got,ubytes);
                        for(unsigned i=0;i<2*XSLOT/4;i++){uint32_t bits=0x7fc5a5a5u;memcpy(peer_ref+i,&bits,4);}
                        memcpy((char *)peer_ref+slot+base*4,mixed+p*COLS+base,cols*4);
                        for(unsigned e=0;e<DIES;e++) {
                            CK(cuMemcpyDtoH(got,inbox[p][e],2*XSLOT));cpu_bad+=hcup_bits("HCUP8 peer CPU",peer_ref,got,2*XSLOT/4);
                            float *saved=(float *)((char *)scalar_peer+((size_t)p*DIES+e)*2*XSLOT);
                            if(mode)gpu_bad+=hcup_bits("HCUP8 peer scalarGPU",saved,got,2*XSLOT/4);else memcpy(saved,got,2*XSLOT);
                        }
                        other_bad+=hcup_unchanged("down",dD[p],down[p],tn*4)+hcup_unchanged("XH",dX[p],xh+p*4*COLS,4*COLS*4);
                        other_bad+=hcup_unchanged("table",dTAB[p],tables[p],sizeof tables[p]);
                    }
                    other_bad+=hcup_unchanged("weights",dW,local_w,wbytes)+hcup_unchanged("program",dP,mode?batch:scalar,sizeof scalar)+test_guard_check();
                    uint32_t ctl[64],flags[DIES*CTA];CK(cuMemcpyDtoH(ctl,dCTL,sizeof ctl));CK(cuMemcpyDtoH(flags,dF,sizeof flags));
                    unsigned phases=mode?1:P;other_bad+=ctl[2]!=(mode?P:1)||ctl[1]!=phases||ctl[8]!=17+phases||ctl[11]!=0;
                    for(unsigned i=0;i<DIES*CTA;i++)other_bad+=flags[i]!=(i/CTA==die?17+phases:UINT32_MAX);
                }
                printf("HCUP8 pattern=%u geometry=%u die=%u cols=%u base=%u tn=%u slot=%u positions=8 peers=6 gpu_mismatches=%d cpu_mismatches=%d guard_exchange_failures=%d scalar_ms=%.6f batch_ms=%.6f speedup=%.6f\n",pattern,geometry,die,cols,base,tn,slot,gpu_bad,cpu_bad,other_bad,times[0],times[1],times[0]/times[1]);
                failures+=!!(cpu_bad||gpu_bad||other_bad);cases++;test_guard_free();
            }
            free(local_w);free(local_up);
        }
    }
    printf("HCUP8 cases=%u failures=%d scalarGPU_and_C_oracle_exact_bits=yes\n",cases,failures);
    free(weights);free(xh);free(up);free(mixed);free(got);free(peer_ref);free(scalar_peer);free(scalar_up);return failures?3:0;
}
#undef HCUP_CHECK
