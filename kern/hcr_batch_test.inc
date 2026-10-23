/* Existing kernels only: original HCRX/W, grouped original, and actual
 * HCWSUM(nd=1)->RMS2560->shared HCX8. CPU entry never initializes CUDA. */
#define HCR_CHECK(x) do { if(!(x)){fprintf(stderr,"HCR %s:%d: %s\n",__FILE__,__LINE__,#x);exit(2);} } while(0)
static uint64_t hcr_pointer(const phase_t *a,unsigned w) { return a->w[w]|((uint64_t)a->w[w+1]<<32); }
static void hcr_descriptor(phase_t *a,int rw,CUdeviceptr W,CUdeviceptr norm,CUdeviceptr S,CUdeviceptr S2,
    CUdeviceptr XH,CUdeviceptr table,CUdeviceptr Y,CUdeviceptr inj,unsigned rows,unsigned offset) {
    memset(a,0,sizeof *a);a->w[0]=rw?OP_BF16HCRWX:OP_BF16HCRX;a->w[1]=16;a->w[30]=3;
    ph_ptr(a,0x140,W);ph_ptr(a,0x148,S);ph_ptr(a,0x150,table);ph_u32(a,0x158,rows);
    ph_u32(a,0x168,offset);ph_ptr(a,0x178,XH);ph_ptr(a,0x190,norm);ph_f32(a,0x198,1e-6f);
    if(rw){ph_ptr(a,0x160,S2);ph_ptr(a,0x1a0,Y);ph_ptr(a,0x1a8,inj);}
}
static int hcr_plan_test(void) {
    const unsigned widths[]={2,8,16,32};unsigned cases=0;
    for(unsigned v=0;v<4;v++)for(int rw=0;rw<2;rw++)for(unsigned change=0;change<25;change++) {
        unsigned width=widths[v];prog_t p;prog_init(&p);CUdeviceptr scratch[32];
        p.xj=7;p.pending_wait=1;p.loop=1;
        for(unsigned j=0;j<width;j++) {
            CUdeviceptr b=0x200000000ull+0x100000ull*j;scratch[j]=b+0x60000;
            phase_t *a=prog_add(&p,OP_BF16HCRX,16);
            hcr_descriptor(a,rw,0x100000000ull,0x101000000ull,b+0x1000,b+0x50000,b+0x40000,b+0x70000,b+0x20000,b+0x30000,65,128);
        }
        p.pending_wait=1;p.p[0].w[30]|=PHASE_GROUP_FLAG|(width<<PHASE_WIDTH_SHIFT);
        phase_t *a=p.p,*z=p.p+width-1;
        if(change==1)for(unsigned j=0;j<width;j++)p.p[j].w[30]&=~2u;
        if(change==2)ph_ptr(z,0x140,0x100001000ull);
        if(change==3)ph_ptr(z,0x190,0x101001000ull);
        if(change==4)z->w[8]++;
        if(change==5)z->w[24]++;
        if(change==6)z->w[31]=1;
        if(change==7)a->w[30]&=~1u;
        if(change==8)z->w[1]=32;
        if(change==9)ph_ptr(z,0x150,hcr_pointer(a,6));
        if(change==10)ph_ptr(z,0x178,hcr_pointer(a,4));
        if(change==11)ph_ptr(z,0x178,hcr_pointer(a,16));
        if(change==12)ph_ptr(z,0x190,0);
        if(change==13)ph_ptr(z,0x178,0);
        if(change==14)a->w[8]=0;
        if(change==15)a->w[8]=325;
        if(change==16)a->w[12]=2*XSLOT;
        if(change==17)a->w[12]++;
        if(change==19)ph_ptr(z,rw?0x160:0x178,hcr_pointer(a,4));
        if(change==20){if(rw)scratch[width-1]=hcr_pointer(a,26);else ph_ptr(z,0x178,hcr_pointer(a,22));}
        if(change==21)ph_ptr(z,rw?0x1a0:0x148,0);
        if(change==22)ph_ptr(z,rw?0x1a8:0x140,0);
        if(change==24)ph_ptr(z,rw?0x160:0x178,hcr_pointer(z,rw?16:6));
        phase_t saved[32];memcpy(saved,p.p,width*sizeof *saved);
        int expected=width>=8 && change<2;
        int changed=prog_batch_decompose_hcr(&p,change==23?NULL:scratch,width,change!=18 && !(change==23&&!rw));
        HCR_CHECK(changed==expected && p.xj==7 && p.pending_wait==1 && p.loop==1);
        if(!expected)HCR_CHECK(p.n==(int)width && !memcmp(saved,p.p,width*sizeof *saved));
        else {
            unsigned parts=rw?3:2;HCR_CHECK(p.n==(int)(parts*width));
            for(unsigned j=0;j<width;j++) {
                phase_t *r=p.p+(rw?width:0)+j,*hc=r+width;
                HCR_CHECK(r->w[0]==OP_RMS2560 && r->w[1]==4 && r->w[8]==saved[j].w[24]);
                HCR_CHECK(hcr_pointer(r,2)==hcr_pointer(saved+j,rw?10:4) && hcr_pointer(r,4)==hcr_pointer(saved+j,22));
                HCR_CHECK(hcr_pointer(r,6)==hcr_pointer(saved+j,16));
                HCR_CHECK(hc->w[0]==OP_BF16HCX && hcr_pointer(hc,4)==hcr_pointer(saved+j,16));
                HCR_CHECK((hc->w[30]&7u)==1 && (r->w[30]&7u)==(rw||change==1?0:2));
                if(rw){phase_t *w=p.p+j;HCR_CHECK(w->w[0]==OP_HCWSUM && w->w[11]==1 && hcr_pointer(w,12)==scratch[j]);
                    HCR_CHECK(hcr_pointer(w,14)==hcr_pointer(saved+j,4) && hcr_pointer(w,6)==hcr_pointer(saved+j,10));
                    HCR_CHECK((w->w[30]&7u)==(change==1?0:2));}
            }
            HCR_CHECK(prog_batch_fuse_bf16(&p,1)==(int)(width/8));
            HCR_CHECK(!prog_batch_decompose_hcr(&p,scratch,width,1));
        }
        prog_free(&p);cases++;
    }
    printf("HCR B2/B8/B16/B32 plan: %u eligibility/alias/ABI/exchange cases PASS\n",cases);return 0;
}
static float hcr_launch(CUfunction fn,CUdeviceptr program,CUdeviceptr ctl,CUdeviceptr flags,
    const phase_t *p,unsigned descriptors,unsigned width,unsigned mode,unsigned die) {
    uint32_t control[64]={0},f[96];for(unsigned i=0;i<96;i++)f[i]=i/16==die?17u:UINT32_MAX;
    for(unsigned e=0;e<6;e++){uint64_t row=flags+die*64;memcpy(control+16+2*e,&row,8);}
    uint64_t own=flags;memcpy(control+28,&own,8);control[3]=die;control[4]=6;control[8]=17;control[9]=mode?width:0;
    CK(cuMemsetD8(ctl,0,8192));CK(cuMemcpyHtoD(ctl,control,sizeof control));CK(cuMemcpyHtoD(flags,f,sizeof f));
    CK(cuMemcpyHtoD(program,p,descriptors*sizeof *p));unsigned end=mode?width:1;
    void *args[]={&program,&descriptors,&ctl,&end};CUevent begin,finish;float ms;
    CK(cuEventCreate(&begin,0));CK(cuEventCreate(&finish,0));CK(cuEventRecord(begin,0));
    CK(cuLaunchKernel(fn,16,1,1,256,1,1,0,0,args,NULL));CK(cuEventRecord(finish,0));CK(cuEventSynchronize(finish));
    CK(cuEventElapsedTime(&ms,begin,finish));CK(cuEventDestroy(begin));CK(cuEventDestroy(finish));return ms;
}
static int hcr_batch_test(CUfunction fn) {
    enum { MAXP=32,N=10240,INPUT=12804 };unsigned width=8;
    const char *option=getenv("HCR_POSITIONS");if(option){char *end;width=(unsigned)strtoul(option,&end,10);HCR_CHECK(!*end && (width==8||width==16||width==32));}
    const unsigned row_cases[]={1,17,65,137,324};const float edge[]={-352,-348,-64,-4,-0.f,0.f,.000244140625f,4,64,348,352};
    unsigned cases=0;int failures=0;
    for(int rw=0;rw<2;rw++)for(unsigned pattern=0;pattern<3;pattern++)for(unsigned variant=0;variant<10;variant++) {
        unsigned rows=row_cases[variant/2],slot=(variant&1)*XSLOT,offset=slot+128,die=(variant+pattern)%6;
        size_t peer_bytes=(size_t)6*2*XSLOT;
        float *inputs=malloc((size_t)width*INPUT*4),*norm=malloc(N*4),*ref_s2=malloc((size_t)width*N*4),*ref_xh=malloc((size_t)width*N*4);
        float *ref_dot=malloc((size_t)width*rows*4),*scalar_s2=malloc((size_t)width*N*4),*scalar_xh=malloc((size_t)width*N*4);
        float *scalar_peer=malloc(width*peer_bytes),*expected_peer=malloc(peer_bytes),*got=malloc(peer_bytes);
        uint16_t *weights=malloc((size_t)rows*N*2);HCR_CHECK(inputs&&norm&&ref_s2&&ref_xh&&ref_dot&&scalar_s2&&scalar_xh&&scalar_peer&&expected_peer&&got&&weights);
        for(unsigned i=0;i<N;i++)norm[i]=.5f+.25f*frand();
        for(size_t i=0;i<(size_t)rows*N;i++){float w=.125f*frand();uint32_t bits;memcpy(&bits,&w,4);weights[i]=(uint16_t)(bits>>16);}
        for(unsigned p=0;p<width;p++) {
            float *s=inputs+(size_t)p*INPUT,*y=s+N,*inj=y+2560,*updated=ref_s2+(size_t)p*N,*xh=ref_xh+(size_t)p*N;
            for(unsigned i=0;i<N;i++)s[i]=!pattern?3.f*frand():pattern==1?((i+p)%3?-1.f:1.f)*(i%71)*.03125f:edge[(i+7*p)%(sizeof edge/sizeof *edge)]*.01f;
            for(unsigned i=0;i<2560;i++)y[i]=2.f*frand();
            for(unsigned l=0;l<4;l++)inj[l]=edge[(p+3*l+pattern)%(sizeof edge/sizeof *edge)];
            memcpy(updated,s,N*4);if(rw)hc_write_gpu(inj,y,updated);
            for(unsigned l=0;l<4;l++)rmsnorm_gpu(updated+2560*l,norm+2560*l,2560,1e-6f,xh+2560*l);
            for(unsigned r=0;r<rows;r++) {
                float dot=0;
                for(unsigned l=0;l<4;l++){float partial;gemv_bf16_ref(weights+(size_t)r*N+2560*l,1,2560,xh+2560*l,&partial);dot=l?dot+partial:partial;}
                ref_dot[p*rows+r]=dot;
            }
        }
        CUdeviceptr dW=hcup_copy("HCR weights",weights,(size_t)rows*N*2),dNorm=hcup_copy("HCR norm",norm,N*4);
        CUdeviceptr dCTL=hcup_copy("HCR control",NULL,8192),dF=hcup_copy("HCR flags",NULL,96*4),dP=hcup_copy("HCR program",NULL,3*width*sizeof(phase_t));
        CUdeviceptr dIn[MAXP],dS2[MAXP],dXH[MAXP],dOut[MAXP],dPeers[MAXP],dTable[MAXP],scratch[MAXP];uint64_t table[MAXP][12];
        phase_t scalar[MAXP],group[MAXP];
        for(unsigned p=0;p<width;p++) {
            dIn[p]=hcup_copy("HCR S/Y/inj",inputs+(size_t)p*INPUT,INPUT*4);dS2[p]=hcup_copy("HCR S2",NULL,N*4);dXH[p]=hcup_copy("HCR XH",NULL,N*4);
            dOut[p]=scratch[p]=hcup_copy("HCR dead output",NULL,2560*4);dPeers[p]=hcup_copy("HCR six peers",NULL,peer_bytes);
            for(unsigned e=0;e<6;e++){table[p][e]=dPeers[p]+e*2*XSLOT;table[p][6+e]=dF;}
            dTable[p]=hcup_copy("HCR table",table[p],sizeof table[p]);
            hcr_descriptor(scalar+p,rw,dW,dNorm,dIn[p],dS2[p],dXH[p],dTable[p],dIn[p]+N*4,dIn[p]+(N+2560)*4,rows,offset);
        }
        memcpy(group,scalar,width*sizeof *group);group[0].w[30]|=PHASE_GROUP_FLAG|(width<<PHASE_WIDTH_SHIFT);
        prog_t plan;prog_init(&plan);free(plan.p);plan.n=plan.cap=width;plan.p=malloc(width*sizeof *group);memcpy(plan.p,group,width*sizeof *group);
        HCR_CHECK(prog_batch_decompose_hcr(&plan,scratch,width,1)==1);HCR_CHECK(prog_batch_fuse_bf16(&plan,1)==(int)(width/8));
        float times[3];int cpu_bad=0,gpu_bad=0,other_bad=0;
        for(unsigned mode=0;mode<3;mode++) {
            for(unsigned p=0;p<width;p++){CK(cuMemsetD32(dS2[p],0x7fc5a5a5u,N));CK(cuMemsetD32(dXH[p],0x7fc5a5a5u,N));CK(cuMemsetD32(dOut[p],0x7fc5a5a5u,2560));CK(cuMemsetD32(dPeers[p],0x7fc5a5a5u,peer_bytes/4));}
            const phase_t *program=mode==2?plan.p:mode?group:scalar;unsigned nph=mode==2?(unsigned)plan.n:width;
            times[mode]=hcr_launch(fn,dP,dCTL,dF,program,nph,width,mode,die);
            for(unsigned p=0;p<width;p++) {
                CK(cuMemcpyDtoH(got,dXH[p],N*4));cpu_bad+=hcup_bits("HCR XH CPU",ref_xh+(size_t)p*N,got,N);
                if(mode)gpu_bad+=hcup_bits("HCR XH scalar",scalar_xh+(size_t)p*N,got,N);else memcpy(scalar_xh+(size_t)p*N,got,N*4);
                CK(cuMemcpyDtoH(got,dS2[p],N*4));
                if(rw)cpu_bad+=hcup_bits("HCR S2 CPU",ref_s2+(size_t)p*N,got,N);else for(unsigned i=0;i<N;i++){uint32_t bits;memcpy(&bits,got+i,4);other_bad+=bits!=0x7fc5a5a5u;}
                if(mode)gpu_bad+=hcup_bits("HCR S2 scalar",scalar_s2+(size_t)p*N,got,N);else memcpy(scalar_s2+(size_t)p*N,got,N*4);
                CK(cuMemcpyDtoH(got,dOut[p],2560*4));
                if(mode==2 && rw)other_bad+=hcup_bits("HCR nd1 Y copy",inputs+(size_t)p*INPUT+N,got,2560);
                else for(unsigned i=0;i<2560;i++){uint32_t bits;memcpy(&bits,got+i,4);other_bad+=bits!=0x7fc5a5a5u;}
                for(size_t i=0;i<peer_bytes/4;i++){uint32_t bits=0x7fc5a5a5u;memcpy(expected_peer+i,&bits,4);}
                for(unsigned e=0;e<6;e++)memcpy((char *)expected_peer+e*2*XSLOT+offset,ref_dot+p*rows,rows*4);
                CK(cuMemcpyDtoH(got,dPeers[p],peer_bytes));cpu_bad+=hcup_bits("HCR peers CPU",expected_peer,got,peer_bytes/4);
                if(mode)gpu_bad+=hcup_bits("HCR peers scalar",scalar_peer+p*peer_bytes/4,got,peer_bytes/4);else memcpy((char *)scalar_peer+p*peer_bytes,got,peer_bytes);
                other_bad+=hcup_unchanged("HCR input",dIn[p],inputs+(size_t)p*INPUT,INPUT*4)+hcup_unchanged("HCR table",dTable[p],table[p],sizeof table[p]);
            }
            other_bad+=hcup_unchanged("HCR weights",dW,weights,(size_t)rows*N*2)+hcup_unchanged("HCR norm",dNorm,norm,N*4);
            other_bad+=hcup_unchanged("HCR program",dP,program,nph*sizeof *program)+test_guard_check();
            uint32_t ctl[64],f[96];CK(cuMemcpyDtoH(ctl,dCTL,sizeof ctl));CK(cuMemcpyDtoH(f,dF,sizeof f));
            unsigned exchanges=mode?1:width,grids=mode==2?(rw?3:2):mode?1:width;
            other_bad+=ctl[2]!=(mode?width:1)||ctl[1]!=grids||ctl[8]!=17+exchanges||ctl[11]!=0;
            for(unsigned i=0;i<96;i++)other_bad+=f[i]!=(i/16==die?17+exchanges:UINT32_MAX);
        }
        printf("HCR%s width=%u rows=%u pattern=%u slot=%u groups_changed=1 HCX8_bodies=%u cpu_mismatches=%d gpu_mismatches=%d guard_exchange_failures=%d scalar_ms=%.6f grouped_ms=%.6f decomposed_ms=%.6f speedup=%.6f\n",rw?"WX":"X",width,rows,pattern,slot,width/8,cpu_bad,gpu_bad,other_bad,times[0],times[1],times[2],times[1]/times[2]);
        failures+=!!(cpu_bad||gpu_bad||other_bad);cases++;prog_free(&plan);test_guard_free();
        free(inputs);free(norm);free(ref_s2);free(ref_xh);free(ref_dot);free(scalar_s2);free(scalar_xh);free(scalar_peer);free(expected_peer);free(got);free(weights);
    }
    printf("HCR width=%u cases=%u failures=%d grouped/decomposed/scalar/C exact_bits=yes\n",width,cases,failures);return failures?3:0;
}
#undef HCR_CHECK
