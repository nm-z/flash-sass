/* Read-only audit of layer 47 scratch, after a complete scalar request.
 * Reference helpers expect 24 Q / 2 KV heads. Repeat the four local Q heads
 * and the one local KV head; compare only the four real Q heads / KV head 0.
 * Each stage consumes captured GPU inputs, not an upstream CPU prediction. */
#include "quant.h"
static unsigned qsa_audit_compare(FILE *out,int die,const char *name,
                                 const void *actual,const void *reference,
                                 size_t count,size_t width,int floats) {
    size_t changed=0,first=0;double max_abs=0;unsigned nonfinite=0;
    const unsigned char *a=actual,*r=reference;
    for(size_t i=0;i<count;i++) {
        if(memcmp(a+i*width,r+i*width,width)){if(!changed)first=i;changed++;}
        if(floats){float x,y;memcpy(&x,a+i*width,4);memcpy(&y,r+i*width,4);
            if(!isfinite(x)||!isfinite(y))nonfinite++;
            else if(fabs((double)x-y)>max_abs)max_abs=fabs((double)x-y);}
    }
    uint32_t ab=0,rb=0;
    if(count){memcpy(&ab,a+first*width,width);memcpy(&rb,r+first*width,width);}
    fprintf(out,"qsaaudit die=%d layer=47 component=%s count=%zu changed=%zu first=%zu actual_bits=%08x reference_bits=%08x max_abs=%.9g nonfinite=%u\n",
            die,name,count,changed,first,ab,rb,max_abs,nonfinite);
    return changed!=0 || nonfinite!=0;
}
static void qsa_audit(FILE *out) {
    unsigned failures=0,components=0;
    if(!row_balance || nd!=6 || nlayers_run!=48 || two_position || last_paired_positions ||
       !last_timing_valid || !last_timing.counts_final ||
       last_timing.error_code || last_positions<=0 || last_positions>MAXT || fegetround()!=FE_TONEAREST){
        fprintf(out,"qsaaudit_end status=FAIL components=0 failures=1 reason=requires_completed_balanced_scalar_full_model\n");fflush(out);return;
    }
    unsigned t=(unsigned)last_positions-1,n=qsa_nkeys(t,nblk_qsa);
    size_t raw_n=(size_t)last_positions+4,kcf=ATTN_KC_FLOATS(n),vcf=((size_t)n+3)/4*4*512;
    /* All allocations precede readback; cleanup is common on failure. */
    int8_t *kq=calloc(raw_n,512),*vq=calloc(raw_n,512),*rawq=malloc(raw_n*256);
    uint16_t *ks=calloc(raw_n,32),*vs=calloc(raw_n,32),*raws=malloc(raw_n*16);
    uint32_t *sel=calloc((size_t)nblk_qsa+8,4);
    float *kc=calloc(kcf,4),*vc=calloc(vcf,4),*rk=calloc(kcf,4),*rv=calloc(vcf,4);
    float *flat=malloc((size_t)n*256*4),*rflat=malloc((size_t)n*256*4);
    float *po=calloc(24*8*256,4),*rpo=calloc(24*8*256,4);
    if(!kq||!vq||!rawq||!ks||!vs||!raws||!sel||!kc||!vc||!rk||!rv||!flat||!rflat||!po||!rpo){
        failures++;fprintf(out,"qsaaudit error=allocation\n");goto cleanup;
    }
    fprintf(out,"qsaaudit_begin position=%u keys=%u layer=47 dies=6 reference=cpu_stage_twins inputs=retained_gpu\n",t,n);
    for(int d=0;d<nd;d++) {
        float qg[24*512],qo[24*256],rq[4*256],k[256],v[256],norm[256],cs[32],sn[32],nk[256];
        float pm[24*8],pd[24*8],rpm[24*8],rpd[24*8],aout[24*256],rout[24*256];
        int8_t codes[256];uint16_t scales[8];
        CK(cuCtxSetCurrent(ctx[d]));
#define QA_READ(dst,src,bytes) CK(cuMemcpyDtoH((dst),(src),(bytes)))
#define QA_CMP(name,a,r,count,width,isfloat) do { components++;failures+=qsa_audit_compare(out,d,name,a,r,count,width,isfloat); } while(0)
        QA_READ(qg,db[d].qg,4*4*512);QA_READ(qo,db[d].qo,4*4*256);
        QA_READ(k,db[d].k,sizeof k);QA_READ(v,db[d].v,sizeof v);
        QA_READ(cs,db[d].rope_cs+(size_t)t*128,sizeof cs);QA_READ(sn,db[d].rope_sn+(size_t)t*128,sizeof sn);
        QA_READ(norm,dw[d][47].attn_q_norm,sizeof norm);
        for(int h=0;h<4;h++)qk_norm_rope_gpu(qg+h*512,norm,eps,cs,sn,rq+h*256);
        QA_CMP("q_norm_rope",qo,rq,1024,4,1);
        QA_READ(norm,dw[d][47].attn_k_norm,sizeof norm);
        qk_norm_rope_gpu(k,norm,eps,cs,sn,nk);
        for(int g=0;g<8;g++)kv_quant32_gpu(nk+g*32,codes+g*32,scales+g);
        QA_READ(rawq,dst_kq[d][47]+(size_t)t*256,256);QA_READ(raws,dst_ks[d][47]+(size_t)t*16,16);
        QA_CMP("k_codes",rawq,codes,256,1,0);QA_CMP("k_scales",raws,scales,8,2,0);
        for(int g=0;g<8;g++)kv_quant32_gpu(v+g*32,codes+g*32,scales+g);
        QA_READ(rawq,dst_vq[d][47]+(size_t)t*256,256);QA_READ(raws,dst_vs[d][47]+(size_t)t*16,16);
        QA_CMP("v_codes",rawq,codes,256,1,0);QA_CMP("v_scales",raws,scales,8,2,0);
        for(int kv=0;kv<2;kv++) {
            QA_READ(rawq,kv?dst_vq[d][47]:dst_kq[d][47],(size_t)last_positions*256);
            QA_READ(raws,kv?dst_vs[d][47]:dst_ks[d][47],(size_t)last_positions*16);
            for(int p=0;p<last_positions;p++)for(int h=0;h<2;h++){
                memcpy((kv?vq:kq)+(size_t)p*512+h*256,rawq+(size_t)p*256,256);
                memcpy((kv?vs:ks)+(size_t)p*16+h*8,raws+(size_t)p*8,16);
            }
        }
        QA_READ(sel,db[d].sel,((size_t)nblk_qsa+1)*4);
        unsigned nv=(t+1)/4;int valid=1;
        if(nv>nblk_qsa)for(unsigned i=0;i<nblk_qsa;i++)if(sel[i]>=nv)valid=0;
        if(!valid){failures++;fprintf(out,"qsaaudit die=%d error=selection_out_of_bounds\n",d);continue;}
        qsa_gather_gpu(kq,ks,vq,vs,sel,t,nblk_qsa,rk,rv);
        QA_READ(kc,db[d].kc,kcf*4);QA_READ(vc,db[d].vc,vcf*4);
        for(unsigned p=0;p<n;p++)for(int j=0;j<256;j++){
            flat[(size_t)p*256+j]=kc[attn_kidx(p,0,j)];rflat[(size_t)p*256+j]=rk[attn_kidx(p,0,j)];
            kc[attn_kidx(p,1,j)]=kc[attn_kidx(p,0,j)];
        }
        QA_CMP("gather_k",flat,rflat,(size_t)n*256,4,1);
        for(unsigned p=0;p<n;p++){
            memcpy(flat+(size_t)p*256,vc+(size_t)p*512,1024);
            memcpy(rflat+(size_t)p*256,rv+(size_t)p*512,1024);
            memcpy(vc+(size_t)p*512+256,vc+(size_t)p*512,1024);
        }
        QA_CMP("gather_v",flat,rflat,(size_t)n*256,4,1);
        for(int h=4;h<24;h++){memcpy(qo+h*256,qo+(h%4)*256,1024);memcpy(qg+h*512,qg+(h%4)*512,2048);}
        attn_partials_gpu(qo,kc,vc,n,rpm,rpd,rpo);
        QA_READ(pm,db[d].pm,4*4*8);QA_READ(pd,db[d].pd,4*4*8);QA_READ(po,db[d].po,4*4*8*256);
        QA_CMP("partial_max",pm,rpm,32,4,1);QA_CMP("partial_den",pd,rpd,32,4,1);QA_CMP("partial_out",po,rpo,8192,4,1);
        for(int h=4;h<24;h++){
            memcpy(pm+h*8,pm+(h%4)*8,32);memcpy(pd+h*8,pd+(h%4)*8,32);
            memcpy(po+h*8*256,po+(h%4)*8*256,8*256*4);
        }
        attn_merge_gpu(pm,pd,po,qg,rout);QA_READ(aout,db[d].aout,4*1024);
        QA_CMP("merge_gate",aout,rout,1024,4,1);
        int16_t aq[1024],eq[1024];int32_t a32[32],e32[32],a16[64],e16[64];float as[32],es[32];
        quant_act_gpu32(aout,1024,eq,e32,e16,es);
        size_t off=(size_t)dw[d][47].attn_head_first*256;
        QA_READ(aq,db[d].oq+off*2,sizeof aq);QA_READ(a32,db[d].os32+off/32*4,sizeof a32);
        QA_READ(a16,db[d].os16+off/16*4,sizeof a16);QA_READ(as,db[d].oas+off/32*4,sizeof as);
        QA_CMP("activation_codes",aq,eq,1024,2,0);QA_CMP("activation_sum32",a32,e32,32,4,0);
        QA_CMP("activation_sum16",a16,e16,64,4,0);QA_CMP("activation_scales",as,es,32,4,1);
#undef QA_READ
#undef QA_CMP
    }
cleanup:
    free(kq);free(vq);free(rawq);free(ks);free(vs);free(raws);free(sel);
    free(kc);free(vc);free(rk);free(rv);free(flat);free(rflat);free(po);free(rpo);
    CK(cuCtxSetCurrent(ctx[0]));
    fprintf(out,"qsaaudit_end status=%s components=%u failures=%u\n",failures?"FAIL":"PASS",components,failures);fflush(out);
}
