#include "mtp/model.h"

static void q8(prog_t *p,unsigned k,CUdeviceptr w,CUdeviceptr x,CUdeviceptr y,
               unsigned rows,unsigned stride,CUdeviceptr idx,unsigned slot,CUdeviceptr ne) {
    /* Shapes are fixed by this model builder; all are supported by mtp_ph_q8. */
    (void)mtp_ph_q8(p,k,w,x,y,rows,stride,idx,slot,ne);
}
static CUdeviceptr hc_read(prog_t *p,const split_t *s,const xchg_t *x,const dn_bufs_t *b,
                           CUdeviceptr norm,CUdeviceptr down,CUdeviceptr up,int head,float eps) {
    unsigned first=head?(unsigned)s->hd0:(unsigned)s->dj0;
    unsigned rows=head?(unsigned)s->hdn:(unsigned)s->djn;
    unsigned total=head?320:324,cols=32u*(unsigned)s->cbn;
    ph_rms(p,OP_RMS2560,b->S,norm,b->xh,eps,4);
    q8(p,10240,down,b->xh,b->ldown,rows,0,0,0,0);
    CUdeviceptr gathered=ph_xchg(p,x,s->d,s->n,b->ldown,4*rows,4*first,4*total);
    ph_hc(p,OP_HCSILU,gathered,b->t,0,total,320,2);
    q8(p,320,up,b->t,b->lup,4*cols,0,0,0,0);
    return ph_hcx(p,x,OP_HCMIXX,b->lup,b->xh+128u*(unsigned)s->cb0,
                   cols,4*cols,(cols+255)/256,128u*(unsigned)s->cb0);
}

int mtp_program(prog_t *p,const split_t *s,const xchg_t *x,const mtp_weights_t *mw,
                const mtp_buffers_t *mb,CUdeviceptr hidden,unsigned t,unsigned nprompt,float eps,int include_head) {
    if(!p || !s || !x || !mw || !mb) return -1;
    const dn_weights_t *w=&mw->block;const dn_bufs_t *b=&mb->base;
    if(!hidden || s->n!=6 || s->d<0 || s->d>=6 ||
       !b->qsa || !mb->embedding || !mb->eh_concat || !mb->expert_rows ||
       !mw->eh_proj || !mw->enorm || !mw->hnorm || !w->output || !w->obases) return -1;
    /* Four normalized residual lanes, each paired with the same normalized
     * token embedding. EH projection is replicated (13.9 MB per die). */
    ph_rms(p,OP_RMS2560,hidden,mw->hnorm,b->xh,eps,4);
    ph_rms(p,OP_RMS2560,mb->embedding,mw->enorm,b->qn,eps,1);
    mtp_ph_eh_concat(p,b->qn,b->xh,mb->eh_concat);
    for(unsigned lane=0;lane<4;lane++) q8(p,5120,mw->eh_proj,mb->eh_concat,b->S,2560,5120*4,0,lane,0);

    CUdeviceptr z=hc_read(p,s,x,b,w->hc_attn_norm,w->hc_attn_dj,w->hc_attn_up,0,eps);
    if(b->kq) {
        q8(p,2560,w->attn_q,z,b->qg,12288,0,0,0,0);
        q8(p,2560,w->attn_k,z,b->k,512,0,0,0,0);
        q8(p,2560,w->attn_v,z,b->v,512,0,0,0,0);
        ph_bf16(p,OP_BF16K2560,w->indexer_x,z,b->iqk,640,0);
        prog_attn_qsa_projected(p,w,b,t,eps,b->qg);
    }
    CUdeviceptr attended=ph_xchg(p,x,s->d,s->n,b->aout,b->kq?4*6144:0,0,4*6144);
    q8(p,6144,w->attn_output,attended,b->ly,(unsigned)s->yn,0,0,0,0);
    CUdeviceptr y=ph_xchg(p,x,s->d,s->n,b->ly,4u*(unsigned)s->yn,4u*(unsigned)s->y0,4*2560);
    ph_hc(p,OP_HCWRITE,b->t+4*320,y,b->S,2560,0,10);

    z=hc_read(p,s,x,b,w->hc_ffn_norm,w->hc_ffn_dj,w->hc_ffn_up,0,eps);
    ph_bf16(p,OP_BF16K2560,w->ffn_gate_inp_x,z,b->r,513,0);
    ph_router(p,b->r,b->r+4*512,b->idx,b->w,b->s);
    ph_moesel(p,b->idx,b->w,w->emap,b->lidx,b->lw,b->ne,b->part);
    for(unsigned slot=0;slot<10;slot++) {
        q8(p,2560,w->ffn_gate_exps,z,b->g,640,0,b->lidx,slot,b->ne);
        q8(p,2560,w->ffn_up_exps,z,b->gu,640,0,b->lidx,slot,b->ne);
        mtp_ph_swiglu(p,b->g,b->gu,b->g,slot,b->ne);
        q8(p,640,w->ffn_down_exps,b->g,mb->expert_rows,2560,640*4,b->lidx,slot,b->ne);
    }
    CUdeviceptr shared=0;
    if(s->shexp) {
        CUdeviceptr gate=b->g+4*640*10,up=b->gu+4*640*10;
        shared=mb->expert_rows+4*2560*10;
        q8(p,2560,w->ffn_gate_shexp,z,gate,640,0,0,0,0);
        q8(p,2560,w->ffn_up_shexp,z,up,640,0,0,0,0);
        mtp_ph_swiglu(p,gate,up,gate,0,0);
        q8(p,640,w->ffn_down_shexp,gate,shared,2560,0,0,0,0);
    }
    mtp_ph_weighted_sum(p,mb->expert_rows,b->lw,b->ne,shared,shared?b->s:0,b->part);
    CUdeviceptr parts=ph_xchg(p,x,s->d,s->n,b->part,4*2560,4u*2560u*(unsigned)s->d,4u*2560u*(unsigned)s->n);
    ph_sumn(p,parts,4*2560,b->out,2560,(unsigned)s->n);
    ph_hc(p,OP_HCWRITE,b->t+4*320,b->out,b->S,2560,0,10);

    if(include_head) {
        z=hc_read(p,s,x,b,mw->head_norm,mw->head_down,mw->head_up,1,eps);
        prog_logits(p,s,x,w,b,z,t,nprompt);
    }
    prog_token_end(p,x,s->d,s->n,b->S);
    return 0;
}
