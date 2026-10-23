/* Every logical die executes a complete local QSA slice. Query heads are
 * disjoint; indexer projection rows are assembled before local selection. */
static void prog_balanced_attention(prog_t *p,const split_t *s,const xchg_t *x,
                                    const dn_weights_t *w,const dn_bufs_t *b,
                                    unsigned t,float eps) {
    CUdeviceptr inj=b->t+1280;
    CUdeviceptr z=ph_hc_read(p,s,x,w->hc_attn_norm,w->hc_attn_dj,w->hc_attn_up,b,eps,0,0,0);
    int first=p->n;
    ph_gemv_any(p,w->attn_q_type,2560,w->attn_q,b->zq,b->zs32,b->zs16,b->qg,w->attn_head_count*512,b->zas,0,0);
    if(w->unique_rows) {
        unsigned kv_rows=balance_row_count(512,(unsigned)s->d,(unsigned)s->n,8);
        ph_gemv_any(p,w->attn_k_type,2560,w->attn_k,b->zq,b->zs32,b->zs16,b->unique_k_local,kv_rows,b->zas,0,0);
        balance_row_descriptor(p->p+p->n-1,(unsigned)s->d,kv_rows,b->unique_k_targets,512,1,0,w->balanced_layer);
        ph_gemv_any(p,w->attn_v_type,2560,w->attn_v,b->zq,b->zs32,b->zs16,b->unique_v_local,kv_rows,b->zas,0,0);
        balance_row_descriptor(p->p+p->n-1,(unsigned)s->d,kv_rows,b->unique_v_targets,512,1,0,w->balanced_layer);
    } else {
        ph_gemv_any(p,w->attn_k_type,2560,w->attn_k,b->zq,b->zs32,b->zs16,b->k,256,b->zas,0,0);
        ph_gemv_any(p,w->attn_v_type,2560,w->attn_v,b->zq,b->zs32,b->zs16,b->v,256,b->zas,0,0);
    }
    unsigned rows=balance_row_count(640,(unsigned)s->d,(unsigned)s->n,8);
    ph_bf16(p,OP_BF16K2560,w->indexer_x,z,b->balance_index_local,rows,0);
    balance_row_descriptor(p->p+p->n-1,(unsigned)s->d,rows,b->balance_index_targets,640,1,0,w->balanced_layer);
    balance_group(p,first,4);
    prog_attn_qsa_projected(p,w,b,t,eps,b->qg);
    ph_gemv_any(p,w->attn_output_type,6144,w->attn_output,b->oq,b->os32,b->os16,b->ly,(unsigned)s->yn,b->oas,0,0);
    CUdeviceptr y=ph_xchg(p,x,s->d,s->n,b->ly,4u*(unsigned)s->yn,4u*(unsigned)s->y0,10240);
    if(getenv("PROG_NORMSW")){ph_hc(p,OP_HCWRITE,inj,y,b->S,2560,0,10);ph_moe_half(p,s,x,w,b,eps,0);}
    else ph_moe_half(p,s,x,w,b,eps,y);
}
