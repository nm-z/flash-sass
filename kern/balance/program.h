/* Included after the ordinary descriptor constructors. Balanced expert IDs
 * remain global; only the matrix/output row ownership changes. */
static void balance_row_descriptor(phase_t *p,unsigned die,unsigned rows,
                                   CUdeviceptr targets,unsigned global_rows,
                                   unsigned ranks,unsigned base,unsigned layer) {
    p->w[1]=rows/FLASH_BALANCE_EXPERT_GRAIN;
    p->w[30]|=PHASE_BALANCED_ROW8;
    /* Ownership remains eight-row-striped. Pack four stripes into one CTA
     * for K640 down projections so four warps compute instead of one. */
    if(global_rows==2560 && ranks==1) {
        p->w[1]=(rows+31)/32;p->w[30]|=PHASE_BALANCED_TILE32;
    }
    if(targets)p->w[30]|=PHASE_BALANCED_PUBLISH;
    ph_ptr(p,0x188,targets);             /* words20..21: six destinations */
    ph_u32(p,0x190,global_rows);         /* word22: destination rows/rank */
    ph_u32(p,0x194,ranks);               /* word23: number of ranked vectors */
    ph_u32(p,0x198,die);                 /* word24: logical, not physical die */
    ph_u32(p,0x19c,base);                /* word25: destination float offset */
    ph_u32(p,0x1ac,layer);               /* word29: token-local layer epoch */
}
static void balance_group(prog_t *p,int first,unsigned width) {
    /* Native independent-descriptor group, not additional dispatches. Each
     * member publishes its payload; the footer publishes flags once, after
     * all members. The next operation acquires all six producers. */
    p->p[first].w[30]|=8u|32u|(width<<8)|1u;
    p->pending_wait=1;p->xj++;
}
static void balance_gate_descriptor(phase_t *a,unsigned die,unsigned rows,CUdeviceptr routes,
                                    unsigned ranks,unsigned base,unsigned layer) {
    balance_row_descriptor(a,die,rows,routes,640,ranks,base,layer);
    a->w[30]|=PHASE_BALANCED_GATE_ROUTE;
}
static void prog_balanced_moe(prog_t *p,const split_t *s,const dn_weights_t *w,
                             const dn_bufs_t *b,CUdeviceptr z,CUdeviceptr sin,
                             CUdeviceptr inj) {
    unsigned gate_rows=balance_row_count(640,(unsigned)s->d,(unsigned)s->n,8);
    unsigned down_rows=balance_row_count(2560,(unsigned)s->d,(unsigned)s->n,8);
    ph_bf16(p,OP_BF16K2560,w->ffn_gate_inp_x,z,b->r,513,0);
    ph_router(p,b->r,b->r+512*4,b->idx,b->w,b->s);
    ph_moesel(p,b->idx,b->w,w->emap,b->lidx,b->lw,b->ne,b->part);
    int first=p->n;
    ph_q2(p,OP_Q2MOEGN,w->ffn_gate_exps,b->zq,b->zs32,b->balance_gate_local,
          gate_rows,b->lidx,b->zas,0,b->ne);
    balance_gate_descriptor(p->p+p->n-1,(unsigned)s->d,gate_rows,b->balance_gate_targets,10,0,w->balanced_layer);
    ph_q2(p,OP_Q2MOEGN,w->ffn_up_exps,b->zq,b->zs32,b->balance_up_local,
          gate_rows,b->lidx,b->zas,0,b->ne);
    balance_gate_descriptor(p->p+p->n-1,(unsigned)s->d,gate_rows,b->balance_up_targets,10,0,w->balanced_layer);
    ph_gemv_any(p,w->ffn_gate_shexp_type,2560,w->ffn_gate_shexp,b->zq,b->zs32,b->zs16,
                b->balance_gate_local+40u*gate_rows,gate_rows,b->zas,0,0);
    balance_gate_descriptor(p->p+p->n-1,(unsigned)s->d,gate_rows,b->balance_gate_targets+10u*(gate_rows/8)*8,1,6400,w->balanced_layer);
    ph_gemv_any(p,w->ffn_up_shexp_type,2560,w->ffn_up_shexp,b->zq,b->zs32,b->zs16,
                b->balance_up_local+40u*gate_rows,gate_rows,b->zas,0,0);
    balance_gate_descriptor(p->p+p->n-1,(unsigned)s->d,gate_rows,b->balance_up_targets+10u*(gate_rows/8)*8,1,6400,w->balanced_layer);
    balance_group(p,first,4);
    first=p->n;
    ph_q2(p,OP_Q2MOEDN,w->ffn_down_exps,b->balance_down_packed,b->balance_down_packed+64,b->balance_down_local,
          down_rows,b->lidx,b->balance_down_packed+76,b->lw,b->ne);
    phase_t *a=p->p+p->n-1;
    balance_row_descriptor(a,(unsigned)s->d,down_rows,0,2560,1,0,w->balanced_layer);
    a->w[30]|=PHASE_BALANCED_PREPARE;ph_ptr(a,0x1a0,b->balance_prepare);
    ph_ptr(a,0x188,b->balance_owner_bounds);
    unsigned shared_offset=(w->balanced_layer%(unsigned)s->n)*down_rows*4u;
    ph_gemv_any(p,w->ffn_down_shexp_type,640,w->ffn_down_shexp,
                b->balance_down_packed+25600,b->balance_down_packed+25664,b->balance_down_packed+25668,b->balance_down_local+shared_offset,
                down_rows,b->balance_down_packed+25676,1,b->s);
    balance_row_descriptor(p->p+p->n-1,(unsigned)s->d,down_rows,b->balance_down_targets,2560,1,0,w->balanced_layer);
    a=p->p+p->n-1;a->w[30]|=PHASE_BALANCED_REDUCE;ph_ptr(a,0x1a0,b->balance_down_local);
    balance_group(p,first,2);
    p->p[first].w[30]|=0x80000000u; /* same-CTA routed -> shared read-after-write */
    a=prog_add(p,OP_HCWSUM,16);
    ph_ptr(a,0x140,inj);ph_ptr(a,0x148,b->part);ph_ptr(a,0x150,b->S);
    ph_u32(a,0x158,2560);ph_u32(a,0x15c,10240);ph_u32(a,0x164,1);
    ph_ptr(a,0x168,b->out);ph_ptr(a,0x170,sin);
    a->w[30]|=PHASE_BALANCED_ASSEMBLED;
}
