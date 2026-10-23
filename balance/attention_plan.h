/* Control descriptors own one physical phase; following descriptors are the
 * real child operations/parameters, not omitted model work. */
static void balanced_attention_control(prog_t *p,int index,unsigned opcode,
                                       unsigned width,CUdeviceptr ready,
                                       unsigned epoch_code,unsigned flags,int *layers) {
    int old=p->n,layer=layers?layers[index]:-1;
    prog_add(p,0,0);
    memmove(p->p+index+1,p->p+index,(size_t)(old-index)*sizeof *p->p);
    if(layers){memmove(layers+index+1,layers+index,(size_t)(old-index)*sizeof *layers);layers[index]=layer;}
    phase_t *a=p->p+index;memset(a,0,sizeof *a);
    a->w[0]=opcode;a->w[1]=16;a->w[30]=24u|(width<<8)|flags;
    ph_ptr(a,0x140,ready);ph_u32(a,0x148,epoch_code);
}
static unsigned balanced_attention_fuse(prog_t *p,const dn_bufs_t *b,int *layers) {
    unsigned attention=0;
    for(int i=0;i+6<p->n;i++) {
        phase_t *a=p->p+i;
        if(a[0].w[0]==OP_ATTNQKQ && a[0].w[1]==5 && a[1].w[0]==OP_IDXQK &&
           a[2].w[0]==103 && a[3].w[0]==102 && a[4].w[0]==OP_KVGATHER &&
           a[5].w[0]==OP_ATTNPARTN && a[5].w[1]==16 &&
           a[6].w[0]==101 && a[6].w[1]==4) {
            unsigned layer=4*attention+3,flags=(a[0].w[30]&2u)|(a[6].w[30]&1u);
            /* Selection has one producer. Its completion releases gather
             * directly; a separate whole-grid dispatch is not necessary. */
            balanced_attention_control(p,i,OP_BALANCED_SELECT,8,b->balance_attn_ready,layer*4,flags,layers);
            attention++;i+=7;
        }
    }
    if(attention!=(unsigned)nlayers_run/4) {
        fprintf(stderr,"balanced attention coverage: attention=%u layers=%d\n",attention,nlayers_run);
        if(in_init)longjmp(init_jb,1);
        exit(1);
    }
    return attention;
}
