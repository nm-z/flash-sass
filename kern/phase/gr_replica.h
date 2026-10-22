/* Materialize the complete GR matrices in each die's VRAM. Holder arenas stay
 * immutable. All copied rows retain their original BF16 representation/order. */
static CUdeviceptr replicate_balanced_up(int d,int layer,const char *name) {
    CUdeviceptr target=dalloc(2ull*10240*320);
    for(int e=0;e<nd;e++) {
        split_t part;execution_split(&part,e,layer<0?0:layer);
        const hold_tensor *t=hold_find(&ar[e],name);
        unsigned n=(unsigned)part.cbn*32,first=(unsigned)part.cb0*32;
        if(!t || t->type!=30 || t->K!=320 || t->rows!=(int)(4*n)) {
            fprintf(stderr,"engine: incompatible balanced GR-up source %s die=%d\n",name,e);longjmp(init_jb,1);
        }
        for(unsigned branch=0;branch<4;branch++) {
            CUdeviceptr dst=target+640ull*(branch*2560+first);
            CUdeviceptr src=ar[e].base+t->off+640ull*branch*n;
            if(e==d)CK(cuMemcpyDtoD(dst,src,640ull*n));
            else CK(cuMemcpyPeer(dst,ctx[d],src,ctx[e],640ull*n));
        }
    }
    return target;
}
static CUdeviceptr replicate_gr_matrix(int d,int bb,int up) {
    const char *suffix=up?"hc_ffn_up.weight":"hc_ffn_dj.weight";
    unsigned columns=up?320:10240,rows=up?10240:324;
    CUdeviceptr target=dalloc((size_t)rows*columns*2);
    for(int e=0;e<nd;e++) {
        split_t part;split_of(&part,e,nd,bb);
        const hold_tensor *t=hold_find(&ar[e],tn(bb,suffix));
        unsigned n=up?32u*part.cbn:(unsigned)part.djn;
        unsigned first=up?32u*part.cb0:(unsigned)part.dj0;
        if(!t || t->type!=30 || t->K!=(int)columns || t->rows!=(int)(up?4*n:n)) {
            fprintf(stderr,"engine: incompatible GR replica source die=%d layer=%d tensor=%s\n",e,bb,suffix);longjmp(init_jb,1);
        }
        for(unsigned branch=0;branch<(up?4u:1u);branch++) {
            CUdeviceptr dst=target+2ull*columns*(first+branch*(up?2560u:0u));
            CUdeviceptr src=ar[e].base+t->off+2ull*columns*branch*n;
            size_t bytes=2ull*columns*n;
            if(e==d)CK(cuMemcpyDtoD(dst,src,bytes));
            else CK(cuMemcpyPeer(dst,ctx[d],src,ctx[e],bytes));
        }
    }
    return target;
}
