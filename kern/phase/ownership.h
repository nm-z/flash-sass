/* Holder placement remains immutable. Runtime attention ownership may differ;
 * all resident tensor sources continue to use the holder's original ATTN_MAP. */
static int phase_attention_balanced(void) {
    const char *v=getenv("FLASH_ATTN_BALANCE");
    return v?atoi(v)!=0:producer_quant;
}
static int runtime_attn_owner(int bb) {
    if(phase_attention_balanced() && bb>=3 && bb<NB)return ((bb-3)/4)%nd;
    return attn_die_of(bb);
}
static CUdeviceptr runtime_attn_tensor(int d,int bb,const char *suffix) {
    if(row_balance)return dptr(d,bb,suffix);
    int source=attn_die_of(bb);
    if(d!=runtime_attn_owner(bb) || d==source)return dptr(d,bb,suffix);
    const hold_tensor *t=hold_find(&ar[source],tn(bb,suffix));
    if(!t || !t->bytes) {
        fprintf(stderr,"engine: missing source attention tensor %s on holder %d\n",tn(bb,suffix),source);
        longjmp(init_jb,1);
    }
    CUdeviceptr target=dalloc(t->bytes);
    CK(cuMemcpyPeer(target,ctx[d],ar[source].base+t->off,ctx[source],t->bytes));
    return target;
}
