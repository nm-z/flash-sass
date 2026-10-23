/* Included by engine.c. Same target weights, GPU-resident quantized DFlash2,
 * four target prefix states and a block-level greedy acceptance boundary. */
static void dflash_quant_error(int result,const char *error) {
    if(!result)return;
    fprintf(stderr,"engine: %s\n",error);
    if(in_init)longjmp(init_jb,1);
    longjmp(run_jb,1);
}
static CUdeviceptr dflash_bank_table(int d,const CUdeviceptr banks[4]) {
    CUdeviceptr table[5]={dflash_state_selector[d],banks[0],banks[1],banks[2],banks[3]};
    CUdeviceptr p=dalloc(sizeof table);CK(cuMemcpyHtoD(p,table,sizeof table));return p;
}
static void dflash_prefix_bind(dn_bufs_t *b,int d,int layer,unsigned p) {
    unsigned previous=p?p-1:3;
    b->prefix_states=1;
    b->dn_state=dflash_dn_bank[previous][d][layer];b->dn_checkpoint=dflash_dn_bank[p][d][layer];
    b->conv_state=dflash_conv_bank[previous][d][layer];b->conv_checkpoint=dflash_conv_bank[p][d][layer];
    b->ring=dflash_ring_bank[previous][d][layer];b->ring_checkpoint=dflash_ring_bank[p][d][layer];
    b->ple_state=dflash_ple_bank[previous][d];b->ple_checkpoint=dflash_ple_bank[p][d];
    if(!p) {
        b->dn_input_table=dflash_dn_table[d][layer];b->conv_input_table=dflash_conv_table[d][layer];
        b->ring_input_table=dflash_ring_table[d][layer];b->ple_input_table=dflash_ple_table[d];
    }
}
static void dflash_prefix_select(unsigned bank) {
    dflash_state_bank=bank;
    for(int d=0;d<nd;d++) {
        CK(cuCtxSetCurrent(ctx[d]));CK(cuMemsetD32(dflash_state_selector[d],bank,1));
        dst_ple[d]=dflash_ple_bank[bank][d];
        for(int l=0;l<nlayers_run;l++) {
            dst_dn[d][l]=dflash_dn_bank[bank][d][l];dst_conv[d][l]=dflash_conv_bank[bank][d][l];
            dst_ring[d][l]=dflash_ring_bank[bank][d][l];
        }
    }
}
static void dflash_quant_init(void) {
    if(!dflash_quant)return;
    for(int d=0;d<nd;d++) {
        CK(cuCtxSetCurrent(ctx[d]));CUfunction marker;
        CK(cuModuleGetFunction(&marker,mod[d],"flash_dflash_features_abi_v1"));
        CK(cuModuleGetFunction(&marker,exact_mod[d],"flash_dflash_prefix_abi_v1"));
        CK(cuModuleGetFunction(&marker,exact_mod[d],"flash_dflash_features_abi_v1"));
        dflash_state_selector[d]=dalloc(4);
        CUdeviceptr banks[4];
        for(unsigned p=0;p<4;p++)banks[p]=dflash_ple_bank[p][d]=p==3?dst_ple[d]:dalloc(4u*10u*10240u);
        dflash_ple_table[d]=dflash_bank_table(d,banks);
        for(int l=0;l<nlayers_run;l++) {
            if(BLK_IS_ATTN(l)) {
                if(dst_ring[d][l]) {
                    for(unsigned p=0;p<4;p++)banks[p]=dflash_ring_bank[p][d][l]=p==3?dst_ring[d][l]:dalloc(2048);
                    dflash_ring_table[d][l]=dflash_bank_table(d,banks);
                }
            } else {
                for(unsigned p=0;p<4;p++)banks[p]=dflash_dn_bank[p][d][l]=p==3?dst_dn[d][l]:dalloc(dn_bytes[d][l]);
                dflash_dn_table[d][l]=dflash_bank_table(d,banks);
                for(unsigned p=0;p<4;p++)banks[p]=dflash_conv_bank[p][d][l]=p==3?dst_conv[d][l]:dalloc(conv_bytes[d][l]);
                dflash_conv_table[d][l]=dflash_bank_table(d,banks);
            }
        }
    }
    dflash_prefix_select(3);
    CK(cuCtxSetCurrent(ctx[dflash_rank]));
    dflash_feature_ring=dalloc((size_t)DFLASH_FEATURE_WINDOW*DFLASH_FEATURE_STRIDE*4);
    dflash_target_binding target[6]={{0}};
    for(int d=0;d<nd;d++) {
        split_t head;execution_split(&head,d,0);
        const hold_tensor *emb=hold_find(&ar[d],"token_embd.weight");
        dn_weights_t *w=&dw[d][0];
        target[d]=(dflash_target_binding){.context=ctx[d],.output_weights=w->output,
            .output_rows=(unsigned)head.on,.output_first=(unsigned)head.o0,.output_stride=1760,
            .embedding=w->token_embd,.embedding_bytes=emb?(size_t)emb->bytes:0,
            .embedding_first_group=row_balance?w->embedding_first/32:0,
            .embedding_groups=row_balance?w->embedding_columns/32:80,
            .embedding_stride=row_balance?w->embedding_stride:1100,.embedding_raw_q3=!row_balance};
    }
    const char *model=getenv("FLASH_DFLASH_GGUF"),*module=getenv("FLASH_DFLASH_DEVICE_CUBIN");
    if(!model)model="/home/nate/models-cache/flash-sass/dflash2-q4km/Qwen3.8-Flash-Next-DFlash2-Q4_K_M.gguf";
    if(!module)module="dflash/device.cubin";
    char error[512]={0};
    dflash_quant_error(dflash_resident_open(&dflash_quant_resident,model,module,target,dflash_rank,
                                         dflash_feature_ring,error,sizeof error),error);
    dflash_resident_target_layers(dflash_quant_resident,dflash_target_layer);
}
static void dflash_quant_fini(void) {
    dflash_resident_close(dflash_quant_resident);dflash_quant_resident=NULL;
    dflash_quant=0;dflash_feature_ring=0;dflash_state_bank=3;
    memset(dflash_state_selector,0,sizeof dflash_state_selector);
    memset(dflash_dn_bank,0,sizeof dflash_dn_bank);memset(dflash_conv_bank,0,sizeof dflash_conv_bank);
    memset(dflash_ring_bank,0,sizeof dflash_ring_bank);memset(dflash_ple_bank,0,sizeof dflash_ple_bank);
    memset(dflash_dn_table,0,sizeof dflash_dn_table);memset(dflash_conv_table,0,sizeof dflash_conv_table);
    memset(dflash_ring_table,0,sizeof dflash_ring_table);memset(dflash_ple_table,0,sizeof dflash_ple_table);
    memset(dflash_program_source,0,sizeof dflash_program_source);memset(dflash_program_cache,0,sizeof dflash_program_cache);
}
