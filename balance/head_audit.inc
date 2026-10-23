/* After complete inference, independently scan every retained local logit.
 * Check all96 CTA winners, including those which did not win globally. */
static void balance_head_audit(FILE *out) {
    unsigned failures=0,parts=0,total_rows=0;float winner=-INFINITY;
    uint32_t winner_index=UINT32_MAX,winner_bits=0;
    if(!row_balance || nd!=6 || nlayers_run!=48 || two_position ||
       !last_timing_valid || !last_timing.counts_final || last_timing.error_code || last_positions<=0){
        fprintf(out,"headaudit_end parts=0 rows=0 failures=1 reason=requires_completed_balanced_scalar_full_model\n");fflush(out);return;
    }
    for(int d=0;d<nd;d++) {
        unsigned rows=(unsigned)sp[d][0].on;float *logits=malloc((size_t)rows*4);
        uint32_t actual[16][2],indices[16],counts[16]={0};float maxima[16];
        if(!logits){failures++;fprintf(out,"headaudit error=allocation die=%d\n",d);break;}
        for(unsigned c=0;c<16;c++){indices[c]=UINT32_MAX;maxima[c]=-INFINITY;}
        CK(cuCtxSetCurrent(ctx[d]));CK(cuMemcpyDtoH(logits,position_buf(d,last_position_slot)->hlo,(size_t)rows*4));
        CK(cuMemcpyDtoH(actual,position_buf(d,last_position_slot)->apart,sizeof actual));
        unsigned nonfinite=0;
        for(unsigned r=0;r<rows;r++) {
            /* Q5_K2560 produces16 adjacent rows per CTA iteration, with
             * a256-row stride between iterations. */
            unsigned c=(r/16)%16;counts[c]++;
            if(!isfinite(logits[r])){nonfinite++;continue;}
            if(logits[r]>maxima[c]){maxima[c]=logits[r];indices[c]=r;}
        }
        if(nonfinite){failures++;fprintf(out,"headaudit die=%d nonfinite_logits=%u\n",d,nonfinite);}
        uint32_t die_index=UINT32_MAX,die_bits=0;float die_max=-INFINITY;
        for(unsigned c=0;c<16;c++) {
            uint32_t bits;memcpy(&bits,maxima+c,4);
            int valid=counts[c]!=0 && indices[c]!=UINT32_MAX && actual[c][0]==indices[c] && actual[c][1]==bits;
            fprintf(out,"headpart die=%d cta=%u rows=%u actual_index=%u expected_index=%u actual_bits=%08x expected_bits=%08x valid=%d\n",
                d,c,counts[c],actual[c][0],indices[c],actual[c][1],bits,valid);
            failures+=!valid;parts++;total_rows+=counts[c];
            if(indices[c]!=UINT32_MAX && (maxima[c]>die_max || (maxima[c]==die_max && indices[c]<die_index))){
                die_max=maxima[c];die_index=indices[c];die_bits=bits;
            }
            uint32_t global=indices[c]+(uint32_t)sp[d][0].o0;
            if(indices[c]!=UINT32_MAX && (maxima[c]>winner || (maxima[c]==winner && global<winner_index))){
                winner=maxima[c];winner_index=global;winner_bits=bits;
            }
        }
        uint32_t die_actual[2];CK(cuMemcpyDtoH(die_actual,position_buf(d,last_position_slot)->amo,sizeof die_actual));
        int die_valid=die_index!=UINT32_MAX && die_actual[0]==die_index && die_actual[1]==die_bits;
        failures+=!die_valid;
        fprintf(out,"headdie die=%d actual_index=%u expected_index=%u actual_bits=%08x expected_bits=%08x valid=%d\n",
            d,die_actual[0],die_index,die_actual[1],die_bits,die_valid);
        free(logits);
    }
    for(int d=0;d<nd;d++) {
        uint32_t actual[2],token;CK(cuCtxSetCurrent(ctx[d]));
        CK(cuMemcpyDtoH(actual,position_buf(d,last_position_slot)->tokout,sizeof actual));
        CK(cuMemcpyDtoH(&token,position_buf(d,last_position_slot)->toks+4u*(size_t)(last_positions-1),4));
        int valid=winner_index!=UINT32_MAX && actual[0]==winner_index && actual[1]==winner_bits && token==winner_index;
        failures+=!valid;
        fprintf(out,"headwinner die=%d actual_index=%u expected_index=%u actual_bits=%08x expected_bits=%08x token=%u valid=%d\n",
            d,actual[0],winner_index,actual[1],winner_bits,token,valid);
    }
    CK(cuCtxSetCurrent(ctx[0]));
    if(parts!=96 || total_rows!=248320)failures++;
    fprintf(out,"headaudit_end parts=%u rows=%u failures=%u\n",parts,total_rows,failures);fflush(out);
}
