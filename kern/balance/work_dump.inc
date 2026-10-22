/* Read the program actually retained after full-model inference. This is
 * evidence for a work audit, not a timing gate or an assertion of balance. */
static void balance_work_dump(FILE *out) {
    unsigned records=0,failures=0;
    if(!row_balance || nd!=6 || nlayers_run!=48 || two_position ||
       !last_timing_valid || !last_timing.counts_final || last_timing.error_code || last_positions<=0) {
        fprintf(out,"workdump_end dies=0 records=0 failures=1\n");fflush(out);return;
    }
    for(int d=0;d<nd;d++) {
        unsigned n=prof_desc[d];
        if(!n || n>program_capacity || prof_nph[d]<=0 || prof_nph[d]>=500){failures++;continue;}
        phase_t *p=malloc((size_t)n*sizeof *p);
        if(!p){failures++;continue;}
        CK(cuCtxSetCurrent(ctx[d]));CK(cuMemcpyDtoH(p,dPROG[d],(size_t)n*sizeof *p));
        unsigned i=0;
        for(int phase=0;phase<prof_nph[d];phase++) {
            unsigned end=phase+1<prof_nph[d]?prof_index[d][phase+1]:n;
            if(i!=prof_index[d][phase] || end<=i || end>n){failures++;break;}
            for(unsigned j=i;j<end;j++) {
                fprintf(out,"workdesc die=%d phase=%d member=%u index=%u name=%s words=",
                        d,phase,j-i,j,profile_op_name(p[j].w[0]));
                for(unsigned w=0;w<32;w++)fprintf(out,"%s%08x",w?",":"",p[j].w[w]);
                fputc('\n',out);records++;
            }
            i=end;
        }
        if(i!=n)failures++;
        fprintf(out,"workdie die=%d phases=%d descriptors=%u positions=%d\n",d,prof_nph[d],n,last_positions);
        free(p);
    }
    /* Final-layer scratch is retained. Equal score inputs imply the same
     * data-dependent histogram predicates; selection length alone would not.
     * This deliberately reports the context branch actually exercised. */
    unsigned nv=(unsigned)last_positions/4,selected=nv<nblk_qsa?nv:nblk_qsa;
    uint32_t ids0[10],ids[10],ne,query0[512],query[512];
    uint32_t *scores0=malloc((size_t)(nv?nv:1)*4),*scores=malloc((size_t)(nv?nv:1)*4);
    uint32_t *sel0=malloc((size_t)(selected?selected:1)*4),*sel=malloc((size_t)(selected?selected:1)*4);
    if(!scores0 || !scores || !sel0 || !sel)failures++;
    else for(int d=0;d<nd;d++) {
        unsigned bad=0;CK(cuCtxSetCurrent(ctx[d]));
        CK(cuMemcpyDtoH(&ne,db[d].ne,4));CK(cuMemcpyDtoH(ids,db[d].lidx,sizeof ids));
        bad+=ne!=10;
        for(unsigned i=0;i<10;i++){bad+=ids[i]>=512;for(unsigned j=0;j<i;j++)bad+=ids[i]==ids[j];}
        CK(cuMemcpyDtoH(query,db[d].iq,sizeof query));
        /* Score/select stages bypass unused arrays when all blocks fit. */
        if(nv>nblk_qsa) {
            CK(cuMemcpyDtoH(scores,db[d].sc,(size_t)nv*4));
            CK(cuMemcpyDtoH(sel,db[d].sel,(size_t)selected*4));
        }
        if(!d) {
            memcpy(ids0,ids,sizeof ids);memcpy(query0,query,sizeof query);
            if(nv>nblk_qsa){memcpy(scores0,scores,(size_t)nv*4);memcpy(sel0,sel,(size_t)selected*4);}
        } else {
            bad+=memcmp(ids0,ids,sizeof ids)!=0;bad+=memcmp(query0,query,sizeof query)!=0;
            if(nv>nblk_qsa){bad+=memcmp(scores0,scores,(size_t)nv*4)!=0;bad+=memcmp(sel0,sel,(size_t)selected*4)!=0;}
        }
        fprintf(out,"workdynamic die=%d layer=47 position=%d experts=%u query_values=512 score_values=%u selection_values=%u selection_exercised=%d failures=%u\n",
                d,last_positions-1,ne,nv>nblk_qsa?nv:0,nv>nblk_qsa?selected:0,nv>nblk_qsa,bad);
        failures+=bad;
    }
    free(scores0);free(scores);free(sel0);free(sel);
    fprintf(out,"workdump_end dies=%d records=%u failures=%u\n",nd,records,failures);fflush(out);
}
