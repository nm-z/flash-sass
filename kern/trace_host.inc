/* Bounded snapshots after launches complete. GPU epochs are local to each die. */
static int trace_dump(FILE *out, int d) {
    if (d < 0 || d >= nd || !trace_n || !trace_valid || !prof_nph[d]) {
        fprintf(out, "error: trace <die> requires FLASH_TRACE=1..256 and a completed run\n"); return -1;
    }
    unsigned nph = (unsigned)prof_nph[d], nt = trace_valid;
    size_t words = (size_t)nt * nph * 8;
    uint64_t *ts = malloc(words * 8);
    if (!ts) { fprintf(out, "error: trace allocation\n"); return -1; }
    CK(cuCtxSetCurrent(ctx[d])); CK(cuMemcpyDtoH(ts, dTRACE[d], words * 8));
    for (size_t i = 0; i < (size_t)nt * nph; i++) {
        const uint64_t *z = ts + i * 8;
        if (!z[0] || z[0] > z[1] || z[1] > z[2] || z[2] > z[3] || z[3] > z[4] ||
            (z[5] && (z[5] < z[2] || z[6] < z[5] || z[6] > z[3])) ||
            (i % nph == nph - 1 && z[7] < z[4])) {
            fprintf(out, "error: incomplete/nonmonotonic trace at record %zu; use DIE_TRACE=1\n", i); free(ts); return -1;
        }
    }
    fprintf(out, "trace_begin request=%llu die=%d first=%u count=%u nph=%u ctas=1 nblk=%u\n", (unsigned long long)trace_request, d, trace_first, nt, nph, nblk_qsa);
    fprintf(out, "position,phase,layer,opcode,units,flags,selected_keys,start_ns,wait_start_ns,wait_end_ns,body_end_ns,end_ns,ple_wait_start_ns,ple_wait_end_ns,publication_ns\n");
    for (unsigned t = 0; t < nt; t++) for (unsigned p = 0; p < nph; p++) {
        const uint64_t *z = ts + ((size_t)t * nph + p) * 8;
        fprintf(out, "%u,%u,%d,%s,%u,%u,%u,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n",
                trace_first+t, p, prof_layer[d][p], die_opname[prof_op[d][p]], prof_units[d][p], prof_flags[d][p],
                qsa_nkeys(trace_first+t, nblk_qsa), (unsigned long long)z[0], (unsigned long long)z[1],
                (unsigned long long)z[2], (unsigned long long)z[3], (unsigned long long)z[4],
                (unsigned long long)z[5], (unsigned long long)z[6], (unsigned long long)z[7]);
    }
    fprintf(out, "trace_end\n"); free(ts); return 0;
}
static int host_trace_dump(FILE *out) {
    if (!trace_n || !trace_valid) { fprintf(out, "error: no completed host trace\n"); return -1; }
    fprintf(out, "hosttrace_begin request=%llu first=%u count=%u clock=CLOCK_MONOTONIC\n", (unsigned long long)trace_request, trace_first, trace_valid);
    fprintf(out, "position,observed_ns,callback_start_ns,callback_end_ns,ple_staging_start_ns,ple_staging_end_ns,ple_ready_ns,ple_bytes,ple_short_reads\n");
    for (unsigned i = 0; i < trace_valid; i++) {
        const host_trace *t = trace_host+i;
        fprintf(out, "%u,%llu,%llu,%llu,%llu,%llu,%llu,%u,%u\n", trace_first+i,
                (unsigned long long)t->observed, (unsigned long long)t->callback_start, (unsigned long long)t->callback_end,
                (unsigned long long)t->staging_start, (unsigned long long)t->staging_end, (unsigned long long)t->ready, t->bytes, t->short_reads);
    }
    fprintf(out, "hosttrace_end\n"); return 0;
}
