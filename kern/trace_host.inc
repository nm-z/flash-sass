/* Bounded snapshots after launches complete. GPU epochs are local to each die. */
static void trace_status_fields(FILE *out) {
    fprintf(out, " status=%s aborted=%u ended=%u completed=%u requested=%llu",
            trace_status, trace_aborted, trace_ended, trace_completed,
            (unsigned long long)trace_requested);
}
static int trace_dump(FILE *out, int d) {
    if (d < 0 || d >= nd || !trace_n || !trace_valid || !prof_nph[d]) {
        fprintf(out, "error: trace <die> requires FLASH_TRACE=1..256 and a completed run; request=%llu", (unsigned long long)trace_request);
        trace_status_fields(out); fputc('\n', out); return -1;
    }
    unsigned nph = (unsigned)prof_nph[d], nt = trace_valid;
    size_t records = (size_t)nt * nph * trace_ctas;
    size_t words = records * TRACE_WORDS;
    uint64_t *ts = malloc(words * 8);
    if (!ts) { fprintf(out, "error: trace allocation\n"); return -1; }
    CK(cuCtxSetCurrent(ctx[d])); CK(cuMemcpyDtoH(ts, dTRACE[d], words * 8));
    for (size_t i = 0; i < records; i++) {
        const uint64_t *z = ts + i * TRACE_WORDS;
        unsigned p = (unsigned)(i / trace_ctas % nph);
        int send = (prof_flags[d][p] & 1u) != 0;
        if (!z[0] || z[0] > z[1] || z[1] > z[2] || z[2] > z[8] || z[8] > z[3] || z[3] > z[4] ||
            (z[5] && (z[5] < z[8] || z[6] < z[5] || z[6] > z[3])) || (!z[5] && z[6]) ||
            (send && (!z[9] || z[9] < z[3] || z[10] < z[9] || z[10] > z[4])) ||
            (!send && (z[9] || z[10])) || (p == nph - 1 ? z[7] < z[4] : z[7] != 0)) {
            fprintf(out, "error: incomplete/nonmonotonic schema-2 trace at record %zu; match FLASH_TRACE_CTAS and image\n", i); free(ts); return -1;
        }
    }
    fprintf(out, "trace_begin request=%llu die=%d first=%u count=%u nph=%u ctas=%u nblk=%u schema=2", (unsigned long long)trace_request, d, trace_first, nt, nph, trace_ctas, nblk_qsa);
    trace_status_fields(out); fputc('\n', out);
    fprintf(out, "position,phase,layer,opcode,units,flags,selected_keys,start_ns,wait_start_ns,wait_end_ns,body_end_ns,end_ns,ple_wait_start_ns,ple_wait_end_ns,publication_ns,cta,body_start_ns,send_before_ns,send_after_ns,exchange_generation\n");
    for (unsigned t = 0; t < nt; t++) for (unsigned p = 0; p < nph; p++) for (unsigned c = 0; c < trace_ctas; c++) {
        const uint64_t *z = ts + (((size_t)t * nph + p) * trace_ctas + c) * TRACE_WORDS;
        fprintf(out, "%u,%u,%d,%s,%u,%u,%u,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%u,%llu,%llu,%llu,%llu\n",
                trace_first+t, p, prof_layer[d][p], die_opname[prof_op[d][p]], prof_units[d][p], prof_flags[d][p],
                qsa_nkeys(trace_first+t, nblk_qsa), (unsigned long long)z[0], (unsigned long long)z[1],
                (unsigned long long)z[2], (unsigned long long)z[3], (unsigned long long)z[4],
                (unsigned long long)z[5], (unsigned long long)z[6], (unsigned long long)z[7], c,
                (unsigned long long)z[8], (unsigned long long)z[9], (unsigned long long)z[10], (unsigned long long)z[11]);
    }
    fprintf(out, "trace_end\n"); free(ts); return 0;
}
static int host_trace_dump(FILE *out) {
    if (!trace_n || !trace_valid) {
        fprintf(out, "error: no completed host trace; request=%llu", (unsigned long long)trace_request);
        trace_status_fields(out); fputc('\n', out); return -1;
    }
    fprintf(out, "hosttrace_begin request=%llu first=%u count=%u clock=CLOCK_MONOTONIC", (unsigned long long)trace_request, trace_first, trace_valid);
    trace_status_fields(out);
    fprintf(out, " schema=2 abort_start_ns=%llu abort_end_ns=%llu abort_position=%d abort_reason=%s request_poll_calls=%llu request_poll_ns=%llu request_readback_calls=%llu request_readback_ns=%llu\n",
            (unsigned long long)trace_abort_start, (unsigned long long)trace_abort_end, trace_abort_position, trace_abort_reason,
            (unsigned long long)trace_poll_calls, (unsigned long long)trace_poll_ns,
            (unsigned long long)trace_readback_calls, (unsigned long long)trace_readback_ns);
    fprintf(out, "position,observed_ns,callback_start_ns,callback_end_ns,ple_staging_start_ns,ple_staging_end_ns,ple_ready_ns,ple_bytes,ple_short_reads,ple_read_start_ns,ple_read_end_ns,ple_ready_start_ns,poll_start_ns,poll_end_ns,readback_start_ns,readback_end_ns\n");
    for (unsigned i = 0; i < trace_valid; i++) {
        const host_trace *t = trace_host+i;
        fprintf(out, "%u,%llu,%llu,%llu,%llu,%llu,%llu,%u,%u,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n", trace_first+i,
                (unsigned long long)t->observed, (unsigned long long)t->callback_start, (unsigned long long)t->callback_end,
                (unsigned long long)t->staging_start, (unsigned long long)t->staging_end, (unsigned long long)t->ready, t->bytes, t->short_reads,
                (unsigned long long)t->read_start, (unsigned long long)t->read_end, (unsigned long long)t->ready_start,
                (unsigned long long)t->poll_start, (unsigned long long)t->poll_end,
                (unsigned long long)t->readback_start, (unsigned long long)t->readback_end);
    }
    fprintf(out, "hosttrace_end\n"); return 0;
}
