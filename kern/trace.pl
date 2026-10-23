# Optional profiling only: DIE_TRACE=1 records CTA 0, DIE_TRACE=16 records every CTA.
# ctl+0xd0: u64 records pointer; +0xd8: first position; +0xdc: position count; +0xe0: allocated CTA count.
# Record layout: [position-first][phase][CTA][8] u64 nanoseconds.
# Five samples: phase start, wait start, wait end, body end, barrier end.
# Registers are framework scratch; op/units and prefetched descriptors stay intact.
sub trace_code {
    my ($stage, $tok_off, $step_off) = @_;
    $step_off = 0xbf50 unless defined $step_off;
    my $ctas = $ENV{DIE_TRACE} || 0;
    return '' unless $ctas;
    die "DIE_TRACE must be 1 or 16\n" unless $ctas == 1 || $ctas == 16;
    my $code = <<'SASS';
3f:-:-:-:6      NOP;
--:-:-:-:6      ISETP.NE.U32.AND P4, PT, warp, RZ, PT;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6  @P4 BRA TRACE_DONE_%S%;
%CTA_GUARD%
--:-:-:-:6      MOV fw0, ctl0;
--:-:-:-:6      MOV fw1, ctl1;
--:-:2:-:2      LDG.E bar_old, [fw+0xe0];
02:-:-:-:6      ISETP.EQ.U32.AND P4, PT, bar_old, %CTAS%, PT;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6 @!P4 BRA TRACE_DONE_%S%;
--:-:1:-:1      LDG.E.64 bar_cp0, [fw+0xd0];
--:-:2:-:1      LDG.E bar_old, [fw+0xd8];
--:-:2:-:1      LDG.E bar_one, [fw+0xdc];
--:-:2:-:2      LDS bar_t, [RZ+%TOK%];
%TOKEN_ADJUST%
02:-:-:-:6      IADD bar_t, bar_t, -bar_old;
--:-:-:-:6      ISETP.LT.U32.AND P4, PT, bar_t, bar_one, PT;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6 @!P4 BRA TRACE_DONE_%S%;
--:-:-:-:6      XMAD bar_t, bar_t, nph, pidx;
%PHASE_ADJUST%
%CTA_OFFSET%
--:-:-:-:6      SHL bar_t, bar_t, 0x6;
01:-:-:-:6      IADD bar_cp0.CC, bar_cp0, bar_t;
--:-:-:-:6      IADD.X bar_cp1, bar_cp1, RZ;
TRACE_CLOCK_%S%:
--:-:6:-:1      S2R bar_g2, SR_GLOBALTIMERHI;
--:-:6:-:1      S2R bar_g, SR_GLOBALTIMERLO;
--:-:6:-:2      S2R bar_old, SR_GLOBALTIMERHI;
20:-:-:-:6      ISETP.NE.U32.AND P4, PT, bar_g2, bar_old, PT;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6  @P4 BRA TRACE_CLOCK_%S%;
--:-:-:-:6      ISETP.EQ.U32.AND P4, PT, tid, RZ, PT;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:1:-:-:6  @P4 STG.E.64 [bar_cp+%OFF%], traceclock0;
TRACE_DONE_%S%:
01:-:-:-:6      IADD bar_t, cta, tid;
--:-:-:-:6      ISETP.EQ.U32.AND P4, PT, bar_t, RZ, PT;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
SASS
    my $guard = $ctas == 1 ? <<'SASS' : '';
--:-:-:-:6      ISETP.NE.U32.AND P4, PT, cta, RZ, PT;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6  @P4 BRA TRACE_DONE_%S%;
SASS
    my $offset = $ctas == 16 ? "--:-:-:-:6      SHL bar_t, bar_t, 0x4;\n--:-:-:-:6      IADD bar_t, bar_t, cta;\n" : '';
    $code =~ s/%CTA_GUARD%/$guard/g;
    $code =~ s/%CTA_OFFSET%/$offset/g;
    my $token_adjust = $stage == 7 ? <<'SASS' : '';
02:-:3:-:2      LDG.E bar_g2, [fw+0x24];
04:-:-:-:6      IMNMX.U32 bar_g2, bar_g2, 0x1, !PT;
--:-:-:-:6      LOP32I.AND bar_t, bar_t, 0x7fffffff;
--:-:-:-:6      IADD bar_t, bar_t, -bar_g2;
SASS
    my $phase_adjust = $stage == 7
        ? sprintf("--:-:2:-:2      LDS bar_g2, [RZ+0x%x];\n02:-:-:-:6      IADD bar_t, bar_t, -bar_g2;\n", $step_off)
        : '';
    $code =~ s/%TOKEN_ADJUST%/$token_adjust/g;
    $code =~ s/%PHASE_ADJUST%/$phase_adjust/g;
    $code =~ s/%S%/$stage/g;
    my $cta_hex = sprintf "0x%x", $ctas;
    $code =~ s/%CTAS%/$cta_hex/g;
    my $tok = sprintf '0x%x', $tok_off;
    my $off = sprintf '0x%x', 8 * $stage;
    $code =~ s/%TOK%/$tok/g;
    $code =~ s/%OFF%/$off/g;
    return $code;
}
1;
