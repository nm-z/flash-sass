# Optional profiling only: DIE_TRACE=1 records CTA 0, DIE_TRACE=16 records every CTA.
# ctl+0xd0: u64 records pointer; +0xd8: first position; +0xdc: position count;
# +0xe0: allocated CTA count; +0xe4: record words (16, schema 2).
# Record layout: [position-first][phase][CTA][16] u64 words.
# Slots 0..7 retain their old meanings. 8: selected-body entry; 9/10: peer
# publication brackets; 11: incoming exchange generation (integer, not time).
# Registers are framework scratch; op/units and prefetched descriptors stay intact.
sub trace_code {
    my ($stage, $tok_off, $label, $flags_off) = @_;
    $label = $stage unless defined $label;
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
--:-:2:-:2      LDG.E bar_old, [fw+0xe4];
02:-:-:-:6      ISETP.EQ.U32.AND P4, PT, bar_old, 0x10, PT;
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
--:-:-:-:6      SHL bar_t, bar_t, 0x7;
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
%GENERATION%
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
    my $token_adjust = $stage == 7 ? "02:-:-:-:6      LOP32I.AND bar_t, bar_t, 0x7fffffff;\n--:-:-:-:6      IADD bar_t, bar_t, -0x1;\n" : '';
    my $phase_adjust = $stage == 7 ? "--:-:-:-:6      IADD bar_t, bar_t, -0x1;\n" : '';
    $code =~ s/%TOKEN_ADJUST%/$token_adjust/g;
    $code =~ s/%PHASE_ADJUST%/$phase_adjust/g;
    my $generation = $stage == 1 ? sprintf(<<'SASS', $tok_off + 4) : '';
01:-:2:-:2      LDS bar_g, [RZ+0x%x];
--:-:-:-:6      MOV bar_g2, RZ;
02:1:-:-:6  @P4 STG.E.64 [bar_cp+0x58], traceclock0;
SASS
    $code =~ s/%GENERATION%/$generation/g;
    $code =~ s/%S%/$label/g;
    my $cta_hex = sprintf "0x%x", $ctas;
    $code =~ s/%CTAS%/$cta_hex/g;
    my $tok = sprintf '0x%x', $tok_off;
    my $off = sprintf '0x%x', 8 * $stage;
    $code =~ s/%TOK%/$tok/g;
    $code =~ s/%OFF%/$off/g;
    if ($stage == 9 || $stage == 10) {
        # Peer pointers, generation, prefetch and P2/P3/P4/P5/P6 are live at
        # publication. Use dead op registers and P0 exclusively here.
        my %rename = (fw => 'tr_fw', fw0 => 'tr_fw0', fw1 => 'tr_fw1',
            bar_cp => 'tr_cp', bar_cp0 => 'tr_cp0', bar_cp1 => 'tr_cp1',
            bar_g => 'tr_clock0', bar_g2 => 'tr_clock1', traceclock0 => 'tr_clock0',
            bar_old => 'tr_old', bar_t => 'tr_t', bar_one => 'tr_one', P4 => 'P0');
        my $names = join '|', sort { length($b) <=> length($a) } keys %rename;
        $code =~ s/\b($names)\b/$rename{$1}/g;
        die "publication trace needs flags offset\n" unless defined $flags_off;
        # Branch uniformly for the warp. P4 is lane 0's send predicate; using
        # it as a branch guard would diverge around the later VOTE/barrier.
        my $guard = sprintf <<'SASS', $flags_off;
3f:-:-:-:6      NOP;
--:-:1:-:2      LDS tr_old, [RZ+0x%x];
01:-:-:-:6      LOP32I.AND tr_old, tr_old, 0x1;
--:-:-:-:6      ISETP.EQ.U32.AND P0, PT, tr_old, RZ, PT;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6  @P0 BRA TRACE_DONE_%%LABEL%%;
SASS
        $guard =~ s/%LABEL%/$label/g;
        $code = $guard . $code;
    }
    return $code;
}
1;
