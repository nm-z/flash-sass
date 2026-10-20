# Fail-closed transformer for the generated die kernel. All input is buffered;
# a missing/ambiguous anchor or a changed framework ABI produces no SASS output.
BEGIN {
    marker = "PHASE_PROBE_DISPATCH_V2"
    bad = 0
}

function fail(message) {
    if (!bad) print "phase_probe.awk: " message > "/dev/stderr"
    bad = 1
}

{
    if (source_file == "") source_file = FILENAME
    else if (FILENAME != source_file) fail("expected exactly one input file")
    line[++n] = $0
    lower = tolower($0)

    if (index($0, marker)) fail("already patched")
    if (index($0, "PHASE_PROBE_DONE")) fail("probe label already exists")
    if ($0 ~ /^[[:space:]]*OP_DISPATCH:[[:space:]]*$/) {
        anchor = n
        anchors++
    }
    if (lower ~ /\[ctl[[:space:]]*\+[[:space:]]*0x0*e8\]/)
        fail("control offset 0xe8 is already used")

    if ($0 ~ /^[[:space:]]*0[[:space:]]*:[[:space:]]*tid[[:space:]]*$/) have_tid = 1
    if ($0 ~ /^[[:space:]]*2[[:space:]]*:[[:space:]]*warp[[:space:]]*$/) have_warp = 1
    if ($0 ~ /^[[:space:]]*3[[:space:]]*:[[:space:]]*cta[[:space:]]*$/) have_cta = 1
    if ($0 ~ /^[[:space:]]*6[[:space:]]*:[[:space:]]*units[[:space:]]*$/) have_units = 1
    if ($0 ~ /^[[:space:]]*7[[:space:]]*:[[:space:]]*pidx[[:space:]]*$/) have_pidx = 1
    if ($0 ~ /^[[:space:]]*11,[[:space:]]*12[[:space:]]*:[[:space:]]*ctl<0-1>[[:space:]]*$/) have_ctl = 1
    if ($0 ~ /^[[:space:]]*13[[:space:]]*:[[:space:]]*op[[:space:]]*$/) have_op = 1
    if ($0 ~ /^[[:space:]]*16,[[:space:]]*17[[:space:]]*:[[:space:]]*dsp<0-1>[[:space:]]*$/) have_dsp = 1
    if ($0 ~ /^[[:space:]]*14,15[[:space:]]*:[[:space:]]*fw<0-1>[[:space:]]*$/) have_fw = 1
    if ($0 ~ /^[[:space:]]*18,19[[:space:]]*:[[:space:]]*traceclock<0-1>[[:space:]]*$/) have_clock = 1
    if ($0 ~ /^[[:space:]]*18[[:space:]]*:[[:space:]]*bar_g[[:space:]]*$/) have_bar_g = 1
    if ($0 ~ /^[[:space:]]*19[[:space:]]*:[[:space:]]*bar_g2[[:space:]]*$/) have_bar_g2 = 1
    if ($0 ~ /^[[:space:]]*20[[:space:]]*:[[:space:]]*bar_old[[:space:]]*$/) have_bar_old = 1
    if ($0 ~ /^[[:space:]]*21[[:space:]]*:[[:space:]]*bar_t[[:space:]]*$/) have_bar_t = 1
    if ($0 ~ /^[[:space:]]*24,25[[:space:]]*:[[:space:]]*bar_cp<0-1>[[:space:]]*$/) have_bar_cp = 1

    if (anchor && n > anchor && !setup_seen) {
        if ($0 ~ /\ydsp([01])?\y/) fail("dsp0/dsp1 is live in the dispatcher")
        if ($0 ~ /\yfw1\y/) fail("fw1 is live in the dispatcher")
        if ($0 ~ /SHL[[:space:]]+fw0,[[:space:]]*op,[[:space:]]*0x5/) {
            if (fw_reset) fail("dispatcher resets fw0 more than once")
            fw_reset = 1
        } else if ($0 ~ /\yfw0\y/ && !fw_reset) {
            fail("fw0 is used before the dispatcher's reset")
        }
        if ($0 ~ /^[[:space:]]*[[:alnum:]_]+_SETUP:[[:space:]]*$/) setup_seen = 1
    }
}

END {
    if (n == 0) fail("empty input")
    if (anchors != 1) fail("expected exactly one OP_DISPATCH label")
    if (!setup_seen) fail("no opcode setup follows OP_DISPATCH")
    if (!have_tid || !have_warp || !have_cta || !have_units || !have_pidx ||
        !have_ctl || !have_op || !have_dsp || !have_fw || !have_clock || !have_bar_g ||
        !have_bar_g2 || !have_bar_old || !have_bar_t || !have_bar_cp)
        fail("generated framework register map does not match the audited ABI")
    if (fw_reset != 1) fail("expected the audited jump-dispatch fw0 reset")
    if (bad) exit 2

    for (i = 1; i <= n; i++) {
        print line[i]
        if (i == anchor) {
            print "# " marker
            print "# Preserve the incoming per-thread P4 in dsp0 (dead in this dispatcher)."
            print "--:-:-:-:6      MOV dsp0, RZ;"
            print "--:-:-:-:6  @P4 MOV32I dsp0, 0x1;"
            print "# Skip all but warp zero uniformly; no lane-dependent branch is used."
            print "--:-:-:-:6      ISETP.NE.U32.AND P4, PT, warp, RZ, PT;"
            print "--:-:-:-:6      NOP;"
            print "--:-:-:-:6      NOP;"
            print "--:-:-:-:6      NOP;"
            print "--:-:-:-:6  @P4 BRA PHASE_PROBE_DONE;"
            print "# Fixed kernel-base-token window [1024,1280); model positions are unchanged."
            print "--:-:2:-:2      LDS bar_t, [RZ+0xbf70];"
            print "02:-:-:-:6      ISETP.LT.U32.AND P4, PT, bar_t, 0x400, PT;"
            print "--:-:-:-:6      NOP;"
            print "--:-:-:-:6      NOP;"
            print "--:-:-:-:6      NOP;"
            print "--:-:-:-:6  @P4 BRA PHASE_PROBE_DONE;"
            print "--:-:-:-:6      ISETP.GE.U32.AND P4, PT, bar_t, 0x500, PT;"
            print "--:-:-:-:6      NOP;"
            print "--:-:-:-:6      NOP;"
            print "--:-:-:-:6      NOP;"
            print "--:-:-:-:6  @P4 BRA PHASE_PROBE_DONE;"
            print "--:-:-:-:6      MOV fw0, ctl0;"
            print "--:-:-:-:6      MOV fw1, ctl1;"
            print "--:-:1:-:2      LDG.E.64 bar_cp0, [fw+0xe8];"
            print "--:-:2:-:2      LDS bar_g2, [RZ+0xbf54];"
            print "# The dependent wait fields above drain probe loads before normal dispatch."
            print "03:-:-:-:6      LOP.OR bar_old, bar_cp0, bar_cp1;"
            print "--:-:-:-:6      ISETP.EQ.U32.AND P4, PT, bar_old, RZ, PT;"
            print "--:-:-:-:6      NOP;"
            print "--:-:-:-:6      NOP;"
            print "--:-:-:-:6      NOP;"
            print "--:-:-:-:6  @P4 BRA PHASE_PROBE_DONE;"
            print "--:-:-:-:6      IADD traceclock0, pidx, 0x1;"
            print "--:-:-:-:6      SHL traceclock0, traceclock0, 0x5;"
            print "--:-:-:-:6      LOP.OR traceclock0, traceclock0, bar_g2;"
            print "--:-:-:-:6      MOV traceclock1, bar_t;"
            print "--:-:-:-:6      SHL bar_t, cta, 0x3;"
            print "--:-:-:-:6      IADD bar_cp0.CC, bar_cp0, bar_t;"
            print "--:-:-:-:6      IADD.X bar_cp1, bar_cp1, RZ;"
            print "--:-:-:-:6      ISETP.EQ.U32.AND P4, PT, tid, RZ, PT;"
            print "--:-:-:-:6      NOP;"
            print "--:-:-:-:6      NOP;"
            print "--:-:-:-:6      NOP;"
            print "--:1:-:-:6  @P4 STG.E.64 [bar_cp], traceclock0;"
            print "PHASE_PROBE_DONE:"
            print "01:-:-:-:6      NOP;"
            print "--:-:-:-:6      ISETP.NE.U32.AND P4, PT, dsp0, RZ, PT;"
            print "--:-:-:-:6      NOP;"
            print "--:-:-:-:6      NOP;"
            print "--:-:-:-:6      NOP;"
        }
    }
}
