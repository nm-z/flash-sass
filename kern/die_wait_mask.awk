# Narrow the HCWSUM exchange wait to the source CTA chunks this CTA reads.
# Descriptor flag bit 0x20 marks HCWSUM; each CTA reduces 160 fp32 columns.
/# ---- exchange wait/ { in_wait = 1 }
{
    if (in_wait && !saved_flags && /LDS bar_g, \[RZ\+0x[[:xdigit:]]+\];/) {
        print
        print "--:-:-:-:6      MOV bar_one, bar_g;"
        saved_flags = 1
        next
    }
    if (in_wait && $0 == "XWSPIN:") {
        print
        print "# HCWSUM CTA c reads byte range [c*640, (c+1)*640); sender chunks are 1024 bytes."
        print "--:-:-:-:6      LOP32I.AND bar_old, bar_one, 0x20;"
        print "--:-:-:-:6      ISETP.EQ.U32.AND P6, PT, bar_old, RZ, PT;"
        print "--:-:-:-:6      LOP32I.AND bar_old, lane, 0xf;"
        print "--:-:-:-:6      SHL bar_t, cta, 0x2;"
        print "--:-:-:-:6      IADD bar_t, bar_t, cta;"
        print "--:-:-:-:6      SHR.U32 bar_t, bar_t, 0x3;"
        print "--:-:-:-:6      SHL bar_g2, cta, 0x2;"
        print "--:-:-:-:6      IADD bar_g2, bar_g2, cta;"
        print "--:-:-:-:6      IADD bar_g2, bar_g2, 0x4;"
        print "--:-:-:-:6      SHR.U32 bar_g2, bar_g2, 0x3;"
        print "--:-:-:-:6      ISETP.GE.U32.OR P4, PT, bar_old, bar_t, P6;"
        print "--:-:-:-:6      ISETP.GE.U32.OR P3, PT, bar_g2, bar_old, P6;"
        mask_range = 1
        next
    }
    if (in_wait && !mask_flags && /LDG.E.CV bar_g2, \[xe\+0x100\];/) {
        print
        print "--:-:-:-:6  @!P4 MOV bar_t, bar_g;"
        print "--:-:-:-:6  @!P3 MOV bar_t, bar_g;"
        print "--:-:-:-:6  @!P4 MOV bar_one, bar_g;"
        print "--:-:-:-:6  @!P3 MOV bar_one, bar_g;"
        print "--:-:-:-:6  @!P4 MOV bar_g2, bar_g;"
        print "--:-:-:-:6  @!P3 MOV bar_g2, bar_g;"
        mask_flags = 1
        next
    }
    if (in_wait && $0 == "XWDONE:") in_wait = 0
    print
}
END {
    if (!saved_flags || !mask_range || !mask_flags) {
        print "die_wait_mask.awk: expected exchange-wait SASS anchors were not found" > "/dev/stderr"
        exit 1
    }
}
