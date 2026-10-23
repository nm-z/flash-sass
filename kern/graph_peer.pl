# GPU publication for independent per-die graphs. Scratch aliases are live
# only before/after the arithmetic body. Shared word zero is a prologue latch.
use strict;
use warnings;
sub graph_peer_mapping {
    return <<'SASS';
    0 : g_tid
    1 : g_lane
    2,3 : g_ctl<0-1>
    4,5 : g_ptr<0-1>
    6 : g_expected
    7 : g_value
    8 : g_time0
    9 : g_time1
    10 : g_flags
    11 : g_index
    12 : g_one
    13 : g_count
    14 : g_tmp
SASS
}
sub graph_peer_prologue {
    my $code=<<'SASS';
--:-:6:-:2      S2R g_tid, SR_TID.X;
--:-:-:-:6      MOV g_ctl0, c[0x0][0x1b8];
--:-:-:-:6      MOV g_ctl1, c[0x0][0x1bc];
20:-:-:-:6      ISETP.EQ.U32.AND P0, PT, g_tid, RZ, PT;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:1:-:2  @P0 LDG.E.CV g_value, [g_ctl+0x2c];
01:2:-:-:2  @P0 STS [RZ], g_value;
02:-:-:-:6      BAR.SYNC 0x0;
--:-:1:-:2      LDS g_value, [RZ];
01:-:-:-:6      ISETP.NE.U32.AND P0, PT, g_value, RZ, PT;
--:-:-:-:6      BAR.SYNC 0x0;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6  @P0 BRA GRAPH_EXIT;
--:-:-:-:6      MOV g_flags, c[0x0][0x1b0];
--:-:-:-:6      LOP32I.AND g_flags, g_flags, 0x2;
--:-:-:-:6      ISETP.EQ.U32.AND P0, PT, g_flags, RZ, PT;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6  @P0 BRA GRAPH_WAIT_DONE;
--:-:-:-:6      ISETP.GE.U32.AND P0, PT, g_tid, 0x20, PT;
--:-:-:-:6      LOP32I.AND g_lane, g_tid, 0x1f;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6  @P0 BRA GRAPH_WAIT_BAR;
--:-:1:-:1      LDG.E.CV g_expected, [g_ctl+0x20];
--:-:1:-:2      LDG.E.64 g_ptr0, [g_ctl+0x70];
--:-:-:-:6      SHL g_tmp, g_lane, 0x6;
01:-:-:-:6      IADD g_ptr0.CC, g_ptr0, g_tmp;
--:-:-:-:6      IADD.X g_ptr1, g_ptr1, RZ;
--:-:-:-:6      ISETP.LT.U32.AND P1, PT, g_lane, 0x6, PT;
--:-:-:-:6      ISETP.EQ.U32.AND P2, PT, g_lane, RZ, PT;
--:-:6:-:2      S2R g_time0, SR_GLOBALTIMERHI;
20:-:-:-:6      MOV32I g_value, 0xffffffff;
GRAPH_WAIT_SPIN:
--:-:1:-:2  @P1 LDG.E.CV g_value, [g_ptr];
01:-:-:-:6      ISETP.GE.U32.AND P0, PT, g_value, g_expected, PT;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6      VOTE.ALL P0, P0;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6  @P0 BRA GRAPH_WAIT_BAR;
--:-:1:-:2      LDG.E.CV g_count, [g_ctl+0x2c];
--:-:6:-:2      S2R g_time1, SR_GLOBALTIMERHI;
20:-:-:-:6      IADD g_time1, g_time1, -g_time0;
--:-:-:-:6      ISETP.GE.U32.AND P0, PT, g_time1, 0x46, PT;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
01:-:-:-:6      ISETP.NE.U32.OR P0, PT, g_count, RZ, P0;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6      VOTE.ANY P0, P0;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6 @!P0 BRA GRAPH_WAIT_SPIN;
--:-:-:-:6      MOV32I g_value, 0x1;
--:2:-:-:6  @P2 STG.E.WT [g_ctl+0x2c], g_value;
--:2:-:-:6  @P2 STS [RZ], g_value;
02:-:-:-:6      MEMBAR.SYS;
GRAPH_WAIT_BAR:
--:-:-:-:6      BAR.SYNC 0x0;
--:-:1:-:2      LDS g_value, [RZ];
01:-:-:-:6      ISETP.NE.U32.AND P0, PT, g_value, RZ, PT;
--:-:-:-:6      BAR.SYNC 0x0;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6  @P0 BRA GRAPH_EXIT;
GRAPH_WAIT_DONE:
--:-:-:-:6      MEMBAR.GL.IVALLD;
SASS
    if ($ENV{GRAPH_PEER_FAST}) {
        # Only peer-wait kernels can block indefinitely. Pure arithmetic
        # kernels finish normally; a failed graph never publishes their output.
        $code=<<'SASS'.$code;
--:-:-:-:6      MOV g_flags, c[0x0][0x1b0];
--:-:-:-:6      LOP32I.AND g_flags, g_flags, 0x2;
--:-:-:-:6      ISETP.EQ.U32.AND P0, PT, g_flags, RZ, PT;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6  @P0 BRA GRAPH_WAIT_DONE;
SASS
    }
    return $code;
}
sub graph_peer_footer {
    my $code=<<'SASS';
GRAPH_DONE:
3f:-:-:-:6      MEMBAR.SYS;
--:-:-:-:6      MOV g_flags, c[0x0][0x1b0];
--:-:-:-:6      LOP32I.AND g_flags, g_flags, 0x1;
--:-:-:-:6      ISETP.EQ.U32.AND P0, PT, g_flags, RZ, PT;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6  @P0 BRA GRAPH_EXIT;
--:-:-:-:6      BAR.SYNC 0x0;
--:-:6:-:2      S2R g_tid, SR_TID.X;
20:-:-:-:6      ISETP.GE.U32.AND P0, PT, g_tid, 0x20, PT;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6  @P0 BRA GRAPH_EXIT;
--:-:-:-:6      ISETP.EQ.U32.AND P2, PT, g_tid, RZ, PT;
--:-:-:-:6      MOV g_ctl0, c[0x0][0x1b8];
--:-:-:-:6      MOV g_ctl1, c[0x0][0x1bc];
--:-:-:-:6      MOV g_index, c[0x0][0x1b4];
--:-:-:-:6      SHL g_index, g_index, 0x2;
--:-:-:-:6      IADD g_index, g_index, 0x100;
--:-:-:-:6      IADD g_ptr0.CC, g_ctl0, g_index;
--:-:-:-:6      IADD.X g_ptr1, g_ctl1, RZ;
--:-:-:-:6      MOV32I g_one, 0x1;
--:-:1:-:6  @P2 ATOM.E.ADD g_count, [g_ptr], g_one;
01:-:3:-:2      SHFL.IDX PT, g_count, g_count, 0x0, 0x1f;
04:-:-:-:6      IADD g_count, g_count, 0x1;
--:-:-:-:6      ISETP.NE.U32.AND P0, PT, g_count, c[0x0][0x14], PT;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6  @P0 BRA GRAPH_EXIT;
--:2:-:-:6  @P2 STG.E [g_ptr], RZ;
--:-:1:-:2      LDG.E.CV g_value, [g_ctl+0x20];
01:-:-:-:6      IADD g_value, g_value, 0x1;
--:2:-:-:6  @P2 STG.E [g_ctl+0x20], g_value;
SASS
    for my $peer (0..5) {
        $code .= sprintf "02:-:1:-:2      LDG.E.64 g_ptr0, [g_ctl+0x%x];\n",0x40+8*$peer;
        $code .= "01:2:-:-:6  \@P2 STG.E.WT [g_ptr], g_value;\n";
    }
    if ($ENV{GRAPH_PEER_FAST}) {
        $code =~ s/GRAPH_DONE:\n3f:-:-:-:6      MEMBAR.SYS;/GRAPH_DONE:\n3f:-:-:-:6      NOP;/;
        $code =~ s/(\@P0 BRA GRAPH_EXIT;\n)(--:-:-:-:6      BAR.SYNC 0x0;)/$1--:-:-:-:6      MEMBAR.SYS;\n$2/ or die 'missing publication fence location';
    }
    return $code."02:-:-:-:6      MEMBAR.SYS;\nGRAPH_EXIT:\n3f:-:-:-:f      EXIT;\n";
}
1;
