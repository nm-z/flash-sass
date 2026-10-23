# Shared scalar/B8 RMS emitter. Batch mode changes descriptor/lane selection
# only; normalization arithmetic, reduction and scalar register numbers stay fixed.
use strict;
use warnings;

sub rmsnorm_regmap {
    my ($n, $batch) = @_;
    my $C = $n / 128;
    my $U = $C <= 20 ? $C : 16;
    my $o = "<REGISTER_MAPPING>\n";
    my $r = 0;
    $o .= sprintf "    %d-%d : x<00-%02d>\n", $r, $r + 4*$U - 1, 4*$U - 1; $r += 4*$U;
    $o .= sprintf "    %d-%d : wt<00-%02d>\n", $r, $r + 4*$U - 1, 4*$U - 1; $r += 4*$U;
    for my $name (qw(ss mm vv s0 hh rr e mneg1 t it tid lane invn eps bid off)) { $o .= sprintf "    %d : %s\n", $r++, $name; }
    $r += $r & 1;
    $o .= sprintf "    %d,%d : xp<0-1>\n    %d,%d : wp<0-1>\n    %d,%d : yp<0-1>\n", $r, $r+1, $r+2, $r+3, $r+4, $r+5;
    if ($batch) {
        die 'RMS batch currently supports K2560 only' if $n != 2560;
        $r += 6;
        $o .= sprintf "    %d : position\n    %d : descriptor\n",$r,$r+1;$r+=2;
        for my $name (qw(X W Y)) {
            $o .= sprintf "    %d,%d : param_%s<0-1>\n",$r,$r+1,$name;$r+=2;
        }
        $o .= sprintf "    %d : param_eps\n",$r;
    }
    $o .= "</REGISTER_MAPPING>\n";
    return $o;
}

sub rmsnorm_code {
    my ($n, $batch) = @_;
    my $C = $n / 128;
    my $U = $C <= 20 ? $C : 16;
    my $NI = $C / $U;
    my $invn = unpack("L", pack("f", 1.0 / $n));
    my $o = '';
    $o .= "--:-:6:-:1      S2R tid, SR_TID.X;\n";
    $o .= "--:-:6:-:1      S2R bid, SR_CTAID.X;\n";                   # CTA b normalises vector b (X, W, Y + 4 n b)
    if ($batch) {
        die 'RMS batch currently supports K2560 only' if $n != 2560;
        $o .= "20:-:-:-:6      SHR.U32 position, tid, 0x5;\n--:-:-:-:6      LOP32I.AND tid, tid, 0x1f;\n";
        $o .= "--:-:-:-:6      ISETP.EQ.U32.AND P6, PT, position, RZ, PT;\n--:-:-:-:6      SHL descriptor, position, 0x7;\n--:-:-:-:6      MOV32I off, 0xbf00;\n--:-:-:-:6      IADD descriptor, off, -descriptor;\n--:-:-:-:6  \@P6 MOV32I descriptor, 0xbf80;\n";
        my @names=qw(X W Y);
        for my $j (0..2) {
            $o .= sprintf "--:-:1:-:1      LDS.64 param_%s0, [descriptor+0x%x];\n",$names[$j],8+8*$j;
        }
        $o .= "--:-:1:-:2      LDS param_eps, [descriptor+0x20];\n01:-:-:-:6      NOP;\n";
    }
    $o .= "--:-:-:-:1      MOV32I mneg1, 0xbf800000;\n";
    $o .= sprintf "--:-:-:-:1      MOV32I invn, 0x%08x;\n", $invn;
    $o .= "--:-:-:-:6      MOV eps, param_eps;\n";
    $o .= "20:-:-:-:6      ISETP.GE.U32.AND P0, PT, tid, 0x20, PT;\n";
    $o .= "--:-:-:-:6      LOP32I.AND lane, tid, 0x1f;\n";
    $o .= "--:-:-:-:6      SHL t, lane, 0x4;\n";
    $o .= sprintf "--:-:-:-:6      XMAD off, bid, 0x%x, RZ;\n", 4 * $n;
    $o .= "--:-:-:-:6      IADD off, off, t;\n";
    $o .= "--:-:-:-:6      MOV it, RZ;\n";
    $o .= "--:-:-:-:6  \@P0 EXIT;\n";
    $o .= "--:-:-:-:6      IADD xp0.CC, off, param_X0;\n";
    $o .= "--:-:-:-:6      IADD.X xp1, RZ, param_X1;\n";
    $o .= "--:-:-:-:6      IADD wp0.CC, off, param_W0;\n";
    $o .= "--:-:-:-:6      IADD.X wp1, RZ, param_W1;\n";
    $o .= "--:-:-:-:6      IADD yp0.CC, off, param_Y0;\n";
    $o .= "--:-:-:-:6      IADD.X yp1, RZ, param_Y1;\n";
    $o .= "--:-:-:-:6      MOV ss, RZ;\n";
    # pass 1: sum of squares
    $o .= "\nPASS1:\n";
    for my $u (0..$U-1) { $o .= sprintf "--:-:1:-:%s      LDG.E.CI.128 x%02d, [xp+0x%x];\n", $u == $U-1 ? '2' : '1', 4*$u, 512*$u; }
    $o .= "--:-:-:-:6      IADD it, it, 0x1;\n";
    $o .= sprintf "--:-:-:-:6      ISETP.LT.U32.AND P4, PT, it, 0x%x, PT;\n", $NI;
    $o .= sprintf "%s:-:-:-:6      FFMA ss, x%02d, x%02d, ss;\n", $_ == 0 ? '01' : '--', $_, $_ for 0..4*$U-1;
    $o .= sprintf "--:-:-:-:6      IADD xp0.CC, xp0, 0x%x;\n", 512*$U;
    $o .= "--:-:-:-:6      IADD.X xp1, xp1, RZ;\n";
    $o .= "--:-:-:-:6  \@P4 BRA PASS1;\n" if $NI > 1;
    for my $x (16, 8, 4, 2, 1) {
        $o .= sprintf "--:-:5:-:3      SHFL.BFLY PT, t, ss, 0x%x, 0x1f;\n", $x;
        $o .= "10:-:-:-:6      FADD ss, ss, t;\n";
    }
    $o .= "--:-:-:-:6      FMUL mm, ss, invn;\n";
    $o .= "--:-:-:-:6      FADD vv, mm, eps;\n";
    $o .= "--:-:3:-:2      MUFU.RSQ rr, vv;\n";
    $o .= "04:-:-:-:1      FMUL.FTZ s0, vv, rr;\n";
    $o .= "--:-:-:-:6      FMUL.FTZ hh, rr, 0.5;\n";
    $o .= "--:-:-:-:6      FADD.FTZ e, -s0, -RZ;\n";
    $o .= "--:-:-:-:6      FFMA e, s0, e, vv;\n";
    $o .= "--:-:-:-:6      FFMA s0, e, hh, s0;\n";
    $o .= "--:-:3:-:2      MUFU.RCP rr, s0;\n";
    $o .= "04:-:-:-:6      FFMA e, s0, rr, mneg1;\n";
    $o .= "--:-:-:-:6      FADD.FTZ e, -e, -RZ;\n";
    $o .= "--:-:-:-:6      FFMA rr, rr, e, rr;\n";
    # reset xp, it
    $o .= "--:-:-:-:6      IADD xp0.CC, off, param_X0;\n";
    $o .= "--:-:-:-:6      IADD.X xp1, RZ, param_X1;\n";
    $o .= "--:-:-:-:6      MOV it, RZ;\n";
    # pass 2: y = (x * r) * w
    $o .= "\nPASS2:\n";
    for my $u (0..$U-1) { $o .= sprintf "%s:-:1:-:1      LDG.E.CI.128 x%02d, [xp+0x%x];\n", $u == 0 ? '08' : '--', 4*$u, 512*$u; }
    for my $u (0..$U-1) { $o .= sprintf "--:-:2:-:%s      LDG.E.CI.128 wt%02d, [wp+0x%x];\n", $u == $U-1 ? '2' : '1', 4*$u, 512*$u; }
    $o .= "--:-:-:-:6      IADD it, it, 0x1;\n";
    $o .= sprintf "--:-:-:-:6      ISETP.LT.U32.AND P4, PT, it, 0x%x, PT;\n", $NI;
    $o .= "01:-:-:-:6      DEPBAR.LE SB0, 0x0;\n" if $ENV{RMS_DEPBAR};
    $o .= sprintf "%s:-:-:-:%s      FMUL x%02d, x%02d, rr;\n", $_ == 0 ? '01' : '--', $_ == 4*$U-1 ? '6' : '1', $_, $_ for 0..4*$U-1;
    $o .= "02:-:-:-:6      DEPBAR.LE SB1, 0x0;\n" if $ENV{RMS_DEPBAR};
    $o .= sprintf "%s:-:-:-:%s      FMUL x%02d, x%02d, wt%02d;\n", $_ == 0 ? '02' : '--', $_ == 4*$U-1 ? '6' : '1', $_, $_, $_ for 0..4*$U-1;
    for my $u (0..$U-1) { $o .= sprintf "--:4:-:-:1      STG.E.128 [yp+0x%x], x%02d;\n", 512*$u, 4*$u; }
    $o .= sprintf "08:-:-:-:6      IADD xp0.CC, xp0, 0x%x;\n", 512*$U;
    $o .= "--:-:-:-:6      IADD.X xp1, xp1, RZ;\n";
    $o .= sprintf "--:-:-:-:6      IADD wp0.CC, wp0, 0x%x;\n", 512*$U;
    $o .= "--:-:-:-:6      IADD.X wp1, wp1, RZ;\n";
    $o .= sprintf "--:-:-:-:6      IADD yp0.CC, yp0, 0x%x;\n", 512*$U;
    $o .= "--:-:-:-:6      IADD.X yp1, yp1, RZ;\n";
    $o .= "--:-:-:-:6  \@P4 BRA PASS2;\n" if $NI > 1;
    $o .= "--:-:-:-:f      EXIT;\n";
    return $o;
}

1;
