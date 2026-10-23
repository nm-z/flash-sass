# Shared scalar Q3_K arithmetic emitter. Integer dots are exact; floating point
# scale multiplication, FFMA chains and caller butterflies retain scalar order.
sub q3_k_dot_code {
    my ($n, $prefetch, $bstep) = @_;
    my $o = '';
    # 2-bit code stream of one group word into acc (base index $ab), masks m[$ms..$ms+7]
    my $cmasks = sub {
        my ($word, $ms, $laststall) = @_;
        my $x = '';
        $x .= sprintf "--:-:-:-:%s      LOP32I.AND m%02d, %s, 0x%08x;\n", ($_ == 7 ? $laststall : '1'), $ms + $_, $word, (0x00030003 << (2*$_)) & 0xffffffff for 0..7;
        return $x;
    };
    my $cxmads = sub {
        my ($i, $gg, $ms, $ab) = @_;
        my $x = '';
        for my $hh (0, 1) {
            for my $s (0..7) {
                $x .= sprintf "--:-:-:-:1      XMAD.U16.S16 acc%02d, m%02d.%s, act%02d.%s, %s;\n",
                    $ab + $s, $ms + $s, $hh ? 'H1' : 'H0', 16*$i + 8*$gg + 4*$hh + ($s >> 1), ($s & 1) ? 'H1' : 'H0',
                    $hh ? sprintf("acc%02d", $ab + $s) : 'RZ';
            }
        }
        return $x;
    };
    # inverted-high-bit stream: source u = hw >> u, masks 0x00010001 << 4v in m[4(u&1) .. +3]
    my $hmasks = sub {
        my ($src, $u) = @_;
        my $x = '';
        $x .= sprintf "--:-:-:-:1      LOP32I.AND m%02d, %s, 0x%08x;\n", 4*($u & 1) + $_, $src, (0x00010001 << (4*$_)) & 0xffffffff for 0..3;
        return $x;
    };
    my $hxmads = sub {
        my ($i, $u) = @_;
        my $x = '';
        for my $hh (0, 1) {
            for my $v (0..3) {
                $x .= sprintf "--:-:-:-:1      XMAD.U16.S16 acch%d, m%02d.%s, act%02d.%s, %s;\n",
                    4*$hh + $v, 4*($u & 1) + $v, $hh ? 'H1' : 'H0', 16*$i + 8*$hh + 2*$v + ($u >> 1), ($u & 1) ? 'H1' : 'H0',
                    $u ? sprintf("acch%d", 4*$hh + $v) : 'RZ';
            }
        }
        return $x;
    };
    # dot = sum_s acc_s >> 2s over 8 accumulators (first SHR >= 7 issues after the last XMAD of acc_7)
    my $ccombine = sub {
        my ($ab, $dst) = @_;
        my $x = '';
        $x .= sprintf "--:-:-:-:%s      SHR acc%02d, acc%02d, 0x%x;\n", ($_ == 7 ? '4' : '1'), $ab + $_, $ab + $_, 2*$_ for 1..7;
        $x .= sprintf "--:-:-:-:1      IADD acc%02d, acc%02d, acc%02d;\n", $ab, $ab, $ab + 1;
        $x .= sprintf "--:-:-:-:1      IADD acc%02d, acc%02d, acc%02d;\n", $ab + 2, $ab + 2, $ab + 3;
        $x .= sprintf "--:-:-:-:1      IADD acc%02d, acc%02d, acc%02d;\n", $ab + 4, $ab + 4, $ab + 5;
        $x .= sprintf "--:-:-:-:6      IADD acc%02d, acc%02d, acc%02d;\n", $ab + 6, $ab + 6, $ab + 7;
        $x .= sprintf "--:-:-:-:1      IADD acc%02d, acc%02d, acc%02d;\n", $ab, $ab, $ab + 2;
        $x .= sprintf "--:-:-:-:6      IADD acc%02d, acc%02d, acc%02d;\n", $ab + 4, $ab + 4, $ab + 6;
        $x .= sprintf "--:-:-:-:6      IADD %s, acc%02d, acc%02d;\n", $dst, $ab, $ab + 4;
        return $x;
    };
    # dh_hh = sum_v acch_{4hh+v} >> 4v, both halves interleaved
    my $hcombine = sub {
        my $x = '';
        for my $v (1..3) {
            $x .= sprintf "--:-:-:-:1      SHR acch%d, acch%d, 0x%x;\n", $v, $v, 4*$v;
            $x .= sprintf "--:-:-:-:%s      SHR acch%d, acch%d, 0x%x;\n", ($v == 3 ? '3' : '1'), 4 + $v, 4 + $v, 4*$v;
        }
        $x .= "--:-:-:-:1      IADD acch0, acch0, acch1;\n";
        $x .= "--:-:-:-:1      IADD acch4, acch4, acch5;\n";
        $x .= "--:-:-:-:1      IADD acch2, acch2, acch3;\n";
        $x .= "--:-:-:-:6      IADD acch6, acch6, acch7;\n";
        $x .= "--:-:-:-:1      IADD dh0, acch0, acch2;\n";
        $x .= "--:-:-:-:6      IADD dh1, acch4, acch6;\n";
        return $x;
    };

    for my $i (0..$n-1) {
        my $wait = $i == 0 ? '03' : '--';
        my $w0 = sprintf "cw%02d", 2*$i;
        my $w1 = sprintf "cw%02d", 2*$i + 1;
        $o .= sprintf "%s:-:4:-:1      F2F.F32.F16 dd, dp%d%s;\n", $wait, $i >> 1, ($i & 1) ? '.H1' : '';
        $o .= sprintf "--:-:-:-:1      PRMT scw, sw%d, 0x9180, RZ;\n", $i;
        $o .= $cmasks->($w0, 0, '1');
        $o .= $cmasks->($w1, 8, '1');
        $o .= $cxmads->($i, 0, 0, 0);
        $o .= sprintf "--:-:-:-:1      SHR.U32 hs1, hw%d, 0x1;\n", $i;
        $o .= sprintf "--:-:-:-:1      SHR.U32 hs2, hw%d, 0x2;\n", $i;
        $o .= sprintf "--:-:-:-:1      SHR.U32 hs3, hw%d, 0x3;\n", $i;
        $o .= $hmasks->("hw$i", 0);
        $o .= $cxmads->($i, 1, 8, 8);
        $o .= $hmasks->('hs1', 1);
        $o .= $hxmads->($i, 0);
        $o .= $hmasks->('hs2', 2);
        $o .= $hxmads->($i, 1);
        $o .= $hmasks->('hs3', 3);
        $o .= $hxmads->($i, 2);
        $o .= $hxmads->($i, 3);
        $o .= $ccombine->(0, 'dot0');
        $o .= $ccombine->(8, 'dot1');
        $o .= $hcombine->();
        $o .= "--:-:-:-:1      SHL dh0, dh0, 0x2;\n";
        $o .= "--:-:-:-:6      SHL dh1, dh1, 0x2;\n";
        $o .= "--:-:-:-:1      IADD3 dot0, dot0, -dh0, RZ;\n";
        $o .= "--:-:-:-:6      IADD3 dot1, dot1, -dh1, RZ;\n";
        $o .= "--:-:-:-:6      XMAD.S16.S16 sumi, dot0, scw, RZ;\n";
        $o .= "--:-:-:-:6      XMAD.S16.S16 sumi, dot1, scw.H1, sumi;\n";
        $o .= "--:-:3:-:6      I2F.F32.S32 sumif, sumi;\n";
        $o .= "0c:-:-:-:6      FMUL u, dd, sumif;\n";
        $o .= sprintf "--:-:-:-:6      FFMA accf, as%d, u, accf;\n", $i;
        if ($prefetch) {
            # This block's codes/high bits/scales are dead. Reuse their registers
            # for the following row while the remaining current blocks compute.
            # Read barrier 2 guards every pointer until the row-end wait; write
            # barrier 1 is consumed by the following row's first F2F (03).
            $o .= sprintf "--:2:1:-:1  \@P3 LDG.E.CI.64 cw%02d, [cp+0x%x];\n", 2*$i, $bstep*$i;
            $o .= sprintf "--:2:1:-:1  \@P3 LDG.E.CI hw%d, [hp+0x%x];\n", $i, $bstep*$i;
            $o .= sprintf "--:2:1:-:2  \@P3 LDG.E.CI.U16 sw%d, [sp+0x%x];\n", $i, $bstep*$i;
            # Packed d contains scales for several current blocks. It cannot
            # be replaced until every block has converted its current scale.
            $o .= "--:2:1:-:2  \@P3 LDG.E.CI.128 dp0, [pp];\n" if $i == $n-1;
        }
    }
    return $o;
}
1;
