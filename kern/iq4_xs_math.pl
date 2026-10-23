# IQ4_XS arithmetic of one block step (the sequence of gemv_iq4_xs.sass's step i, with its registers
# named for step 0): nibbles nb00-nb03, the int8 scale sc0, the block's fp16 d in the low half of dp0,
# the activations act00-act15, their scale as0 and (biased dot) the dot seed sbias0 = -128 * S32.
# The integer dot is exact; the float multiply and the FFMA into accf keep the scalar order.
# The first instruction waits on $wait; F2F uses barrier 4, I2F barrier 3.
sub iq4_xs_step_code {
    my ($biased_dot, $wait) = @_;
    my $add_lookup = $biased_dot;
    my $o = '';
    my @w = map { sprintf "nb%02d", $_ } 0..3;
    $o .= sprintf "%s:-:4:-:1      F2F.F32.F16 dd, dp0;\n", $wait;
    $o .= sprintf "--:-:-:-:1      SHR.U32 tt%d, %s, 0x10;\n", $_, $w[$_] for 0..3;
    $o .= sprintf "--:-:-:-:%s      LOP32I.XOR x0%d, %s, 0x88888888;\n", ($_ == 3 ? '3' : '1'), $_, $w[$_] for 0..3;
    $o .= sprintf "--:-:-:-:1      SHR.U32 x1%d, x0%d, 0x10;\n", $_, $_ for 0..3;
    for my $k (0..3) {
        $o .= sprintf "--:-:-:-:1      PRMT lo%d, tl0, %s, tl1;\n", $k, $w[$k];
        $o .= sprintf "--:-:-:-:1      PRMT hi%d, th0, x0%d, th1;\n", $k, $k;
        $o .= sprintf "--:-:-:-:1      PRMT mk%d, t7f, %s, t7f;\n", $k, $w[$k] unless $add_lookup;
    }
    for my $k (0..3) {
        $o .= sprintf "--:-:-:-:1      PRMT lo2%d, tl0, tt%d, tl1;\n", $k, $k;
        $o .= sprintf "--:-:-:-:1      PRMT hi2%d, th0, x1%d, th1;\n", $k, $k;
        $o .= sprintf "--:-:-:-:1      PRMT mk2%d, t7f, tt%d, t7f;\n", $k, $k unless $add_lookup;
    }
    if ($add_lookup) {
        $o .= sprintf "--:-:-:-:1      LOP32I.AND lo%d, lo%d, 0x7f7f7f7f;\n", $_, $_ for 0..3;
        $o .= sprintf "--:-:-:-:1      LOP32I.AND lo2%d, lo2%d, 0x7f7f7f7f;\n", $_, $_ for 0..3;
        $o .= sprintf "--:-:-:-:1      IADD R%d, lo%d, hi%d;\n", $_, $_, $_ for 0..3;
        $o .= sprintf "--:-:-:-:1      IADD R%d, lo2%d, hi2%d;\n", 4 + $_, $_, $_ for 0..3;
    } else {
        $o .= sprintf "--:-:-:-:1      SHL mb%d, mk%d, 0x1;\n", $_, $_ for 0..3;
        $o .= sprintf "--:-:-:-:1      SHL mb%d, mk2%d, 0x1;\n", 4 + $_, $_ for 0..3;
        $o .= sprintf "--:-:-:-:1      LOP.OR mk%d, mk%d, mb%d;\n", $_, $_, $_ for 0..3;
        $o .= sprintf "--:-:-:-:1      LOP.OR mk2%d, mk2%d, mb%d;\n", $_, $_, 4 + $_ for 0..3;
        $o .= sprintf "--:-:-:-:1      LOP3.LUT R%d, lo%d, mk%d, hi%d, 0xe2;\n", $_, $_, $_, $_ for 0..3;
        $o .= sprintf "--:-:-:-:1      LOP3.LUT R%d, lo2%d, mk2%d, hi2%d, 0xe2;\n", 4 + $_, $_, $_, $_ for 0..3;
    }
    for my $k (0..3) {
        my ($low, $high) = $biased_dot ? ('4140', '4342') : ('9180', 'b3a2');
        $o .= sprintf "--:-:-:-:1      PRMT p%02d, R%d, 0x%s, RZ;\n", 4*$k, $k, $low;
        $o .= sprintf "--:-:-:-:1      PRMT p%02d, R%d, 0x%s, RZ;\n", 4*$k + 1, $k, $high;
        $o .= sprintf "--:-:-:-:1      PRMT p%02d, R%d, 0x%s, RZ;\n", 4*$k + 2, 4 + $k, $low;
        $o .= sprintf "--:-:-:-:1      PRMT p%02d, R%d, 0x%s, RZ;\n", 4*$k + 3, 4 + $k, $high;
    }
    my $j = 0;
    for my $k (0..3) {
        for my $kk (0..3) {
            for my $hh (0, 1) {
                my $acc = sprintf "acc%d", $j & 7;
                $o .= sprintf "--:-:-:-:1      XMAD.S16.S16 %s, p%02d.%s, act%02d.%s, %s;\n",
                    $acc, 4*$k + $kk, $hh ? 'H1' : 'H0', 4*$k + $kk, $hh ? 'H1' : 'H0', $j == 0 && $biased_dot ? 'sbias0' : $j < 8 ? 'RZ' : $acc;
                $j++;
            }
        }
    }
    $o .= "--:-:-:-:2      IADD acc0, acc0, acc1;\n";
    $o .= "--:-:-:-:1      IADD acc2, acc2, acc3;\n";
    $o .= "--:-:-:-:2      IADD acc4, acc4, acc5;\n";
    $o .= "--:-:-:-:6      IADD acc6, acc6, acc7;\n";
    $o .= "--:-:-:-:1      IADD acc0, acc0, acc2;\n";
    $o .= "--:-:-:-:6      IADD acc4, acc4, acc6;\n";
    $o .= "--:-:-:-:6      IADD dot, acc0, acc4;\n";
    $o .= "--:-:-:-:1      XMAD.U16.S16 sumi, dot, sc0, RZ;\n";
    $o .= "--:-:-:-:6      XMAD.S16.S16 t, dot.H1, sc0, RZ;\n";
    $o .= "--:-:-:-:6      ISCADD sumi, t, sumi, 0x10;\n";
    $o .= "--:-:3:-:6      I2F.F32.S32 sumif, sumi;\n";
    $o .= "0c:-:-:-:6      FMUL u, dd, sumif;\n";
    $o .= "--:-:-:-:6      FFMA accf, as0, u, accf;\n";
    return $o;
}
1;
