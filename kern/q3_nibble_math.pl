# Experimental Q3 integer dot. Coefficients are unsigned q+4 in four-bit slots;
# a resident -4*sum16 bias recovers the original signed Q3 dot exactly.
sub q3_nibble_bias_code {
    my ($n) = @_;
    my $o = '';
    for my $pair ([dh0 => 0x0f0f0f0f], [dh1 => 0x33333333],
                  [hs1 => 0x03030303], [hs2 => 0x11111111],
                  [hs3 => 0x44444444]) {
        $o .= sprintf "--:-:-:-:6      MOV32I %s, 0x%08x;\n", @$pair;
    }
    # Ten/twelve independent chains leave at least six issues between writes.
    for my $e (0..15) {
        for my $g (0..2*$n-1) {
            $o .= sprintf "--:-:-:-:1      XMAD.S16.S16 bias%02d, act%02d.%s, 0xfffc, %s;\n",
                $g, 8*$g + ($e >> 1), ($e & 1) ? 'H1' : 'H0',
                $e ? sprintf('bias%02d', $g) : 'RZ';
        }
    }
    # The two-chain standalone integer fixture also uses this emitter.
    if (2*$n < 6) {
        die 'nibble bias requires at least six independent chains';
    }
    return $o;
}

# The integer fixture uses one activation pair, so it schedules its two bias
# chains with explicit six-cycle spacing instead of the many-chain prologue.
sub q3_nibble_pair_bias_code {
    my $o = '';
    for my $pair ([dh0 => 0x0f0f0f0f], [dh1 => 0x33333333],
                  [hs1 => 0x03030303], [hs2 => 0x11111111],
                  [hs3 => 0x44444444]) {
        $o .= sprintf "--:-:-:-:6      MOV32I %s, 0x%08x;\n", @$pair;
    }
    for my $e (0..15) {
        for my $g (0,1) {
            $o .= sprintf "--:-:-:-:%d      XMAD.S16.S16 bias%02d, act%02d.%s, 0xfffc, %s;\n",
                $g ? 5 : 1, $g, 8*$g + ($e >> 1), ($e & 1) ? 'H1' : 'H0',
                $e ? sprintf('bias%02d', $g) : 'RZ';
        }
    }
    return $o;
}

sub q3_nibble_integer_code {
    my ($i) = @_;
    my $o = '';
    my $round = sub {
        my ($f) = @_;
        my $x = '';
        $x .= $f->($_, $_ == 3 ? 3 : 1) for 0..3;
        return $x;
    };
    # Each packed word contains four coefficients in each 16-bit half. PRMT
    # selects bytes 0/2 or 1/3 of the original 2-bit code group word.
    for my $p (0..3) {
        $o .= sprintf "--:-:-:-:1      PRMT acch%d, cw%02d, 0x%s, RZ;\n",
            $p, 2*$i + ($p >> 1), ($p & 1) ? '4341' : '4240';
    }
    $o .= sprintf "--:-:-:-:1      PRMT m04, hw%d, 0x4140, RZ;\n", $i;
    $o .= sprintf "--:-:-:-:1      PRMT m05, hw%d, 0x4342, RZ;\n", $i;
    $o .= $round->(sub { my ($p,$s)=@_; sprintf "--:-:-:-:%d      SHL m%02d, acch%d, 0x4;\n",$s,$p,$p; });
    $o .= $round->(sub { my ($p,$s)=@_; sprintf "--:-:-:-:%d      LOP3.LUT acch%d, acch%d, m%02d, dh0, 0xa8;\n",$s,$p,$p,$p; });
    $o .= $round->(sub { my ($p,$s)=@_; sprintf "--:-:-:-:%d      SHL m%02d, acch%d, 0x2;\n",$s,$p,$p; });
    $o .= $round->(sub { my ($p,$s)=@_; sprintf "--:-:-:-:%d      LOP3.LUT acch%d, acch%d, m%02d, dh1, 0xa8;\n",$s,$p,$p,$p; });
    $o .= $round->(sub { my ($p,$s)=@_; sprintf "--:-:-:-:%d      SHR.U32 acch%d, m%02d, 0x%x;\n",$s,4+$p,4+($p>>1),4*($p&1); });
    $o .= $round->(sub { my ($p,$s)=@_; sprintf "--:-:-:-:%d      LOP32I.AND acch%d, acch%d, 0x000f000f;\n",$s,4+$p,4+$p; });
    $o .= $round->(sub { my ($p,$s)=@_; sprintf "--:-:-:-:%d      SHL m%02d, acch%d, 0x6;\n",$s,$p,4+$p; });
    $o .= $round->(sub { my ($p,$s)=@_; sprintf "--:-:-:-:%d      LOP3.LUT acch%d, acch%d, m%02d, hs1, 0xa8;\n",$s,4+$p,4+$p,$p; });
    $o .= $round->(sub { my ($p,$s)=@_; sprintf "--:-:-:-:%d      SHL m%02d, acch%d, 0x3;\n",$s,$p,4+$p; });
    $o .= $round->(sub { my ($p,$s)=@_; sprintf "--:-:-:-:%d      LOP3.LUT acch%d, acch%d, m%02d, hs2, 0xa8;\n",$s,4+$p,4+$p,$p; });
    $o .= $round->(sub { my ($p,$s)=@_; sprintf "--:-:-:-:%d      SHL m%02d, acch%d, 0x2;\n",$s,$p,4+$p; });
    $o .= $round->(sub { my ($p,$s)=@_; sprintf "--:-:-:-:%d      LOP3.LUT acch%d, acch%d, m%02d, hs3, 0xf6;\n",$s,$p,$p,$p; });
    for my $p (0..3) {
        for my $s (0..3) {
            $o .= sprintf "--:-:-:-:1      LOP32I.AND m%02d, acch%d, 0x%08x;\n",
                4*$p+$s, $p, (0x00070007 << (4*$s)) & 0xffffffff;
        }
    }
    # Sixteen independent chains, one pair of products per chain. The bias is
    # added only to unshifted s=0 chains, so every later shift remains exact.
    for my $half (0,1) {
        for my $p (0..3) {
            for my $s (0..3) {
                my $a=4*$p+$s;
                my $e=16*($p>>1)+4*($p&1)+8*$half+$s;
                my $add=$half ? sprintf('acc%02d',$a) : $a==0 ? sprintf('bias%02d',2*$i) :
                    $a==8 ? sprintf('bias%02d',2*$i+1) : 'RZ';
                $o .= sprintf "--:-:-:-:1      XMAD.U16.S16 acc%02d, m%02d.%s, act%02d.%s, %s;\n",
                    $a,$a,$half?'H1':'H0',16*$i+($e>>1),($e&1)?'H1':'H0',$add;
            }
        }
    }
    for my $group (0,1) {
        my $a=8*$group;
        # Retain the baseline combine schedule, including the harmless >>0.
        for my $s (1..7) {
            $o .= sprintf "--:-:-:-:%d      SHR acc%02d, acc%02d, 0x%x;\n",
                $s==7?4:1,$a+$s,$a+$s,4*($s&3);
        }
        for my $p ([0,0,1,1],[2,2,3,1],[4,4,5,1],[6,6,7,6],[0,0,2,1],[4,4,6,6]) {
            $o .= sprintf "--:-:-:-:%d      IADD acc%02d, acc%02d, acc%02d;\n",$p->[3],map {$a+$_} @$p[0..2];
        }
        $o .= sprintf "--:-:-:-:6      IADD dot%d, acc%02d, acc%02d;\n",$group,$a,$a+4;
    }
    return $o;
}

sub q3_nibble_dot_code {
    my ($n,$prefetch,$bstep)=@_;
    my $o='';
    for my $i (0..$n-1) {
        $o .= sprintf "%s:-:4:-:1      F2F.F32.F16 dd, dp%d%s;\n",$i==0?'03':'--',$i>>1,($i&1)?'.H1':'';
        $o .= sprintf "--:-:-:-:1      PRMT scw, sw%d, 0x9180, RZ;\n",$i;
        $o .= q3_nibble_integer_code($i);
        $o .= "--:-:-:-:6      XMAD.S16.S16 sumi, dot0, scw, RZ;\n";
        $o .= "--:-:-:-:6      XMAD.S16.S16 sumi, dot1, scw.H1, sumi;\n";
        $o .= "--:-:3:-:6      I2F.F32.S32 sumif, sumi;\n";
        $o .= "0c:-:-:-:6      FMUL u, dd, sumif;\n";
        $o .= sprintf "--:-:-:-:6      FFMA accf, as%d, u, accf;\n",$i;
        if ($prefetch) {
            $o .= sprintf "--:2:1:-:1  \@P3 LDG.E.CI.64 cw%02d, [cp+0x%x];\n",2*$i,$bstep*$i;
            $o .= sprintf "--:2:1:-:1  \@P3 LDG.E.CI hw%d, [hp+0x%x];\n",$i,$bstep*$i;
            $o .= sprintf "--:2:1:-:2  \@P3 LDG.E.CI.U16 sw%d, [sp+0x%x];\n",$i,$bstep*$i;
            $o .= "--:2:1:-:2  \@P3 LDG.E.CI.128 dp0, [pp];\n" if$i==$n-1;
        }
    }
    return $o;
}
1;
