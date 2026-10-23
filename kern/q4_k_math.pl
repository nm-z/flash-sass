# Q4_K / Q5_K arithmetic of one sub-block step (the sequence of gemv_q4_k.sass's step i, with its
# registers named for step 0): header hdr00-hdr03 (fp16 d, dmin, 12 scale bytes), nibbles q00-q03,
# (q5_k) high bits qh0, the activations act00-act15, their per-32 sum sum0 and scale as0. The lane's
# sub-block index j enters through shj = 8 (j & 3) and the predicate $jp = (j >= 4). The integer dot
# is exact; the float sequence and the FFMA into accf keep the scalar order. The first instruction
# waits on $wait; F2F uses barrier 4, I2F barrier 3.
sub q4_k_step_code {
    my ($q5, $wait, $jp) = @_;
    my $stream = sub {
        my ($src, $maskf, $accn, $elem) = @_;
        my $x = '';
        $x .= sprintf "--:-:-:-:%s      LOP32I.AND m%d, %s, %s;\n", $_ == 3 ? '3' : '1', $_, $src->[0], $maskf->($_) for 0..3;
        for my $k (0..3) {
            my $set = ($k & 1) * 4;
            $x .= sprintf "--:-:-:-:1      LOP32I.AND m%d, %s, %s;\n", (1 - ($k & 1)) * 4 + $_, $src->[$k+1], $maskf->($_) for ($k < 3 ? (0..3) : ());
            my @ops;
            for my $hh (0, 1) {
                for my $v (0..3) {
                    my $e = $elem->($k, $v, $hh);
                    my $acc = $accn . (4*$hh + $v);
                    push @ops, sprintf "--:-:-:-:1      XMAD.U16.S16 %s, m%d.%s, act%02d.%s, %s;\n",
                        $acc, $set + $v, $hh ? 'H1' : 'H0', int($e/2), ($e & 1) ? 'H1' : 'H0', ($k == 0 ? 'RZ' : $acc);
                }
            }
            $ops[7] =~ s/^--:-:-:-:1/--:-:-:-:6/ if $k == 3;
            $x .= $_ for @ops;
        }
        return $x;
    };
    my $combine = sub {
        my ($accn, $dst) = @_;
        my $x = '';
        $x .= sprintf "--:-:-:-:6      IADD %s%d, %s%d, %s%d;\n", $accn, $_, $accn, $_, $accn, $_ + 4 for 0..3;
        $x .= sprintf "--:-:-:-:6      SHR %s%d, %s%d, 0x%x;\n", $accn, $_, $accn, $_, 4*$_ for 1..3;
        $x .= sprintf "--:-:-:-:6      IADD %s, %s0, %s1;\n", $dst, $accn, $accn;
        $x .= sprintf "--:-:-:-:6      IADD %s2, %s2, %s3;\n", $accn, $accn, $accn;
        $x .= sprintf "--:-:-:-:6      IADD %s, %s, %s2;\n", $dst, $dst, $accn;
        return $x;
    };
    my $o = '';
    $o .= sprintf "%s:-:4:-:1      F2F.F32.F16 dd, hdr00;\n", $wait;
    $o .= "--:-:4:-:2      F2F.F32.F16 dm, hdr00.H1;\n";
    $o .= "--:-:-:-:1      SHR.U32 u1, hdr01, shj;\n";
    $o .= "--:-:-:-:1      SHR.U32 u2, hdr02, shj;\n";
    $o .= "--:-:-:-:6      SHR.U32 u3, hdr03, shj;\n";
    $o .= "--:-:-:-:1 \@!$jp LOP32I.AND sc, u1, 0x3f;\n";
    $o .= "--:-:-:-:1 \@!$jp LOP32I.AND mn, u2, 0x3f;\n";
    $o .= "--:-:-:-:1  \@$jp LOP32I.AND sc, u3, 0xf;\n";
    $o .= "--:-:-:-:1  \@$jp SHR.U32 mn, u3, 0x4;\n";
    $o .= "--:-:-:-:1  \@$jp SHR.U32 u1, u1, 0x6;\n";
    $o .= "--:-:-:-:6  \@$jp SHR.U32 u2, u2, 0x6;\n";
    $o .= "--:-:-:-:1  \@$jp LOP32I.AND u1, u1, 0x3;\n";
    $o .= "--:-:-:-:1  \@$jp LOP32I.AND u2, u2, 0x3;\n";
    $o .= "--:-:-:-:6  \@$jp LOP32I.AND mn, mn, 0xf;\n";
    $o .= "--:-:-:-:1  \@$jp ISCADD sc, u1, sc, 0x4;\n";
    $o .= "--:-:-:-:6  \@$jp ISCADD mn, u2, mn, 0x4;\n";
    $o .= "--:-:3:-:1      I2F.F32.S32 scf, sc;\n";
    $o .= "--:-:-:-:6      XMAD.U16.S16 t2, mn, sum0, RZ;\n";
    $o .= "--:-:3:-:2      I2F.F32.S32 t2f, t2;\n";
    $o .= $stream->([map { sprintf "q%02d", $_ } 0..3], sub { sprintf "0x%08x", (0x000F000F << (4*$_[0])) & 0xffffffff },
                    'acc', sub { my ($k, $v, $hh) = @_; 8*$k + 4*$hh + $v });
    if ($q5) {
        $o .= "--:-:-:-:1      SHR.U32 hs1, qh0, 0x1;\n";
        $o .= "--:-:-:-:1      SHR.U32 hs2, qh0, 0x2;\n";
        $o .= "--:-:-:-:6      SHR.U32 hs3, qh0, 0x3;\n";
        $o .= $stream->(['qh0', 'hs1', 'hs2', 'hs3'], sub { sprintf "0x%08x", (0x00010001 << (4*$_[0])) & 0xffffffff },
                        'acch', sub { my ($k, $v, $hh) = @_; 16*$hh + 4*$v + $k });
    }
    $o .= $combine->('acc', 'dot');
    if ($q5) {
        $o .= $combine->('acch', 'dh');
        $o .= "--:-:-:-:6      SHL dh, dh, 0x4;\n";
        $o .= "--:-:-:-:6      IADD dot, dot, dh;\n";
    }
    $o .= "--:-:3:-:6      I2F.F32.S32 dotf, dot;\n";
    $o .= "0c:-:-:-:6      FMUL t1, dotf, scf;\n";
    $o .= "--:-:-:-:6      FMUL u, dd, t1;\n";
    $o .= "--:-:-:-:6      FFMA u, dm, -t2f, u;\n" unless $ENV{GEMV_NOMIN};
    $o .= "--:-:-:-:6      FFMA accf, as0, u, accf;\n";
    return $o;
}
1;
