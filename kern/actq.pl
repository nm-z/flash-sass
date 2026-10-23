# actq.pl: emitter for the per-32 activation quantiser (kern/actq.sass) and its
# SiLU-gated variant (kern/ffnact.sass: x = SiLU(g) * u before quantising, for
# the MoE expert activations). One warp per vector; CTA b handles vector b
# (pointers offset by b * vector). Bit-exact twins: quant_act_gpu32, and
# silu_gpu(g) * u followed by quant_act_gpu32.
#
#   gmax = max(max_{32} |x|, 2^-100)
#   r    = rcp_rn(gmax)          (MUFU.RCP + one Newton step = IEEE 1/gmax)
#   inv  = r * 127 ; q_i = F2I.RN(x_i * inv) ; as = gmax * (1/127)
#
# Lane l owns the 4-element chunks at 4 (l + 32 j); a 32-group is 8 consecutive
# lanes of one chunk, so the group max and the sums are 3-round SHFL butterflies
# (xor 1, 2, 4). Everything is done in rounds over the U chunks of an iteration
# (U instructions between dependent ops; for U < 6 the last instruction of a round
# carries the remaining stall). Barriers: 1 = loads, 2 = store reads, 3 = MUFU/F2I,
# 5 = SHFL, 6 = S2R.
#
#   actq_regmap($n, $silu)   the <REGISTER_MAPPING> text
#   actq_code($n, $silu)     the kernel body

use strict;
use warnings;
require './fmath.pl';

sub actq_shape {
    my ($n) = @_;
    my $C = $n / 128;
    my $U = $ENV{ACTQ_U} || ($C <= 20 ? $C : 16);      # ACTQ_U caps the unroll (the die build keeps K=2560 under 255 registers)
    die "actq: U=$U does not divide C=$C" if $C % $U;
    return ($C, $U, $C / $U);
}

sub actq_regmap {
    my ($n, $silu, $batch) = @_;
    my ($C, $U, $NI) = actq_shape($n);
    my $o = "<REGISTER_MAPPING>\n";
    my $r = 0;
    $o .= sprintf "    %d-%d : x<00-%02d>\n", $r, $r + 4*$U - 1, 4*$U - 1; $r += 4*$U;
    $o .= sprintf "    %d-%d : pk<00-%02d>\n", $r, $r + 2*$U - 1, 2*$U - 1; $r += 2*$U;
    for my $name (qw(s16_ s32_ g r tt)) { $o .= sprintf "    %d-%d : %s<00-%02d>\n", $r, $r + $U - 1, $name, $U - 1; $r += $U; }
    for my $name (qw(mneg1 c127 tid lane it t bid t2)) { $o .= sprintf "    %d : %s\n", $r++, $name; }
    if ($silu) {
        $r = ($r + 3) & ~3;
        $o .= sprintf "    %d-%d : u<00-%02d>\n", $r, $r + 4*$U - 1, 4*$U - 1; $r += 4*$U;
        for my $name (qw(log2e ln2hi ln2lo c720 c120 c24 c6 chalf cone)) { $o .= sprintf "    %d : %s\n", $r++, $name; }
        $o .= sprintf "    %d-%d : ft<00-09>\n", $r, $r + 9; $r += 10;
    }
    $r += $r & 1;
    my @ptr = qw(xp qp hp gp ap);
    push @ptr, 'up' if $silu;
    for my $p (@ptr) { $o .= sprintf "    %d,%d : %s<0-1>\n", $r, $r + 1, $p; $r += 2; }
    if ($batch) {
        die 'ACTQ batch currently supports plain K256 only' if $silu || $n != 256;
        $o .= sprintf "    %d : position\n    %d : descriptor\n",$r,$r+1;$r+=2;
        for my $name (qw(X Q S32 S16 AS)) {
            $o .= sprintf "    %d,%d : param_%s<0-1>\n",$r,$r+1,$name;$r+=2;
        }
    }
    $o .= "</REGISTER_MAPPING>\n";
    return $o;
}

sub actq_code {
    my ($n, $silu, $batch) = @_;
    my ($C, $U, $NI) = actq_shape($n);
    my $rs = $U >= 6 ? '1' : (7 - $U);     # stall on the last instruction of a round
    my %FC = map { $_ => $_ } qw(log2e ln2hi ln2lo c720 c120 c24 c6 chalf cone mneg1);
    my $o = '';
    # a round: $f->($u) returns one instruction line with WW (wait) and SS (stall) placeholders
    my $round = sub {
        my ($f, $wait) = @_;
        my $x = '';
        for my $u (0..$U-1) {
            my $line = $f->($u);
            my $w = ($u == 0 && $wait) ? $wait : '--';
            my $s = $u == $U-1 ? $rs : '1';
            $s = 2 if $u == $U-1 && $s eq '1' && $line =~ /\b(SHFL|MUFU|F2I|LDG)\b/;   # variable-latency op: stall >= 2 before the dependent wait
            $line =~ s/^WW/$w/; $line =~ s/:SS /:$s /;
            $x .= $line;
        }
        return $x;
    };
    # pointer = base + vector offset (bid * bytes per vector) + lane part
    my $ptr = sub {
        my ($p, $lanepart, $pervec, $base) = @_;
        my $x = $lanepart;
        $x .= sprintf "--:-:-:-:6      XMAD t2, bid, 0x%x, RZ;\n", $pervec;
        $x .= "--:-:-:-:6      IADD t, t, t2;\n";
        $x .= sprintf "--:-:-:-:6      IADD %s0.CC, t, param_%s0;\n", $p, $base;
        $x .= sprintf "--:-:-:-:6      IADD.X %s1, RZ, param_%s1;\n", $p, $base;
        return $x;
    };
    $o .= "--:-:6:-:1      S2R tid, SR_TID.X;\n";
    $o .= "--:-:6:-:1      S2R bid, SR_CTAID.X;\n";
    if ($batch) {
        die 'ACTQ batch currently supports plain K256 only' if $silu || $n != 256;
        # CTA unit remains the original256-element vector; its eight warps
        # select eight independent position descriptors. The arithmetic below
        # is unchanged and sees the original warp lane and vector index.
        $o .= "20:-:-:-:6      SHR.U32 position, tid, 0x5;\n--:-:-:-:6      LOP32I.AND tid, tid, 0x1f;\n";
        $o .= "--:-:-:-:6      ISETP.EQ.U32.AND P6, PT, position, RZ, PT;\n--:-:-:-:6      SHL descriptor, position, 0x7;\n--:-:-:-:6      MOV32I t2, 0xbf00;\n--:-:-:-:6      IADD descriptor, t2, -descriptor;\n--:-:-:-:6  \@P6 MOV32I descriptor, 0xbf80;\n";
        my @names=qw(X Q S32 S16 AS);
        for my $j (0..4) {
            $o .= sprintf "--:-:1:-:%d      LDS.64 param_%s0, [descriptor+0x%x];\n",$j==4?2:1,$names[$j],8+8*$j;
        }
        $o .= "01:-:-:-:6      NOP;\n";
    }
    $o .= "--:-:-:-:1      MOV32I mneg1, 0xbf800000;\n";
    $o .= "--:-:-:-:1      MOV32I c127, 0x3c010204;\n";          # 1/127
    $o .= fm_consts({ map { $_ => $_ } qw(log2e ln2hi ln2lo c720 c120 c24 c6 chalf cone) }) if $silu;
    $o .= "--:-:-:-:6      MOV it, RZ;\n";
    $o .= "20:-:-:-:6      ISETP.GE.U32.AND P0, PT, tid, 0x20, PT;\n";
    $o .= "--:-:-:-:6      LOP32I.AND lane, tid, 0x1f;\n";
    $o .= "--:-:-:-:6      LOP32I.AND t, lane, 0x3;\n";
    $o .= "--:-:-:-:6      ISETP.EQ.U32.AND P2, PT, t, RZ, PT;\n";             # lane % 4 == 0: writes s16
    $o .= "--:-:-:-:6  \@P0 EXIT;\n";
    $o .= "--:-:-:-:6      LOP32I.AND t, lane, 0x7;\n";
    $o .= "--:-:-:-:6      ISETP.EQ.U32.AND P3, PT, t, RZ, PT;\n";             # lane % 8 == 0: writes s32 and as
    # xp = X + 16 lane ; qp = Q + 8 lane ; hp = S16 + 4 (lane >> 2) ; gp = S32 + 4 (lane >> 3) ; ap = AS + 4 (lane >> 3)
    $o .= $ptr->('xp', "--:-:-:-:6      SHL t, lane, 0x4;\n", 4 * $n, 'X');
    $o .= $ptr->('up', "--:-:-:-:6      SHL t, lane, 0x4;\n", 4 * $n, 'U') if $silu;
    $o .= $ptr->('qp', "--:-:-:-:6      SHL t, lane, 0x3;\n", 2 * $n, 'Q');
    $o .= $ptr->('hp', "--:-:-:-:6      SHR.U32 t, lane, 0x2;\n--:-:-:-:6      SHL t, t, 0x2;\n", $n / 4, 'S16');
    $o .= $ptr->('gp', "--:-:-:-:6      SHR.U32 t, lane, 0x3;\n--:-:-:-:6      SHL t, t, 0x2;\n", $n / 8, 'S32');
    $o .= $ptr->('ap', "--:-:-:-:6      SHR.U32 t, lane, 0x3;\n--:-:-:-:6      SHL t, t, 0x2;\n", $n / 8, 'AS');

    $o .= "\nLOOP:\n";
    for my $u (0..$U-1) {
        $o .= sprintf "%s:-:1:-:%s      LDG.E.CI.128 x%02d, [xp+0x%x];\n", $u == 0 ? '02' : '--', ($u == $U-1 && !$silu) ? '2' : '1', 4*$u, 512*$u;
    }
    if ($silu) {
        for my $u (0..$U-1) {
            $o .= sprintf "--:-:1:-:%s      LDG.E.CI.128 u%02d, [up+0x%x];\n", $u == $U-1 ? '2' : '1', 4*$u, 512*$u;
        }
    }
    $o .= "--:-:-:-:6      IADD it, it, 0x1;\n";
    $o .= sprintf "--:-:-:-:6      ISETP.LT.U32.AND P4, PT, it, 0x%x, PT;\n", $NI;
    if ($silu) {
        # x = SiLU(x) * u, two interleaved chains (fm_zip lowers the stalls)
        for (my $i = 0; $i < 4*$U; $i += 2) {
            my @ch;
            for my $k (0, 1) {
                my $e = sprintf "x%02d", $i + $k;
                my $c = fm_silu($e, $e, map({ sprintf "ft%02d", 5*$k + $_ } 0..4), \%FC);
                $c .= sprintf "--:-:-:-:6      FMUL %s, %s, u%02d;\n", $e, $e, $i + $k;
                push @ch, $c;
            }
            my $z = fm_zip(@ch);
            $z =~ s/^--/01/ if $i == 0;      # first instruction waits for the loads
            $o .= $z;
        }
    }
    # lane max of the chunk
    $o .= $round->(sub { my $u = shift; sprintf "WW:-:-:-:SS      FMNMX g%02d, |x%02d|, |x%02d|, !PT;\n", $u, 4*$u, 4*$u + 1 }, '01');
    $o .= $round->(sub { my $u = shift; sprintf "WW:-:-:-:SS      FMNMX g%02d, g%02d, |x%02d|, !PT;\n", $u, $u, 4*$u + 2 });
    $o .= $round->(sub { my $u = shift; sprintf "WW:-:-:-:SS      FMNMX g%02d, g%02d, |x%02d|, !PT;\n", $u, $u, 4*$u + 3 });
    # group max over 8 lanes
    for my $k (1, 2, 4) {
        $o .= $round->(sub { my $u = shift; sprintf "WW:-:5:-:SS      SHFL.BFLY PT, tt%02d, g%02d, 0x%x, 0x1f;\n", $u, $u, $k });
        $o .= $round->(sub { my $u = shift; sprintf "WW:-:-:-:SS      FMNMX g%02d, g%02d, tt%02d, !PT;\n", $u, $u, $u }, '10');
    }
    # clamp, reciprocal, inv and as
    $o .= $round->(sub { my $u = shift; sprintf "WW:-:-:-:SS      FMNMX g%02d, g%02d, 7.8886090522101180541e-31, !PT;\n", $u, $u });
    $o .= $round->(sub { my $u = shift; sprintf "WW:-:3:-:SS      MUFU.RCP r%02d, g%02d;\n", $u, $u });
    $o .= $round->(sub { my $u = shift; sprintf "WW:-:-:-:SS      FFMA tt%02d, g%02d, r%02d, mneg1;\n", $u, $u, $u }, '04');
    $o .= $round->(sub { my $u = shift; sprintf "WW:-:-:-:SS      FADD.FTZ tt%02d, -tt%02d, -RZ;\n", $u, $u });
    $o .= $round->(sub { my $u = shift; sprintf "WW:-:-:-:SS      FFMA r%02d, r%02d, tt%02d, r%02d;\n", $u, $u, $u, $u });
    $o .= $round->(sub { my $u = shift; sprintf "WW:-:-:-:SS      FMUL r%02d, r%02d, 127;\n", $u, $u });
    $o .= $round->(sub { my $u = shift; sprintf "WW:-:-:-:SS      FMUL g%02d, g%02d, c127;\n", $u, $u });
    $o .= $round->(sub { my $u = shift; sprintf "WW:2:-:-:SS  \@P3 STG.E [ap+0x%x], g%02d;\n", 16*$u, $u });
    # codes
    for my $k (0..3) {
        $o .= $round->(sub { my $u = shift; sprintf "WW:-:-:-:SS      FMUL x%02d, x%02d, r%02d;\n", 4*$u + $k, 4*$u + $k, $u });
    }
    for my $k (0..3) {
        $o .= $round->(sub { my $u = shift; sprintf "WW:-:3:-:SS      F2I.S32.F32 x%02d, x%02d;\n", 4*$u + $k, 4*$u + $k });
    }
    $o .= $round->(sub { my $u = shift; sprintf "WW:-:-:-:SS      IADD tt%02d, x%02d, x%02d;\n", $u, 4*$u, 4*$u + 1 }, '04');
    $o .= $round->(sub { my $u = shift; sprintf "WW:-:-:-:SS      IADD r%02d, x%02d, x%02d;\n", $u, 4*$u + 2, 4*$u + 3 });
    $o .= $round->(sub { my $u = shift; sprintf "WW:-:-:-:SS      PRMT pk%02d, x%02d, 0x5410, x%02d;\n", 2*$u, 4*$u, 4*$u + 1 });
    $o .= $round->(sub { my $u = shift; sprintf "WW:-:-:-:SS      PRMT pk%02d, x%02d, 0x5410, x%02d;\n", 2*$u + 1, 4*$u + 2, 4*$u + 3 });
    $o .= $round->(sub { my $u = shift; sprintf "WW:-:-:-:SS      IADD tt%02d, tt%02d, r%02d;\n", $u, $u, $u });
    $o .= $round->(sub { my $u = shift; sprintf "WW:2:-:-:SS      STG.E.64 [qp+0x%x], pk%02d;\n", 256*$u, 2*$u });
    # sums: 16 = 4 lanes (xor 1, 2), 32 = 8 lanes (xor 4)
    $o .= $round->(sub { my $u = shift; sprintf "WW:-:5:-:SS      SHFL.BFLY PT, r%02d, tt%02d, 0x1, 0x1f;\n", $u, $u });
    $o .= $round->(sub { my $u = shift; sprintf "WW:-:-:-:SS      IADD s16_%02d, tt%02d, r%02d;\n", $u, $u, $u }, '10');
    $o .= $round->(sub { my $u = shift; sprintf "WW:-:5:-:SS      SHFL.BFLY PT, r%02d, s16_%02d, 0x2, 0x1f;\n", $u, $u });
    $o .= $round->(sub { my $u = shift; sprintf "WW:-:-:-:SS      IADD s16_%02d, s16_%02d, r%02d;\n", $u, $u, $u }, '10');
    $o .= $round->(sub { my $u = shift; sprintf "WW:2:-:-:SS  \@P2 STG.E [hp+0x%x], s16_%02d;\n", 32*$u, $u });
    $o .= $round->(sub { my $u = shift; sprintf "WW:-:5:-:SS      SHFL.BFLY PT, r%02d, s16_%02d, 0x4, 0x1f;\n", $u, $u });
    $o .= $round->(sub { my $u = shift; sprintf "WW:-:-:-:SS      IADD s32_%02d, s16_%02d, r%02d;\n", $u, $u, $u }, '10');
    $o .= $round->(sub { my $u = shift; sprintf "WW:2:-:-:SS  \@P3 STG.E [gp+0x%x], s32_%02d;\n", 16*$u, $u });
    # next iteration (the stores must have read their registers: barrier 2)
    $o .= sprintf "02:-:-:-:6      IADD xp0.CC, xp0, 0x%x;\n", 512*$U;
    $o .= "--:-:-:-:6      IADD.X xp1, xp1, RZ;\n";
    if ($silu) {
        $o .= sprintf "--:-:-:-:6      IADD up0.CC, up0, 0x%x;\n", 512*$U;
        $o .= "--:-:-:-:6      IADD.X up1, up1, RZ;\n";
    }
    $o .= sprintf "--:-:-:-:6      IADD qp0.CC, qp0, 0x%x;\n", 256*$U;
    $o .= "--:-:-:-:6      IADD.X qp1, qp1, RZ;\n";
    $o .= sprintf "--:-:-:-:6      IADD hp0.CC, hp0, 0x%x;\n", 32*$U;
    $o .= "--:-:-:-:6      IADD.X hp1, hp1, RZ;\n";
    $o .= sprintf "--:-:-:-:6      IADD gp0.CC, gp0, 0x%x;\n", 16*$U;
    $o .= "--:-:-:-:6      IADD.X gp1, gp1, RZ;\n";
    $o .= sprintf "--:-:-:-:6      IADD ap0.CC, ap0, 0x%x;\n", 16*$U;
    $o .= "--:-:-:-:6      IADD.X ap1, ap1, RZ;\n";
    $o .= "--:-:-:-:6  \@P4 BRA LOOP;\n" if $NI > 1;
    $o .= "--:-:-:-:f      EXIT;\n";
    return $o;
}

1;
