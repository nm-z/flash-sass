# fmath.pl: SASS emitters for the transcendentals, built only from IEEE fp32
# FADD/FMUL/FFMA, FMNMX, F2I and the rcp_rn sequence, so ref/fmath.c is a
# bit-exact twin. Loaded from a generator's <CODE> block with
#     require './fmath.pl';
# Every emitter returns SASS text whose instructions carry stall 6 (every op
# depends on the previous one); callers interleave independent work if they
# care. Constants live in registers named by the %C hash (see fm_consts).
#
#   fm_consts(\%C)                        MOV32I the constants into their registers
#   fm_exp($dst, $x, $t1, $t2, $t3, \%C)  dst = exp(clamp(x, -87, 87)); x is left clamped
#   fm_rcp($dst, $x, $t1, \%C)            dst = 1/x (IEEE for normal x < 2^125)
#   fm_sigmoid($dst, $x, $t1, $t2, $t3, $t4, \%C)
#   fm_silu($dst, $x, $t1, $t2, $t3, $t4, $t5, \%C)   (x must differ from the temps)
#   fm_log1p($dst, $e, $t1, $t2, $t3, \%C)   log(1 + e), e in [0, 1]
#   fm_softplus($dst, $x, $t1, $t2, $t3, $t4, $t5, \%C)

use strict;
use warnings;

our %FM_CONST = (
    log2e => 0x3fb8aa3b, ln2hi => 0x3f317200, ln2lo => 0x35bfbe8e,
    c720 => 0x3ab60b61, c120 => 0x3c088889, c24 => 0x3d2aaaab, c6 => 0x3e2aaaab,
    c11 => 0x3dba2e8c, c9 => 0x3de38e39, c7 => 0x3e124925, c5 => 0x3e4ccccd, c3 => 0x3eaaaaab,
    mneg1 => 0xbf800000, chalf => 0x3f000000, cone => 0x3f800000,
);

sub fm_consts {
    my ($C) = @_;
    my $o = '';
    for my $k (sort keys %$C) {
        die "fm_consts: unknown constant $k" unless exists $FM_CONST{$k};
        $o .= sprintf "--:-:-:-:1      MOV32I %s, 0x%08x;\n", $C->{$k}, $FM_CONST{$k};
    }
    return $o;
}

# dst = 1/x: MUFU.RCP + one Newton step (needs $C->{mneg1})
sub fm_rcp {
    my ($dst, $x, $t1, $C) = @_;
    my $o = '';
    $o .= sprintf "--:-:3:-:2      MUFU.RCP %s, %s;\n", $dst, $x;
    $o .= sprintf "04:-:-:-:6      FFMA %s, %s, %s, %s;\n", $t1, $x, $dst, $C->{mneg1};
    $o .= sprintf "--:-:-:-:6      FADD.FTZ %s, -%s, -RZ;\n", $t1, $t1;
    $o .= sprintf "--:-:-:-:6      FFMA %s, %s, %s, %s;\n", $dst, $dst, $t1, $dst;
    return $o;
}

# dst = exp(x) with x clamped to [-87, 88] in place. Temps: t1 (t/p), t2 (n/ni), t3 (r).
sub fm_exp {
    my ($dst, $x, $t1, $t2, $t3, $C) = @_;
    my $o = '';
    $o .= sprintf "--:-:-:-:6      FMNMX %s, %s, 87, PT;\n", $x, $x;            # min
    $o .= sprintf "--:-:-:-:6      FMNMX %s, %s, -87, !PT;\n", $x, $x;          # max
    $o .= sprintf "--:-:-:-:6      FMUL %s, %s, %s;\n", $t1, $x, $C->{log2e};
    $o .= sprintf "--:-:-:-:6      FADD %s, %s, 12582912;\n", $t2, $t1;
    $o .= sprintf "--:-:-:-:6      FADD %s, %s, -12582912;\n", $t2, $t2;        # n = rint(t)
    $o .= sprintf "--:-:-:-:6      FFMA %s, %s, -%s, %s;\n", $t3, $t2, $C->{ln2hi}, $x;
    $o .= sprintf "--:-:-:-:6      FFMA %s, %s, -%s, %s;\n", $t3, $t2, $C->{ln2lo}, $t3;
    $o .= sprintf "--:-:-:-:6      FFMA %s, %s, %s, %s;\n", $t1, $C->{c720}, $t3, $C->{c120};
    $o .= sprintf "--:-:-:-:6      FFMA %s, %s, %s, %s;\n", $t1, $t1, $t3, $C->{c24};
    $o .= sprintf "--:-:-:-:6      FFMA %s, %s, %s, %s;\n", $t1, $t1, $t3, $C->{c6};
    $o .= sprintf "--:-:-:-:6      FFMA %s, %s, %s, %s;\n", $t1, $t1, $t3, $C->{chalf};
    $o .= sprintf "--:-:-:-:6      FFMA %s, %s, %s, %s;\n", $t1, $t1, $t3, $C->{cone};
    $o .= sprintf "--:-:-:-:6      FFMA %s, %s, %s, %s;\n", $t1, $t1, $t3, $C->{cone};
    $o .= sprintf "--:-:3:-:2      F2I.S32.F32 %s, %s;\n", $t2, $t2;
    $o .= sprintf "04:-:-:-:6      IADD %s, %s, 0x7f;\n", $t2, $t2;
    $o .= sprintf "--:-:-:-:6      SHL %s, %s, 0x17;\n", $t2, $t2;              # bits of 2^n
    $o .= sprintf "--:-:-:-:6      FMUL %s, %s, %s;\n", $dst, $t1, $t2;
    return $o;
}

# dst = sigmoid(x) = 1 / (1 + exp(-x)); x is preserved. Temps t1..t4.
sub fm_sigmoid {
    my ($dst, $x, $t1, $t2, $t3, $t4, $C) = @_;
    my $o = sprintf "--:-:-:-:6      FADD %s, -%s, RZ;\n", $t4, $x;
    $o .= fm_exp($t4, $t4, $t1, $t2, $t3, $C);
    $o .= sprintf "--:-:-:-:6      FADD %s, %s, 1;\n", $t4, $t4;
    $o .= fm_rcp($dst, $t4, $t1, $C);
    return $o;
}

sub fm_silu {
    my ($dst, $x, $t1, $t2, $t3, $t4, $t5, $C) = @_;
    my $o = fm_sigmoid($t5, $x, $t1, $t2, $t3, $t4, $C);
    $o .= sprintf "--:-:-:-:6      FMUL %s, %s, %s;\n", $dst, $x, $t5;
    return $o;
}

# dst = log(1 + e), e in [0, 1]: u = e / (2 + e), 2u (1 + u^2/3 + ... + u^10/11). Temps t1..t3.
sub fm_log1p {
    my ($dst, $e, $t1, $t2, $t3, $C) = @_;
    my $o = '';
    $o .= sprintf "--:-:-:-:6      FADD %s, %s, 2;\n", $t1, $e;
    $o .= fm_rcp($t2, $t1, $t3, $C);
    $o .= sprintf "--:-:-:-:6      FMUL %s, %s, %s;\n", $t1, $e, $t2;             # u
    $o .= sprintf "--:-:-:-:6      FMUL %s, %s, %s;\n", $t2, $t1, $t1;            # u^2
    $o .= sprintf "--:-:-:-:6      FFMA %s, %s, %s, %s;\n", $t3, $C->{c11}, $t2, $C->{c9};
    $o .= sprintf "--:-:-:-:6      FFMA %s, %s, %s, %s;\n", $t3, $t3, $t2, $C->{c7};
    $o .= sprintf "--:-:-:-:6      FFMA %s, %s, %s, %s;\n", $t3, $t3, $t2, $C->{c5};
    $o .= sprintf "--:-:-:-:6      FFMA %s, %s, %s, %s;\n", $t3, $t3, $t2, $C->{c3};
    $o .= sprintf "--:-:-:-:6      FFMA %s, %s, %s, %s;\n", $t3, $t3, $t2, $C->{cone};
    $o .= sprintf "--:-:-:-:6      FMUL %s, %s, 2;\n", $t1, $t1;
    $o .= sprintf "--:-:-:-:6      FMUL %s, %s, %s;\n", $dst, $t1, $t3;
    return $o;
}

# dst = softplus(x) = max(x, 0) + log1p(exp(-|x|)); x preserved. Temps t1..t5.
sub fm_softplus {
    my ($dst, $x, $t1, $t2, $t3, $t4, $t5, $C) = @_;
    my $o = sprintf "--:-:-:-:6      FADD %s, -|%s|, RZ;\n", $t4, $x;
    $o .= fm_exp($t4, $t4, $t1, $t2, $t3, $C);
    $o .= fm_log1p($t5, $t4, $t1, $t2, $t3, $C);
    $o .= sprintf "--:-:-:-:6      FMNMX %s, %s, RZ, !PT;\n", $t1, $x;
    $o .= sprintf "--:-:-:-:6      FADD %s, %s, %s;\n", $dst, $t1, $t5;
    return $o;
}

1;
