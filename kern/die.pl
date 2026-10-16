#!/usr/bin/perl
# die.pl: assembles die.sass, the persistent per-die kernel, out of the standalone
# op generators (kern/*.sass). Run from kern/:   perl die.pl > die.sass
#
# The kernel runs a program of phases (128-byte descriptors in `prog`): for each
# phase every CTA copies the descriptor into shared memory, dispatches on its op,
# loops its unit over the op's virtual units and then passes a grid barrier.
# Framework registers R0-R15: tid lane warp cta vid vtid units pidx nph prog<0-1>
# ctl<0-1> op fw<0-1>. Every op keeps its own code; this script
#   - evaluates its <CODE> blocks the way MaxAs does, with the op's environment,
#   - moves its registers up by 16 and prefixes its register names and labels,
#   - turns its param_* constants into registers loaded (LDS) from the descriptor
#     in shared memory (DESC_OFF), gridDimX becoming the unit count,
#   - replaces S2R SR_TID.X / SR_CTAID.X with the virtual thread / unit ids,
#   - replaces EXIT with a branch to the op's unit-loop end.
# Descriptor words: [0] op, [1] units, [2 + (addr - 0x140) / 4] = the op's
# parameter words (its CONSTANT_MAPPING layout). Unit kinds: cta (256 threads,
# vid = cta, cta + 16, ...) and warp (32 threads, vid = 8 cta + warp, + 128, ...,
# vtid = lane). Ctl: u32 ctr @0, u32 gen @4 (grid barrier state).

use strict;
use warnings;

my $BASE = 16;             # first op register
my $NCTA = 16;             # CTAs per die (1 per SM)
my $DESC_OFF = 0xbf80;     # descriptor in shared memory (ops use smem from 0)

# opcode = index. tag, file, unit kind, environment for the generator
my @OPS = (
    # tag           file                unit    env
    [ 'embrow',     'embrow.sass',      'cta',  {} ],
    [ 'plegate',    'plegate.sass',     'cta',  {} ],
    [ 'argmax',     'argmax.sass',      'cta',  {} ],
    [ 'actq2560',   'actq.sass',        'warp', { GEMV_K => 2560, ACTQ_U => 10 } ],
    [ 'ffnact',     'ffnact.sass',      'warp', { GEMV_K => 640 } ],
    [ 'actq6144',   'actq.sass',        'warp', { GEMV_K => 6144 } ],
    [ 'actq10240',  'actq.sass',        'warp', { GEMV_K => 10240 } ],
    [ 'rms2560',    'rmsnorm.sass',     'warp', { GEMV_K => 2560 } ],
    [ 'rms10240',   'rmsnorm.sass',     'warp', { GEMV_K => 10240 } ],
    # normq (RMSNorm + quantiser, 228 registers) does not fit beside the framework; nothing in the v1 block needs it
    [ 'hcsilu',     'hcops.sass',       'cta',  { HC_OP => 'silu' } ],
    [ 'hcmix',      'hcops.sass',       'cta',  { HC_OP => 'mix' } ],
    [ 'hcwrite',    'hcops.sass',       'cta',  { HC_OP => 'write' } ],
    [ 'conv1d',     'conv1d.sass',      'cta',  {} ],
    [ 'dnstep',     'dnstep.sass',      'cta',  {} ],
    [ 'dnnorm',     'dnnorm.sass',      'warp', {} ],
    [ 'router',     'router.sass',      'warp', {} ],
    [ 'attnqk',     'attnqk.sass',      'warp', {} ],
    [ 'attnpart',   'attnpart.sass',    'cta',  {} ],
    [ 'attnmerge',  'attnmerge.sass',   'cta',  {} ],
    [ 'plerows',    'plerows.sass',     'warp', {} ],
    [ 'pleconv',    'pleconv.sass',     'cta',  {} ],
    [ 'q2k2560',    'gemv_q2_0.sass',   'cta',  { GEMV_K => 2560 } ],
    [ 'q2k640',     'gemv_q2_0.sass',   'cta',  { GEMV_K => 640 } ],
    [ 'q2moeg',     'gemv_q2_0.sass',   'cta',  { GEMV_K => 2560, GEMV_MOE => 'gather' } ],
    [ 'q2moed',     'gemv_q2_0.sass',   'cta',  { GEMV_K => 640, GEMV_MOE => 'down' } ],
    [ 'q2acc640',   'gemv_q2_0.sass',   'cta',  { GEMV_K => 640, GEMV_ACC => 1 } ],
    [ 'bf16bias',   'gemv_bf16.sass',   'cta',  { GEMV_K => 2560, GEMV_BIAS => 1 } ],
    [ 'bf16k2560',  'gemv_bf16.sass',   'cta',  { GEMV_K => 2560 } ],
    [ 'bf16k320',   'gemv_bf16.sass',   'cta',  { GEMV_K => 320 } ],
    [ 'bf16hc',     'gemv_bf16.sass',   'cta',  { GEMV_K => 2560, GEMV_SLICES => 4 } ],
    [ 'q4kk2560',   'gemv_q4_k.sass',   'cta',  { GEMV_K => 2560 } ],
    [ 'q4kk6144',   'gemv_q4_k.sass',   'cta',  { GEMV_K => 6144 } ],
    [ 'q5kk2560',   'gemv_q4_k.sass',   'cta',  { GEMV_K => 2560, GEMV_FMT => 'q5_k' } ],
    [ 'q5kk6144',   'gemv_q4_k.sass',   'cta',  { GEMV_K => 6144, GEMV_FMT => 'q5_k' } ],
    [ 'q3kk2560',   'gemv_q3_k.sass',   'cta',  { GEMV_K => 2560 } ],
    [ 'q3kk6144',   'gemv_q3_k.sass',   'cta',  { GEMV_K => 6144 } ],
    [ 'q6kk2560',   'gemv_q6_k.sass',   'cta',  { GEMV_K => 2560 } ],
    # q6kk6144 (236 registers + 13 params) does not fit yet: attn_output of blk 11, 27, 31 — needs a register diet
    [ 'iq4k2560',   'gemv_iq4_xs.sass', 'cta',  { GEMV_K => 2560 } ],
    [ 'iq4k6144',   'gemv_iq4_xs.sass', 'cta',  { GEMV_K => 6144 } ],
    [ 'b32q40',     'gemv_b32.sass',    'cta',  { GEMV_K => 640, GEMV_FMT => 'q4_0' } ],
    [ 'b32q50',     'gemv_b32.sass',    'cta',  { GEMV_K => 640, GEMV_FMT => 'q5_0' } ],
    [ 'b32q80',     'gemv_b32.sass',    'cta',  { GEMV_K => 640, GEMV_FMT => 'q8_0' } ],
    [ 'b32iq4nl',   'gemv_b32.sass',    'cta',  { GEMV_K => 640, GEMV_FMT => 'iq4_nl' } ],
    [ 'b32q40acc',  'gemv_b32.sass',    'cta',  { GEMV_K => 640, GEMV_FMT => 'q4_0', GEMV_ACC => 1 } ],
    [ 'b32q50acc',  'gemv_b32.sass',    'cta',  { GEMV_K => 640, GEMV_FMT => 'q5_0', GEMV_ACC => 1 } ],
    [ 'b32q80acc',  'gemv_b32.sass',    'cta',  { GEMV_K => 640, GEMV_FMT => 'q8_0', GEMV_ACC => 1 } ],
    [ 'b32iq4nlacc','gemv_b32.sass',    'cta',  { GEMV_K => 640, GEMV_FMT => 'iq4_nl', GEMV_ACC => 1 } ],
    # ---- the 6-die split ----
    [ 'xsend',      'xchg.sass',        'cta',  { XCHG_MODE => 'send' } ],
    [ 'xwait',      'xchg.sass',        'cta',  { XCHG_MODE => 'wait' } ],
    [ 'moesel',     'moesel.sass',      'warp', {} ],
    [ 'sumn',       'sumn.sass',        'cta',  {} ],
    [ 'q2moegn',    'gemv_q2_0.sass',   'cta',  { GEMV_K => 2560, GEMV_MOE => 'gather', GEMV_NEPTR => 1 } ],
    [ 'q2moedn',    'gemv_q2_0.sass',   'cta',  { GEMV_K => 640, GEMV_MOE => 'down', GEMV_NEPTR => 1 } ],
);

sub expand_names {                       # x<00-15> -> x00 .. x15
    my ($field) = @_;
    my @out;
    for my $tok (split /[\s,]+/, $field) {
        next unless length $tok;
        if ($tok =~ /^(\w+)<(\d+)-(\d+)>$/) {
            my ($n, $a, $b) = ($1, $2, $3);
            push @out, sprintf "%s%0*d", $n, length($a), $_ for $a .. $b;
        } else {
            push @out, $tok;
        }
    }
    return @out;
}

sub load_op {
    my ($tag, $file, $env) = @_;
    local %ENV = (%ENV, %$env);
    open my $fh, '<', $file or die "$file: $!";
    my $text = do { local $/; <$fh> };
    close $fh;
    $text =~ s/#.*$//mg;                                  # comments, as MaxAs strips them before CODE runs
    1 while $text =~ s|<CODE>(.*?)</CODE>| my $out = eval "package MaxAs::MaxAs::CODE; $1"; die "CODE in $file: $@" if $@; $out |se;
    # parameters
    my @params;
    my $has_grid = 0;
    if ($text =~ s|<CONSTANT_MAPPING>(.*?)</CONSTANT_MAPPING>||s) {
        for my $l (split /\n/, $1) {
            next unless $l =~ /^\s*(\w+)\s*:\s*c\[0x0\]\[0x([0-9a-f]+)\]/;
            my ($name, $addr) = ($1, hex $2);
            if ($addr == 0x14) { $has_grid = 1; next; }
            die "$file: parameter $name at 0x$2 outside the descriptor" if $addr < 0x140 || $addr >= 0x140 + 4 * 30;
            push @params, [ $name, 2 + ($addr - 0x140) / 4 ];
        }
    }
    # registers: keep each mapping line (templates included, so MaxAs still knows the
    # 64-bit pair base names), offset by BASE and prefixed
    my (@maplines, @names, $max);
    $max = -1;
    while ($text =~ s|<REGISTER_MAPPING>(.*?)</REGISTER_MAPPING>||s) {
        for my $l (split /\n/, $1) {
            next unless $l =~ /^\s*([\d,\-]+)\s*:\s*(.+?)\s*$/;
            my ($nums, $field) = ($1, $2);
            my @n = $nums =~ /^(\d+)-(\d+)$/ ? ($1 .. $2) : split /,/, $nums;
            my @nm = expand_names($field);
            die "$file: '$l' lists ", scalar(@n), " registers for ", scalar(@nm), " names" unless @n == @nm;
            $max = $_ > $max ? $_ : $max for @n;
            push @names, @nm;
            my @toks = grep { length } split /[\s,]+/, $field;
            for my $t (@toks) { push @names, $1 if $t =~ /^(\w+)</; }          # pair/vector base names ([xp])
            my $shifted = $nums =~ /^(\d+)-(\d+)$/ ? sprintf("%d-%d", $1 + $BASE, $2 + $BASE) : join(",", map { $_ + $BASE } split /,/, $nums);
            push @maplines, sprintf "    %s : %s\n", $shifted, join(" ", map { "${tag}_$_" } @toks);
        }
    }
    push @params, [ 'gridDimX', 1 ] if $has_grid;
    my $r = $max + 1;
    my @pregs = map { [ $r++, $_->[0] ] } @params;
    warn sprintf("%-10s %3d op registers + %2d params -> last register %d\n", $tag, $max + 1, scalar(@params), $BASE + $r - 1) if $ENV{DIE_VERBOSE};
    die "$tag: registers reach ", $BASE + $r - 1, " (> 255)" if $BASE + $r - 1 > 255;
    # labels and names to prefix
    my @labels = $text =~ /^\s*([A-Za-z_]\w*):\s*$/mg;
    my %seen;
    @names = grep { !$seen{$_}++ } (@names, map($_->[1], @pregs), @labels);
    my $alt = join '|', map { quotemeta } sort { length $b <=> length $a } @names;
    my $code = $text;
    $code =~ s/\b($alt)\b/${tag}_$1/g;
    # virtual ids, exits
    $code =~ s/^.*?S2R\s+(\w+),\s*SR_TID\.X;.*$/--:-:-:-:6      MOV $1, vtid;/mg;
    $code =~ s/^.*?S2R\s+(\w+),\s*SR_CTAID\.X;.*$/--:-:-:-:6      MOV $1, vid;/mg;
    $code =~ s/\bEXIT;/BRA ${tag}_END;/g;
    $code =~ s/^\s*\n//mg;
    # parameter prologue
    my $pro = '';
    for my $i (0 .. $#params) {
        my ($name, $word) = @{$params[$i]};
        $pro .= sprintf "--:-:1:-:%d      LDS %s_%s, [RZ+0x%x];\n", $i == $#params ? 2 : 1, $tag, $name, $DESC_OFF + 4 * $word;
    }
    $pro .= "01:-:-:-:6      NOP;\n" if @params;
    my $map = join('', @maplines) . join('', map { sprintf "    %d : %s_%s\n", $BASE + $_->[0], $tag, $_->[1] } @pregs);
    return { map => $map, code => $pro . $code };
}

# "perl die.pl header" prints the opcode header for the host
if (@ARGV && $ARGV[0] eq 'header') {
    print "/* generated by die.pl: opcodes of the persistent kernel */\n#ifndef DIEOPS_H\n#define DIEOPS_H\n";
    printf "#define OP_%s %d\n", uc($OPS[$_][0]), $_ for 0 .. $#OPS;
    printf "#define OP_COUNT %d\n#define DIE_NOPS OP_COUNT\n", scalar @OPS;
    print "static const char *const die_opname[] __attribute__((unused)) = { ", join(', ', map { "\"$_->[0]\"" } @OPS), " };\n#endif\n";
    exit 0;
}
my $maps = '';
my $body = '';
my $dispatch = '';
for my $k (0 .. $#OPS) {
    my ($tag, $file, $kind, $env) = @{$OPS[$k]};
    my $op = load_op($tag, $file, $env);
    $maps .= $op->{map};
    $dispatch .= sprintf "--:-:-:-:6      ISETP.EQ.U32.AND P6, PT, op, 0x%x, PT;\n", $k;
    $dispatch .= "--:-:-:-:6      NOP;\n" x 3;
    $dispatch .= "--:-:-:-:6  \@P6 BRA ${tag}_SETUP;\n";
    my $step = $kind eq 'cta' ? $NCTA : 8 * $NCTA;
    $body .= "\n${tag}_SETUP:\n";
    if ($kind eq 'cta') {
        $body .= "--:-:-:-:6      MOV vid, cta;\n";
        $body .= "--:-:-:-:6      MOV vtid, tid;\n";
    } else {
        $body .= "--:-:-:-:6      SHL vid, cta, 0x3;\n";
        $body .= "--:-:-:-:6      IADD vid, vid, warp;\n";
        $body .= "--:-:-:-:6      MOV vtid, lane;\n";
    }
    $body .= "--:-:-:-:6      ISETP.GE.U32.AND P6, PT, vid, units, PT;\n";
    $body .= "--:-:-:-:6      NOP;\n" x 3;
    $body .= "--:-:-:-:6  \@P6 BRA PHASE_END;\n";
    $body .= "${tag}_BEGIN:\n";
    $body .= $op->{code};
    $body .= "${tag}_END:\n";
    # a CTA unit owns the shared memory: no warp may start the next unit (and
    # store into smem) while a lagging warp still reads the previous unit's data,
    # so the whole CTA syncs here, after every outstanding load has landed.
    $body .= "3f:-:-:-:6      BAR.SYNC 0x0;\n" if $kind eq 'cta';
    $body .= sprintf "--:-:-:-:6      IADD vid, vid, 0x%x;\n", $step;
    $body .= "--:-:-:-:6      ISETP.LT.U32.AND P6, PT, vid, units, PT;\n";
    $body .= "--:-:-:-:6      NOP;\n" x 3;
    $body .= "--:-:-:-:6  \@P6 BRA ${tag}_BEGIN;\n";
    $body .= "--:-:-:-:6      BRA PHASE_END;\n";
}

print <<"EOF";
# Kernel: die
#
# Generated by die.pl: the persistent per-die kernel (see die.pl for the design).
# Params: 0:0x140 prog (128 B descriptors)  1:0x148 nph (u32)  2:0x150 ctl

<CONSTANT_MAPPING>
    param_prog0 : c[0x0][0x140]
    param_prog1 : c[0x0][0x144]
    param_nph   : c[0x0][0x148]
    param_ctl0  : c[0x0][0x150]
    param_ctl1  : c[0x0][0x154]
</CONSTANT_MAPPING>

<REGISTER_MAPPING>
    0     : tid
    1     : lane
    2     : warp
    3     : cta
    4     : vid
    5     : vtid
    6     : units
    7     : pidx
    8     : nph
    9,10  : prog<0-1>
    11,12 : ctl<0-1>
    13    : op
    14,15 : fw<0-1>
    16,17 : dsp<0-1>
    18    : bar_g
    19    : bar_g2
    20    : bar_old
    21    : bar_t
    22    : bar_one
    24,25 : bar_cp<0-1>
$maps</REGISTER_MAPPING>

--:-:6:-:1      S2R tid, SR_TID.X;
--:-:6:-:1      S2R cta, SR_CTAID.X;
--:-:-:-:6      MOV pidx, RZ;
--:-:-:-:6      MOV nph, param_nph;
--:-:-:-:6      MOV prog0, param_prog0;
--:-:-:-:6      MOV prog1, param_prog1;
--:-:-:-:6      MOV ctl0, param_ctl0;
--:-:-:-:6      MOV ctl1, param_ctl1;
20:-:-:-:6      LOP32I.AND lane, tid, 0x1f;
--:-:-:-:6      SHR.U32 warp, tid, 0x5;
# launch timestamp -> ctl[15] (ctl<0-1> is an odd pair: address through fw)
--:-:-:-:6      MOV fw0, ctl0;
--:-:-:-:6      MOV fw1, ctl1;
--:-:-:-:6      IADD op, cta, tid;
--:-:-:-:6      ISETP.EQ.U32.AND P4, PT, op, RZ, PT;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:6:-:2  \@P4 S2R op, SR_GLOBALTIMERLO;
20:1:-:-:6  \@P4 STG.E [fw+0x3c], op;

PHASE:
# descriptor -> smem (warp 0, lane = word)
01:-:-:-:6      ISETP.LT.U32.AND P6, PT, tid, 0x20, PT;
--:-:-:-:6      SHL fw0, pidx, 0x7;
--:-:-:-:6      IADD dsp0.CC, fw0, prog0;
--:-:-:-:6      IADD.X dsp1, RZ, prog1;
--:-:-:-:6      SHL op, lane, 0x2;
--:-:-:-:6      IADD fw0.CC, dsp0, op;
--:-:-:-:6      IADD.X fw1, dsp1, RZ;
--:-:-:-:6      IADD units, op, 0x@{[sprintf '%x', $DESC_OFF]};
--:-:1:-:2  \@P6 LDG.E op, [fw];
01:2:-:-:6  \@P6 STS [units], op;
02:-:-:-:6      BAR.SYNC 0x1;
--:-:1:-:1      LDS op, [RZ+0x@{[sprintf '%x', $DESC_OFF]}];
--:-:1:-:2      LDS units, [RZ+0x@{[sprintf '%x', $DESC_OFF + 4]}];
01:-:-:-:6      NOP;
$dispatch--:-:-:-:6      BRA PHASE_END;
$body
PHASE_END:
# ---- grid barrier (warp 0 counts in; everyone waits at BAR 1) ----
--:-:-:-:6      BAR.SYNC 0x1;
--:-:-:-:6      ISETP.GE.U32.AND P6, PT, tid, 0x20, PT;
--:-:-:-:6      ISETP.NE.U32.AND P5, PT, tid, RZ, PT;
--:-:-:-:6      MOV bar_cp0, ctl0;
--:-:-:-:6      MOV bar_cp1, ctl1;
--:-:-:-:6      MOV bar_one, 0x1;
--:-:-:-:6  \@P6 BRA BDONE;
--:-:1:-:2      LDG.E.CV bar_g, [bar_cp+0x4];
--:-:-:-:6      MEMBAR.GL.IVALLD;
01:-:2:-:2 \@!P5 ATOM.E.ADD bar_old, [bar_cp], bar_one;
02:-:3:-:2      SHFL.IDX PT, bar_old, bar_old, RZ, 0x1f;
04:-:-:-:6      IADD bar_t, bar_old, 0x1;
--:-:-:-:6      ISETP.EQ.U32.AND P3, PT, bar_t, 0x@{[sprintf '%x', $NCTA]}, PT;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6 \@!P3 BRA BSPIN;
--:-:-:-:6 \@!P5 STG.E.WT [bar_cp], RZ;
--:-:-:-:6      MEMBAR.GL.IVALLD;
--:-:-:-:6      IADD bar_g2, bar_g, 0x1;
--:-:-:-:6 \@!P5 STG.E.WT [bar_cp+0x4], bar_g2;
--:-:-:-:6      BRA BREL;
BSPIN:
--:-:1:-:2      LDG.E.CV bar_g2, [bar_cp+0x4];
01:-:-:-:6      ISETP.NE.U32.AND P4, PT, bar_g2, bar_g, PT;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6 \@!P4 BRA BSPIN;
BREL:
--:-:-:-:6      MEMBAR.GL.IVALLD;
BDONE:
--:-:-:-:6      BAR.SYNC 0x1;
# ---- phase timestamp: CTA 0 thread 0 writes %globaltimer.lo to ctl[16 + pidx] ----
--:-:-:-:6      IADD bar_t, cta, tid;
--:-:-:-:6      ISETP.EQ.U32.AND P4, PT, bar_t, RZ, PT;
--:-:-:-:6      SHL bar_g, pidx, 0x2;
--:-:-:-:6      IADD bar_cp0.CC, bar_cp0, bar_g;
--:-:-:-:6      IADD.X bar_cp1, bar_cp1, RZ;
--:-:6:-:2  \@P4 S2R bar_g2, SR_GLOBALTIMERLO;
20:-:-:-:6  \@P4 STG.E [bar_cp+0x40], bar_g2;
# ---- next phase ----
--:-:-:-:6      IADD pidx, pidx, 0x1;
--:-:-:-:6      ISETP.LT.U32.AND P6, PT, pidx, nph, PT;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6  \@P6 BRA PHASE;
--:-:-:-:f      EXIT;
EOF
