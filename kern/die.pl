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
my $TOK_OFF = 0xbf70;      # the token counter (u32), kept per CTA in shared memory
my $XG_OFF = 0xbf74;       # the exchange count (u32), per CTA in shared memory (xsend bumps it)
my $GEN_OFF = 0xbf78;      # the grid barrier generation (u32), per CTA in shared memory
my $PAIR_DESC_OFF = 0xbe80; # second position's unchanged 128-byte descriptor
my $STEP_OFF = 0xbf50;     # physical descriptors consumed by this phase (1 or 2)
my $PAIR_MODE_OFF = 0xbf54; # word 30 bits 3/4: generic paired / fused paired body
my $FLAGS_OFF = 0xbf58;    # primary descriptor's exchange / barrier flags

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
    [ 'actq256',    'actq.sass',        'warp', { GEMV_K => 256 } ],    # n/256 units in parallel instead of one warp
    [ 'rms2560',    'rmsnorm.sass',     'warp', { GEMV_K => 2560 } ],
    [ 'rms10240',   'rmsnorm.sass',     'warp', { GEMV_K => 10240 } ],
    # normq (RMSNorm + quantiser, 228 registers) does not fit beside the framework; nothing in the v1 block needs it
    [ 'hcsilu',     'hcops.sass',       'cta',  { HC_OP => 'silu' } ],
    [ 'hcmix',      'hcops.sass',       'cta',  { HC_OP => 'mix' } ],
    [ 'hcmixx',     'hcops.sass',       'cta',  { HC_OP => 'mix', HC_XOUT => 1 } ],
    [ 'hcwrite',    'hcops.sass',       'cta',  { HC_OP => 'write' } ],
    [ 'hcwsum',     'hcops.sass',       'cta',  { HC_OP => 'write', HC_SUMN => 1 } ],   # the MoE partials' fixed-order sum fused into the write
    [ 'conv1d',     'conv1d.sass',      'cta',  {} ],
    [ 'dnstep',     'dnstep.sass',      'cta',  { DN_TILED => 1 } ],   # local value head j reads local key head j % 16 (value head h pairs with key head h % 16)
    [ 'dnnorm',     'dnnorm.sass',      'warp', {} ],
    [ 'dnnormx',    'dnnorm.sass',      'warp', { DN_XOUT => 1 } ],
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
    [ 'bf16hcup',   'gemv_bf16.sass',   'cta',  { GEMV_K => 320, GEMV_HCUP => 1 } ],   # SiLU + hc up + lane mix + inbox write in one phase
    [ 'bf16hc',     'gemv_bf16.sass',   'cta',  { GEMV_K => 2560, GEMV_SLICES => 4 } ],
    [ 'bf16hcx',    'gemv_bf16.sass',   'cta',  { GEMV_K => 2560, GEMV_SLICES => 4, GEMV_XOUT => 1 } ],
    [ 'bf16hcrx',   'gemv_bf16.sass',   'cta',  { GEMV_K => 2560, GEMV_SLICES => 4, GEMV_XOUT => 1, GEMV_RMS => 1 } ],   # RMSNorm of S + hc down ++ inject -> slots
    [ 'bf16hcrwx',  'gemv_bf16.sass',   'cta',  { GEMV_K => 2560, GEMV_SLICES => 4, GEMV_XOUT => 1, GEMV_RMS => 1, GEMV_RMSW => 1 } ],   # + the pending stream write (S -> SOUT)
    [ 'q4kk2560',   'gemv_q4_k.sass',   'cta',  { GEMV_K => 2560 } ],
    [ 'q4kk6144',   'gemv_q4_k.sass',   'cta',  { GEMV_K => 6144 } ],
    [ 'q5kk2560',   'gemv_q4_k.sass',   'cta',  { GEMV_K => 2560, GEMV_FMT => 'q5_k' } ],
    [ 'q5kk6144',   'gemv_q4_k.sass',   'cta',  { GEMV_K => 6144, GEMV_FMT => 'q5_k' } ],
    [ 'q3kk2560',   'gemv_q3_k.sass',   'cta',  { GEMV_K => 2560 } ],
    [ 'q3kk6144',   'gemv_q3_k.sass',   'cta',  { GEMV_K => 6144 } ],
    [ 'q6kk2560',   'gemv_q6_k.sass',   'cta',  { GEMV_K => 2560 } ],
    # K = 6144 (attn_output of blk 11, 27, 31, 43; ssm_out of 38, 44): four K = 1536 column slices, y = ((p0 + p1) + p2) + p3
    [ 'q6kk1536',   'gemv_q6_k.sass',   'cta',  { GEMV_K => 1536 } ],
    [ 'q6kk1536acc','gemv_q6_k.sass',   'cta',  { GEMV_K => 1536, GEMV_ACC => 1 } ],
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
    [ 'amax',       'amax.sass',        'warp', {} ],
    [ 'topk',       'topk.sass',        'cta',  {} ],
    [ 'sample',     'sample.sass',      'warp', {} ],
    # ---- QSA: int8 KV cache, indexer, gather (ref/qsa.h) ----
    [ 'kvgather',   'kvgather.sass',    'warp', {} ],
    [ 'attnpartn',  'attnpartn.sass',   'cta',  {} ],
    [ 'attnqkq',    'attnqkq.sass',     'warp', {} ],
    [ 'idxqk',     'idxqk.sass',      'warp', {} ],
    [ 'idxscore',  'idxscore.sass',   'cta',  {} ],
    [ 'idxsel',    'idxsel.sass',     'cta',  {} ],
    [ 'xcopy',     'xcopy.sass',      'cta',  {} ],
    # Paired BF16 opcodes are appended to retain the scalar descriptor ABI.
    [ 'bf16biasp',  'gemv_bf16_pair.sass', 'cta', { GEMV_K => 2560, GEMV_BIAS => 1 } ],
    [ 'bf16k2560p', 'gemv_bf16_pair.sass', 'cta', { GEMV_K => 2560 } ],
    [ 'bf16k320p',  'gemv_bf16_pair.sass', 'cta', { GEMV_K => 320 } ],
    [ 'bf16hcp',    'gemv_bf16_pair.sass', 'cta', { GEMV_K => 2560, GEMV_SLICES => 4 } ],
    [ 'bf16hcxp',   'gemv_bf16_pair.sass', 'cta', { GEMV_K => 2560, GEMV_SLICES => 4, GEMV_XOUT => 1 } ],
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
            next unless $l =~ /^\s*(\w+)\s*:\s*c\[0x([02])\]\[0x([0-9a-f]+)\]/;
            my ($name, $bank, $addr) = ($1, $2, hex $3);
            if ($addr == 0x14) { $has_grid = 1; next; }
            die "$file: parameter $name at 0x$3 outside the descriptor" if $addr < 0x140 || $addr >= 0x140 + 4 * 30;
            push @params, [ $name, 2 + ($addr - 0x140) / 4, $bank eq '2' ? $PAIR_DESC_OFF : $DESC_OFF ];
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
        my ($name, $word, $desc) = @{$params[$i]};
        $desc = $DESC_OFF unless defined $desc;
        $pro .= sprintf "--:-:1:-:%d      LDS %s_%s, [RZ+0x%x];\n", $i == $#params ? 2 : 1, $tag, $name, $desc + 4 * $word;
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
require './trace.pl';
my $maps = '';
my $body = '';
my $dispatch = '';
for my $k (0 .. $#OPS) {
    my ($tag, $file, $kind, $env) = @{$OPS[$k]};
    my $op = load_op($tag, $file, $env);
    if ($tag eq 'plerows' && $ENV{DIE_TRACE}) {
        # Hash inputs in R18..R25 are dead here; the wait's live pointers and
        # predicates use other registers. Keep the standalone op unchanged.
        my $begin = trace_code(5, $TOK_OFF);
        my $end = trace_code(6, $TOK_OFF);
        $op->{code} =~ s/(plerows_WAITRDY:)/$begin$1/ or die 'PLE wait label missing';
        $op->{code} =~ s/([^\n]*BRA plerows_WAITRDY;\n)/$1$end/ or die 'PLE wait branch missing';
    }
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
    $body .= "--:-:-:-:6  \@P6 BRA POSITION_END;\n";
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
    $body .= "--:-:-:-:6      BRA POSITION_END;\n";
}

my @trace = map { trace_code($_, $TOK_OFF, $STEP_OFF) } 0..7;
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
    param_ntok  : c[0x0][0x158]
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
    18,19 : traceclock<0-1>
    18    : bar_g
    19    : bar_g2
    20    : bar_old
    21    : bar_t
    22    : bar_one
    24,25 : bar_cp<0-1>
    14,15 : xa<0-1>
    16,17 : xb<0-1>
    20,21 : xc<0-1>
    22,23 : xd<0-1>
    24,25 : xe<0-1>
    26,27 : xf<0-1>
    28,29 : xg<0-1>
    30,31 : xh<0-1>
    32,33 : xi<0-1>
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
# the token counter lives in shared memory (TOK, below the descriptor), seeded from
# ctl[2]; ctl<0-1> is an odd pair: address through fw
--:-:-:-:6      MOV fw0, ctl0;
--:-:-:-:6      MOV fw1, ctl1;
--:-:1:-:2      LDG.E op, [fw+0x8];
01:2:-:-:6      STS [RZ+0x@{[sprintf '%x', $TOK_OFF]}], op;
# the exchange count (XG) likewise, seeded from ctl[8]; xsend bumps it per CTA
02:-:1:-:2      LDG.E op, [fw+0x20];
01:2:-:-:6      STS [RZ+0x@{[sprintf '%x', $XG_OFF]}], op;
# and the grid barrier generation (GEN), seeded from ctl[1]
02:-:1:-:2      LDG.E op, [fw+0x4];
01:2:-:-:6      STS [RZ+0x@{[sprintf '%x', $GEN_OFF]}], op;
# launch timestamp -> ctl[15]
02:-:-:-:6      IADD op, cta, tid;
--:-:-:-:6      ISETP.EQ.U32.AND P4, PT, op, RZ, PT;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:6:-:2  \@P4 S2R op, SR_GLOBALTIMERLO;
20:1:-:-:6  \@P4 STG.E [fw+0x3c], op;
# descriptor 0 prefetch: warp 0 lane i fetches word i into vtid and word 31 (the +tok
# mask) into vid; later phases prefetch theirs during the previous grid barrier
01:-:-:-:6      ISETP.GE.U32.AND P6, PT, tid, 0x20, PT;
--:-:-:-:6      SHL op, lane, 0x2;
--:-:-:-:6      IADD fw0.CC, prog0, op;
--:-:-:-:6      IADD.X fw1, prog1, RZ;
--:-:-:-:6      MOV dsp0, prog0;
--:-:-:-:6      MOV dsp1, prog1;
--:-:4:-:1 \@!P6 LDG.E vtid, [fw];
--:-:4:-:2 \@!P6 LDG.E vid, [dsp+0x7c];

PHASE:
$trace[0]# descriptor -> smem (warp 0, lane = word; the words were prefetched); word 31 is a
# mask of the parameter words that get the token counter added
01:-:-:-:6      ISETP.LT.U32.AND P6, PT, tid, 0x20, PT;
--:-:-:-:6      SHL op, lane, 0x2;
--:-:-:-:6      IADD units, op, 0x@{[sprintf '%x', $DESC_OFF]};
--:-:1:-:2      LDS bar_t, [RZ+0x@{[sprintf '%x', $TOK_OFF]}];
09:-:-:-:6      SHR.U32 vid, vid, lane;
--:-:-:-:6      LOP32I.AND vid, vid, 0x1;
--:-:-:-:6      ISETP.NE.U32.AND P5, PT, vid, RZ, PT;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6  \@P5 IADD vtid, vtid, bar_t;
--:2:-:-:6  \@P6 STS [units], vtid;
02:-:-:-:6      BAR.SYNC 0x1;
--:-:1:-:1      LDS op, [RZ+0x@{[sprintf '%x', $DESC_OFF]}];
--:-:1:-:2      LDS units, [RZ+0x@{[sprintf '%x', $DESC_OFF + 4]}];
01:-:-:-:6      NOP;
# A paired phase is two adjacent scalar descriptors. Save the primary flags, and
# load the second descriptor (with its own token mask) before running either body.
# Bit 3 marks a pair, bit 4 a fused body that consumes both descriptor banks.
--:-:1:-:2      LDS bar_g, [RZ+0x@{[sprintf '%x', $DESC_OFF + 0x78]}];
01:2:-:-:6      STS [RZ+0x@{[sprintf '%x', $FLAGS_OFF]}], bar_g;
--:-:-:-:6      LOP32I.AND bar_old, bar_g, 0x18;
--:2:-:-:6      STS [RZ+0x@{[sprintf '%x', $PAIR_MODE_OFF]}], bar_old;
--:-:-:-:6      SHR.U32 bar_old, bar_old, 0x3;
--:-:-:-:6      LOP32I.AND bar_old, bar_old, 0x1;
--:-:-:-:6      IADD bar_g2, bar_old, 0x1;
--:2:-:-:6      STS [RZ+0x@{[sprintf '%x', $STEP_OFF]}], bar_g2;
--:-:-:-:6      ISETP.EQ.U32.AND P5, PT, bar_old, RZ, PT;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
02:-:-:-:6  \@P5 BRA PAIR_LOADED;
--:-:-:-:6      IADD bar_t, pidx, 0x1;
--:-:-:-:6      SHL bar_t, bar_t, 0x7;
--:-:-:-:6      IADD dsp0.CC, prog0, bar_t;
--:-:-:-:6      IADD.X dsp1, prog1, RZ;
--:-:-:-:6      SHL op, lane, 0x2;
--:-:-:-:6      IADD fw0.CC, dsp0, op;
--:-:-:-:6      IADD.X fw1, dsp1, RZ;
--:-:-:-:6      ISETP.LT.U32.AND P6, PT, tid, 0x20, PT;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:4:-:1  \@P6 LDG.E vtid, [fw];
--:-:4:-:2  \@P6 LDG.E vid, [dsp+0x7c];
--:-:1:-:2      LDS bar_t, [RZ+0x@{[sprintf '%x', $TOK_OFF]}];
09:-:-:-:6      SHR.U32 vid, vid, lane;
--:-:-:-:6      LOP32I.AND vid, vid, 0x1;
--:-:-:-:6      ISETP.NE.U32.AND P5, PT, vid, RZ, P6;
--:-:-:-:6      IADD units, op, 0x@{[sprintf '%x', $PAIR_DESC_OFF]};
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6  \@P5 IADD vtid, vtid, bar_t;
--:2:-:-:6  \@P6 STS [units], vtid;
02:-:-:-:6      BAR.SYNC 0x1;
PAIR_LOADED:
--:-:1:-:1      LDS op, [RZ+0x@{[sprintf '%x', $DESC_OFF]}];
--:-:1:-:2      LDS units, [RZ+0x@{[sprintf '%x', $DESC_OFF + 4]}];
01:-:-:-:6      NOP;
$trace[1]# ---- exchange wait (descriptor word 30 bit 1): warp 0 of every CTA polls this die's
# flags (ctl[28..29] -> u32[6][16], one per sending die and CTA; absent dies hold ~0;
# lane l reads words l, l + 32, l + 64) until all 96 read >= the exchange count (XG
# in shared memory), invalidates L1, and the CTA syncs ----
--:-:1:-:2      LDS bar_g, [RZ+0x@{[sprintf '%x', $DESC_OFF + 0x78]}];
01:-:-:-:6      LOP32I.AND bar_g, bar_g, 0x2;
--:-:-:-:6      ISETP.EQ.U32.AND P5, PT, bar_g, RZ, PT;
--:-:-:-:6      ISETP.GE.U32.AND P6, PT, tid, 0x20, PT;
--:-:-:-:6      MOV bar_cp0, ctl0;
--:-:-:-:6      MOV bar_cp1, ctl1;
--:-:-:-:6      SHL bar_old, lane, 0x2;
--:-:-:-:6  \@P5 BRA XWDONE;
--:-:-:-:6  \@P6 BRA XWBAR;
--:-:1:-:1      LDS bar_g, [RZ+0x@{[sprintf '%x', $XG_OFF]}];
--:-:1:-:2      LDG.E.64 xe0, [bar_cp+0x70];
01:-:-:-:6      IADD xe0.CC, xe0, bar_old;
--:-:-:-:6      IADD.X xe1, xe1, RZ;
XWSPIN:
--:-:3:-:1      LDG.E.CV bar_t, [xe];
--:-:3:-:1      LDG.E.CV bar_one, [xe+0x80];
--:-:3:-:2      LDG.E.CV bar_g2, [xe+0x100];
04:-:-:-:6      IMNMX.U32 bar_t, bar_t, bar_one, PT;
--:-:-:-:6      IMNMX.U32 bar_t, bar_t, bar_g2, PT;
--:-:-:-:6      ISETP.GE.U32.AND P4, PT, bar_t, bar_g, PT;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6      VOTE.ALL P4, P4;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6 \@!P4 BRA XWSPIN;
--:-:-:-:6      MEMBAR.GL.IVALLD;
XWBAR:
--:-:-:-:6      BAR.SYNC 0x1;
XWDONE:
$trace[2]OP_DISPATCH:
$dispatch--:-:-:-:6      BRA POSITION_END;
$body
POSITION_END:
# Units of a stateful op belong to the same CTA for both positions. Finish every
# unit and make its writes visible before replaying this op with position 1.
# Independent ops also take this path; the grid barrier and peer flags happen
# once, after both positions. A fused pair has already produced both outputs.
3f:-:-:-:6      BAR.SYNC 0x1;
--:-:1:-:2      LDS bar_g, [RZ+0x@{[sprintf '%x', $PAIR_MODE_OFF]}];
01:-:-:-:6      ISETP.EQ.U32.AND P5, PT, bar_g, 0x8, PT;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
# Every warp must sample MODE before any warp resets it for the second body.
--:-:-:-:6      BAR.SYNC 0x1;
--:-:-:-:6 \@!P5 BRA PHASE_END;
--:-:-:-:6      MEMBAR.GL.IVALLD;
--:-:-:-:6      MOV bar_g, RZ;
--:2:-:-:6      STS [RZ+0x@{[sprintf '%x', $PAIR_MODE_OFF]}], bar_g;
--:-:-:-:6      SHL bar_t, lane, 0x2;
--:-:-:-:6      IADD bar_old, bar_t, 0x@{[sprintf '%x', $PAIR_DESC_OFF]};
--:-:-:-:6      IADD bar_t, bar_t, 0x@{[sprintf '%x', $DESC_OFF]};
--:-:-:-:6      ISETP.LT.U32.AND P6, PT, tid, 0x20, PT;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:1:-:2  \@P6 LDS bar_g, [bar_old];
01:2:-:-:6  \@P6 STS [bar_t], bar_g;
02:-:-:-:6      BAR.SYNC 0x1;
--:-:1:-:1      LDS op, [RZ+0x@{[sprintf '%x', $DESC_OFF]}];
--:-:1:-:2      LDS units, [RZ+0x@{[sprintf '%x', $DESC_OFF + 4]}];
01:-:-:-:6      BRA OP_DISPATCH;

PHASE_END:
$trace[3]# ---- grid barrier (warp 0 counts in; everyone waits at BAR 1) ----
--:-:-:-:6      BAR.SYNC 0x1;
--:-:-:-:6      ISETP.GE.U32.AND P6, PT, tid, 0x20, PT;
--:-:-:-:6      ISETP.NE.U32.AND P5, PT, tid, RZ, PT;
# prefetch the next logical phase (skip a pair's second descriptor)
--:-:1:-:2      LDS bar_t, [RZ+0x@{[sprintf '%x', $STEP_OFF]}];
01:-:-:-:6      IADD bar_t, pidx, bar_t;
--:-:-:-:6      ISETP.LT.U32.AND P3, PT, bar_t, nph, PT;
--:-:-:-:6      SHL op, lane, 0x2;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6      SEL bar_t, bar_t, RZ, P3;
--:-:-:-:6      SHL bar_t, bar_t, 0x7;
--:-:-:-:6      IADD dsp0.CC, bar_t, prog0;
--:-:-:-:6      IADD.X dsp1, RZ, prog1;
--:-:-:-:6      IADD fw0.CC, dsp0, op;
--:-:-:-:6      IADD.X fw1, dsp1, RZ;
--:3:4:-:1 \@!P6 LDG.E vtid, [fw];
--:3:4:-:2 \@!P6 LDG.E vid, [dsp+0x7c];
--:-:-:-:6      MOV bar_cp0, ctl0;
--:-:-:-:6      MOV bar_cp1, ctl1;
# (the prefetch reads its address registers late: read barrier 3 guards fw/dsp, which
# the barrier code below reuses as xa/xb)
# flag-array barrier: thread 0 of every CTA writes the generation it arrives at
# (GEN in shared memory + 1) to ctl[32 + cta]; lanes 0..15 of warp 0 poll the 16
# words until all read >= it (VOTE.ALL keeps the warp together); no atomics, no
# last-arriver release. CTA 0 thread 0 mirrors the generation to ctl[1].
--:-:1:-:1      LDS bar_g, [RZ+0x@{[sprintf '%x', $GEN_OFF]}];
--:-:-:-:6      SHL bar_t, lane, 0x2;
--:-:-:-:6      SHL bar_g2, cta, 0x2;
--:-:-:-:6      ISETP.LT.U32.AND P3, PT, lane, 0x10, PT;
# descriptor word 30 bit 2: the next phase does not depend on this one (and writes
# nothing this one reads), so no grid barrier: the CTA syncs and moves on
--:-:2:-:2      LDS bar_old, [RZ+0x@{[sprintf '%x', $FLAGS_OFF]}];
02:-:-:-:6      LOP32I.AND bar_old, bar_old, 0x4;
--:-:-:-:6      ISETP.NE.U32.AND P4, PT, bar_old, RZ, PT;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6  \@P4 BRA BDONE;
--:-:-:-:6  \@P6 BRA BDONE;
04:-:-:-:6      IADD xa0.CC, bar_cp0, bar_t;
--:-:-:-:6      IADD.X xa1, bar_cp1, RZ;
--:-:-:-:6      IADD xb0.CC, bar_cp0, bar_g2;
--:-:-:-:6      IADD.X xb1, bar_cp1, RZ;
--:-:-:-:6      MEMBAR.GL.IVALLD;
01:-:-:-:6      IADD bar_g, bar_g, 0x1;
--:-:-:-:6      MOV32I bar_g2, 0xffffffff;
--:2:-:-:6 \@!P5 STG.E.WT [xb+0x80], bar_g;
# a send phase (descriptor word 30 bit 0): thread 0 of every CTA fences (its CTA's
# stores to the peers' inboxes are ordered before it by the BAR.SYNC above), bumps the
# CTA's exchange count XG and stores it to flags_e[me][cta] of every die e
# (ctl[16 + 2e] = &flags_e[me][0]; the host points absent dies at a scratch word, so
# all six stores are unconditional); CTA 0 mirrors the count to ctl[8]. This runs
# while the other CTAs' arrivals propagate, ahead of the poll below.
--:-:1:-:2      LDS bar_old, [RZ+0x@{[sprintf '%x', $FLAGS_OFF]}];
01:-:-:-:6      LOP32I.AND bar_old, bar_old, 0x1;
--:-:-:-:6      ISETP.NE.U32.AND P4, PT, bar_old, RZ, !P5;
--:-:-:-:6      SHL units, cta, 0x2;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:1:-:1  \@P4 LDG.E.64 xc0, [bar_cp+0x40];
--:-:1:-:1  \@P4 LDG.E.64 xd0, [bar_cp+0x48];
--:-:1:-:1  \@P4 LDG.E.64 xf0, [bar_cp+0x50];
--:-:1:-:1  \@P4 LDG.E.64 xg0, [bar_cp+0x58];
--:-:1:-:1  \@P4 LDG.E.64 xh0, [bar_cp+0x60];
--:-:1:-:1  \@P4 LDG.E.64 xi0, [bar_cp+0x68];
--:-:1:-:2  \@P4 LDS op, [RZ+0x@{[sprintf '%x', $XG_OFF]}];
--:-:-:-:6  \@P4 MEMBAR.SYS;
01:-:-:-:6      IADD op, op, 0x1;
--:-:-:-:6      IADD xc0.CC, xc0, units;
--:-:-:-:6      IADD.X xc1, xc1, RZ;
--:-:-:-:6      IADD xd0.CC, xd0, units;
--:-:-:-:6      IADD.X xd1, xd1, RZ;
--:-:-:-:6      IADD xf0.CC, xf0, units;
--:-:-:-:6      IADD.X xf1, xf1, RZ;
--:-:-:-:6      IADD xg0.CC, xg0, units;
--:-:-:-:6      IADD.X xg1, xg1, RZ;
--:-:-:-:6      IADD xh0.CC, xh0, units;
--:-:-:-:6      IADD.X xh1, xh1, RZ;
--:-:-:-:6      IADD xi0.CC, xi0, units;
--:-:-:-:6      IADD.X xi1, xi1, RZ;
--:-:-:-:6      ISETP.EQ.U32.AND P2, PT, cta, RZ, P4;
--:2:-:-:1  \@P4 STG.E.WT [xc], op;
--:2:-:-:1  \@P4 STG.E.WT [xd], op;
--:2:-:-:1  \@P4 STG.E.WT [xf], op;
--:2:-:-:1  \@P4 STG.E.WT [xg], op;
--:2:-:-:1  \@P4 STG.E.WT [xh], op;
--:2:-:-:1  \@P4 STG.E.WT [xi], op;
--:2:-:-:1  \@P4 STS [RZ+0x@{[sprintf '%x', $XG_OFF]}], op;
--:2:-:-:6  \@P2 STG.E [bar_cp+0x20], op;
BSPIN:
--:-:1:-:2  \@P3 LDG.E.CV bar_g2, [xa+0x80];
01:-:-:-:6      ISETP.GE.U32.AND P4, PT, bar_g2, bar_g, PT;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6      VOTE.ALL P4, P4;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6 \@!P4 BRA BSPIN;
--:-:-:-:6      MEMBAR.GL.IVALLD;
02:-:-:-:6      IADD bar_t, cta, tid;
--:-:-:-:6      ISETP.EQ.U32.AND P4, PT, bar_t, RZ, PT;
--:2:-:-:6      STS [RZ+0x@{[sprintf '%x', $GEN_OFF]}], bar_g;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:2:-:-:6  \@P4 STG.E [bar_cp+0x4], bar_g;
BDONE:
02:-:-:-:6      BAR.SYNC 0x1;
# ---- phase timestamp: CTA 0 thread 0 writes %globaltimer.lo to ctl[16 + pidx] ----
--:-:-:-:6      IADD bar_t, cta, tid;
--:-:-:-:6      ISETP.EQ.U32.AND P4, PT, bar_t, RZ, PT;
--:-:-:-:6      SHL bar_g, pidx, 0x2;
--:-:-:-:6      IADD bar_cp0.CC, bar_cp0, bar_g;
--:-:-:-:6      IADD.X bar_cp1, bar_cp1, RZ;
--:-:6:-:2  \@P4 S2R bar_g2, SR_GLOBALTIMERLO;
20:-:-:-:6  \@P4 STG.E [bar_cp+0x100], bar_g2;
$trace[4]# ---- next phase ----
--:-:1:-:2      LDS bar_t, [RZ+0x@{[sprintf '%x', $STEP_OFF]}];
01:-:-:-:6      IADD pidx, pidx, bar_t;
--:-:-:-:6      ISETP.LT.U32.AND P6, PT, pidx, nph, PT;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6  \@P6 BRA PHASE;
# ---- token done: thread 0 bumps TOK (and CTA 0 mirrors it to ctl[2]); everyone
# re-reads it behind a barrier and runs the program again while tok < ntok ----
# End of turn: amax sets ctl[5] = 0x80000000 when it picks a stop token after the
# prompt (every die picks the same token, so every die stops after the same one).
# Thread 0 ors it into TOK: the unsigned tok < ntok test fails on every CTA, and
# ctl[2] tells the host both the count (low 31 bits) and that the turn ended.
--:-:-:-:6      ISETP.NE.U32.AND P5, PT, tid, RZ, PT;
--:-:-:-:6      MOV bar_cp0, ctl0;
--:-:-:-:6      MOV bar_cp1, ctl1;
--:-:-:-:6      BAR.SYNC 0x1;
--:-:1:-:1 \@!P5 LDS bar_t, [RZ+0x@{[sprintf '%x', $TOK_OFF]}];
--:-:3:-:1 \@!P5 LDG.E.CV bar_old, [bar_cp+0x14];
--:-:3:-:1 \@!P5 LDG.E.CV bar_g2, [bar_cp+0x18];
--:-:3:-:2 \@!P5 LDG.E.CV bar_one, [bar_cp+0x24];
# ctl[9] is the number of known positions in a pass. Zero retains the B=1 ABI.
05:-:-:-:6      ISETP.EQ.U32.AND P3, PT, bar_one, RZ, !P5;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6  \@P3 MOV bar_one, 0x1;
--:-:-:-:6 \@!P5 IADD bar_t, bar_t, bar_one;
04:-:-:-:6 \@!P5 LOP.OR bar_t, bar_t, bar_old;
# Abort: the host writes ctl[6] = a token count a few tokens ahead of every die (0: none); the
# die whose count reaches it ends the turn exactly like a stop token, so all dies stop together.
--:-:-:-:6      ISETP.NE.U32.AND P3, PT, bar_g2, RZ, !P5;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6      ISETP.GE.U32.AND P3, PT, bar_t, bar_g2, P3;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6  \@P3 LOP32I.OR bar_t, bar_t, 0x80000000;
--:2:-:-:6 \@!P5 STS [RZ+0x@{[sprintf '%x', $TOK_OFF]}], bar_t;
--:2:-:-:6  \@P4 STG.E [bar_cp+0x8], bar_t;
02:-:-:-:6      BAR.SYNC 0x1;
$trace[7]# publication sample uses the completed position and last phase slot
--:-:1:-:2      LDS bar_t, [RZ+0x@{[sprintf '%x', $TOK_OFF]}];
--:-:-:-:6      MOV pidx, RZ;
01:-:-:-:6      ISETP.LT.U32.AND P6, PT, bar_t, param_ntok, PT;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6      NOP;
--:-:-:-:6  \@P6 BRA PHASE;
--:-:-:-:f      EXIT;
EOF
