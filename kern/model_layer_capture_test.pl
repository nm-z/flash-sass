#!/usr/bin/env perl
use strict;
use warnings;
use File::Temp qw(tempdir);
use JSON::PP;
use Test::More;
use FindBin;
my $tool = $ARGV[0] // "$FindBin::Bin/model_layer_capture_test";
my $tmp = tempdir(CLEANUP => 1);
sub read_raw {
    my ($p) = @_;
    open my $f, '<:raw', $p or die "$p: $!";
    local $/; my $s = <$f>; close $f or die $!;
    return $s;
}
sub json { return decode_json(read_raw($_[0])); }
sub run {
    my ($out, $lengths) = @_;
    my $pid = fork();
    die "fork: $!" unless defined $pid;
    if (!$pid) {
        open STDOUT, '>', "$tmp/stdout" or die $!;
        open STDERR, '>', "$tmp/stderr" or die $!;
        exec {$tool} $tool, $out, $lengths;
        die "exec: $!";
    }
    waitpid($pid, 0);
    die "child terminated by signal" if $? & 127;
    return $? >> 8;
}
my $out = "$tmp/fixtures";
is(run($out, '1,4,8,9'), 0, 'CPU readback mock capture succeeds');
my $m = json("$out/complete.json");
is($m->{schema}, 'flash-sass.qsa-layer-capture.v1', 'completion schema');
is($m->{positions}, 9, 'complete prefix length');
is_deeply($m->{snapshot_lengths}, [1,4,8,9], 'predeclared lengths retained');
ok(!$m->{quality_validation} && !$m->{independent_checkpoint_evidence},
   'capture does not attest independent quality');
ok(!$m->{upstream_activation_quantization_isolated},
   'postprojection boundary does not claim upstream isolation');
is(-s "$out/raw_k.f32", 9*512*4, 'raw K prefix shape');
is(-s "$out/raw_v.f32", 9*512*4, 'raw V prefix shape');
is(-s "$out/raw_ik.f32", 9*128*4, 'raw indexer key prefix shape');
my @k = unpack('f<*', read_raw("$out/raw_k.f32"));
is($k[8*512], 800, 'all K prefix rows kept in token order');
my @ik = unpack('f<*', read_raw("$out/raw_ik.f32"));
is($ik[8*128], 80, 'raw indexer keys exclude query projections');
is(-s "$out/rope_cs.f32", 9*32*4, 'exact supplied RoPE table shape');
my @gamma = unpack('f<*', read_raw("$out/qn.f32"));
is($gamma[0], 1, 'supplied gamma is not offset by one');
my $first = json("$out/n1/snapshot.json");
is($first->{attended_keys}, 1, 'first token includes its one tail key');
is(-s "$out/n1/pool.f16", 0, 'no complete-block pool at first token');
is(-s "$out/n1/selected_ids.u32", 0, 'no implicit complete block at first token');
my $dense = json("$out/n4/snapshot.json");
is($dense->{scores_source}, 'not_dispatched', 'dense scores are explicitly unavailable');
ok(!-e "$out/n4/scores.f32", 'uncleared dense score buffer is not exported');
is($dense->{selected_ids_source}, 'implicit_all_blocks', 'dense identity origin explicit');
is_deeply([unpack('V*', read_raw("$out/n4/selected_ids.u32"))], [0],
          'dense selected block is implicit zero');
my $sparse = json("$out/n8/snapshot.json");
ok($sparse->{sparse_exclusion}, 'sparse exclusion recorded above block budget');
is($sparse->{attended_keys}, 4, 'sparse key count at exact complete block');
is($sparse->{selected_ids_source}, 'gpu', 'sparse IDs have readback origin');
is_deeply([unpack('V*', read_raw("$out/n8/selected_ids.u32"))], [1],
          'selected ID readback is preserved');
is_deeply([unpack('f<*', read_raw("$out/n8/scores.f32"))], [0.5,1.5],
          'valid score readback is preserved');
my @pool = unpack('v*', read_raw("$out/n8/pool.f16"));
is_deeply(\@pool, [map { 0x3000+$_ } 0..255], 'tiled pool is exported as block-major FP16');
is(-s "$out/n9/kq.i8", 9*512, 'only valid INT8 prefix is exported');
is(-s "$out/n9/ks.f16", 9*16*2, 'scale prefix shape');
my $tail = json("$out/n9/snapshot.json");
is($tail->{attended_keys}, 5, 'sparse tail included');
is_deeply([unpack('f<', substr(read_raw("$out/n9/aout.f32"),0,4))], [1008],
          'snapshot stores the selected token output');
my $before = read_raw("$out/complete.json");
is(run($out, '1,4,8,9'), 2, 'existing capture directory is rejected');
is(read_raw("$out/complete.json"), $before, 'existing completion bytes remain intact');
my $i = 0;
for my $bad ('0','10','1,1','1,',',1','1x','-1','1,,4') {
    my $dir = "$tmp/invalid-" . ++$i;
    is(run($dir, $bad), 2, "invalid length list rejected: $bad");
    ok(!-e $dir, 'invalid configuration creates no capture');
}
done_testing();
