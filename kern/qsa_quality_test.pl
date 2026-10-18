#!/usr/bin/env perl
# Small analytic frozen-input cases: validates separation of score arithmetic
# from selector consistency, plus rejection of malformed captures.
use strict;
use warnings;
use File::Temp qw(tempdir);
use File::Path qw(make_path);
use FindBin;
use JSON::PP qw(encode_json decode_json);
use Test::More;
my $tmp=tempdir(CLEANUP=>1);
my $capture="$tmp/capture";
make_path("$capture/n8");
sub put {
    my ($path,$bytes)=@_;
    open my $f,'>:raw',$path or die $!;
    print {$f} $bytes or die $!;
    close $f or die $!;
}
sub read_json {
    my ($path)=@_;
    open my $f,'<:raw',$path or die $!;
    local $/;
    return decode_json(<$f>);
}
my $complete={
    schema=>'flash-sass.qsa-layer-capture.v1',positions=>8,layer=>3,die=>0,
    budget=>1,epsilon=>1e-6,byte_order=>'little',snapshot_lengths=>[8],
    input_boundary=>'post_projection_before_norm_rope_and_kv_quantization',
    gamma=>'supplied_multiplicative_f32',evidence_kind=>'analytic_cpu_test',
};
my $snapshot={
    schema=>'flash-sass.qsa-layer-snapshot.v1',length=>8,position=>7,layer=>3,
    die=>0,budget=>1,complete_blocks=>2,tail=>0,attended_keys=>4,
    sparse_exclusion=>JSON::PP::true,
};
put("$capture/complete.json",encode_json($complete));
put("$capture/n8/snapshot.json",encode_json($snapshot));
put("$capture/raw_ik.f32",pack('f<*',(1)x(8*128)));
put("$capture/rope_cs.f32",pack('f<*',(1)x(8*32)));
put("$capture/rope_sn.f32",pack('f<*',(0)x(8*32)));
put("$capture/iqn.f32",pack('f<*',(1)x128));
put("$capture/ikn.f32",pack('f<*',(1)x128));
my @query=((1,(0)x127)x4);
put("$capture/n8/iq.f32",pack('f<*',@query));
put("$capture/n8/iqk.f32",pack('f<*',@query,(1)x128));
put("$capture/n8/pool.f16",pack('v*',0x3c00,(0)x127,0x4000,(0)x127));
put("$capture/n8/scores.f32",pack('f<*',4,8));
put("$capture/n8/selected_ids.u32",pack('V',1));
my $seq=0;
sub run {
    my $out="$tmp/result".++$seq.".json";
    my $pid=fork(); die 'fork' unless defined $pid;
    if(!$pid) {
        open STDOUT,'>',"$tmp/stdout" or die $!;
        open STDERR,'>',"$tmp/stderr" or die $!;
        exec {$^X} $^X,"$FindBin::Bin/qsa_quality_replay.pl",$capture,8,$out;
        die "exec: $!";
    }
    waitpid($pid,0);
    die 'child signaled' if $?&127;
    return ($?>>8,(-f $out?read_json($out):undef),$out);
}
my ($rc,$r)=run();
is($rc,0,'analytic frozen replay runs');
ok($r->{complete},'final report completes after hashing');
ok(!$r->{quality_validation}&&!$r->{independent_checkpoint_evidence},'CPU replay makes no checkpoint claim');
is($r->{input_origin},'unverified_external_capture','CPU input is not automatically labeled GPU');
is($r->{candidate_raw_scores_vs_exported_inputs_fp64}{max_abs},0,'fixed query and pool produce exact unscaled scores 4 and 8');
is($r->{candidate_scaled_scores_vs_exported_inputs_fp64}{max_abs},0,'explicit sqrt128 score conversion agrees');
ok($r->{provided_selection_vs_candidate_score_readback}{exact_tie_compatible},'IDs agree with actual supplied scores');
is($r->{candidate_score_readback_ranking}{cutoff_margin},4,'readback cutoff margin is independent of score recomputation');
ok($r->{input_hashes_match_before_after_replay},'input endpoint hashes match');
put("$capture/n8/scores.f32",pack('f<*',9,8));
($rc,$r)=run();
is($rc,0,'finite score inconsistency is reported numerically');
ok(!$r->{provided_selection_vs_candidate_score_readback}{exact_tie_compatible},'selector discrepancy is exposed using the supplied scores');
is($r->{provided_selection_vs_candidate_score_readback}{selected_below_cutoff},1,'wrong selected ID lies below cutoff');
is($r->{candidate_raw_scores_vs_exported_inputs_fp64}{max_abs},5,'separate arithmetic discrepancy is measured');
put("$capture/n8/scores.f32",pack('f<*',8,8));
($rc,$r)=run();
ok($r->{provided_selection_vs_candidate_score_readback}{exact_tie_compatible},'noncanonical tied ID is allowed');
ok($r->{candidate_score_readback_ranking}{ambiguous_cutoff},'exact tie ambiguity is explicit');
put("$capture/n8/selected_ids.u32",pack('V',2));
($rc,$r)=run();
ok($rc!=0&&!defined($r),'out-of-range selected ID leaves no final report');
put("$capture/n8/selected_ids.u32",pack('V',1));
put("$capture/n8/scores.f32",pack('V*',0x7fc00000,0));
($rc,$r)=run();
ok($rc!=0&&!defined($r),'nonfinite score leaves no final report');
put("$capture/n8/scores.f32",pack('f<*',4,8));
$snapshot->{sparse_exclusion}='false';
put("$capture/n8/snapshot.json",encode_json($snapshot));
($rc,$r)=run();
ok($rc!=0&&!defined($r),'truthy string cannot replace sparse boolean');
done_testing();
