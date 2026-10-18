#!/usr/bin/env perl
# Analytic CPU fixtures and malformed-input tests, not model/GPU measurements.
use strict;
use warnings;
use File::Temp qw(tempdir);
use File::Path qw(make_path);
use FindBin;
use JSON::PP qw(encode_json decode_json);
use Test::More;
my $tmp=tempdir(CLEANUP=>1);
my $seq=0;
my $capture="$tmp/capture";
make_path("$capture/n11","$capture/n1");
sub put {
    my ($p,$bytes)=@_;
    open my $f,'>:raw',$p or die "$p: $!";
    print {$f} $bytes or die $!;
    close $f or die $!;
}
sub read_json {
    my ($p)=@_;
    open my $f,'<:raw',$p or die $!;
    local $/; my $r=decode_json(<$f>); close $f or die $!; return $r;
}
my $epsilon=unpack('f<',pack('f<',1e-6));
my $complete={schema=>'flash-sass.qsa-layer-capture.v1',positions=>11,layer=>3,die=>0,
    budget=>1,epsilon=>$epsilon,byte_order=>'little',snapshot_lengths=>[11,1],
    input_boundary=>'post_projection_before_norm_rope_and_kv_quantization',
    gamma=>'supplied_multiplicative_f32',rope_tables=>'supplied_f32',
    teacher_forced=>JSON::PP::true,pool_dtype=>'fp16',rope_base=>10000000,
    FLASH_KV_HOST=>0,build_revision=>'analytic_cpu_test',evidence_kind=>'analytic_cpu_test'};
my $snapshot={schema=>'flash-sass.qsa-layer-snapshot.v1',length=>11,position=>10,layer=>3,die=>0,
    budget=>1,complete_blocks=>2,tail=>3,attended_keys=>7,sparse_exclusion=>JSON::PP::true,
    scores_source=>'gpu',selected_ids_source=>'gpu'};
sub metadata {
    put("$capture/complete.json",encode_json($complete));
    put("$capture/n11/snapshot.json",encode_json($snapshot));
}
metadata();
put("$capture/ids.u32",pack('V*',10..20));
my $raw_k=pack('f<*',(1)x(11*512));
put("$capture/raw_k.f32",$raw_k);
my (@raw_v,@vq,@vs);
for my $t (0..10) {
    for my $kv (0..1) {
        my $value=$t<4 ? 1000 : 2*($t-3)+10*$kv;
        push @raw_v,($value)x256;
        push @vq,($t<4 ? 100 : $value)x256;
        push @vs,($t<4 ? 0x4900 : 0x3c00)x8;
    }
}
put("$capture/raw_v.f32",pack('f<*',@raw_v));
put("$capture/kn.f32",pack('f<*',(1)x256));
put("$capture/qn.f32",pack('f<*',(1)x256));
put("$capture/rope_cs.f32",pack('f<*',(1)x(11*32)));
put("$capture/rope_sn.f32",pack('f<*',(0)x(11*32)));
my $aout=pack('f<*',((4)x(12*256)),((9)x(12*256)));
my $scales=pack('v*',(0x3c00)x(11*16));
for my $n (1,11) {
    put("$capture/n$n/qg.f32",pack('f<*',(0)x12288));
    put("$capture/n$n/qo.f32",pack('f<*',(0)x6144));
    put("$capture/n$n/kq.i8",pack('c*',(1)x($n*512)));
    put("$capture/n$n/vq.i8",pack('c*',@vq[0..$n*512-1]));
    put("$capture/n$n/ks.f16",substr($scales,0,$n*32));
    put("$capture/n$n/vs.f16",pack('v*',@vs[0..$n*16-1]));
}
put("$capture/n11/aout.f32",$aout);
put("$capture/n11/selected_ids.u32",pack('V',1));
put("$capture/n1/aout.f32",pack('f<*',(500)x6144));
put("$capture/n1/selected_ids.u32",'');
put("$capture/n1/snapshot.json",encode_json({%$snapshot,length=>1,position=>0,
    complete_blocks=>0,tail=>1,attended_keys=>1,sparse_exclusion=>JSON::PP::false,
    scores_source=>'not_dispatched',selected_ids_source=>'implicit_all_blocks'}));
my $limits={schema=>'flash-sass.qsa-attention-limits.v1',
    arithmetic_max_abs=>0,arithmetic_rmse=>0,arithmetic_relative_l2=>0,query_max_abs=>0};
put("$tmp/limits.json",encode_json($limits));
sub run {
    my ($n,@options)=@_;
    my $out="$tmp/result".++$seq.".json";
    my $pid=fork(); defined($pid) or die 'fork';
    if (!$pid) {
        open STDOUT,'>',"$tmp/stdout" or die $!;
        open STDERR,'>',"$tmp/stderr" or die $!;
        exec {$^X} $^X,"$FindBin::Bin/qsa_attention_replay.pl",@options,$capture,$n,$out;
        die "exec: $!";
    }
    waitpid($pid,0);
    die 'child signaled' if $?&127;
    return ($?>>8,(-f $out ? read_json($out) : undef),$out);
}
my ($rc,$r,$out)=run(11);
if (!defined $r) {
    open my $error,'<',"$tmp/stderr" or die $!;
    diag do { local $/; <$error> };
    close $error;
    BAIL_OUT('valid analytic fixture did not produce a report');
}
is($rc,0,'analytic same-selection attention replay completes');
is($r->{status},'diagnostic_only','no limits means diagnostic only');
ok(!$r->{quality}&&!$r->{qsa_quality_validated}&&!$r->{independent_checkpoint_reference},
    'analytic replay does not certify checkpoint quality');
ok($r->{input_binding}{before_after_hashes_match},'metadata, arrays and tools are bound');
is_deeply($r->{reference_result}{attended_token_ids},[4..10],'selected block and three tail tokens, in order');
is($r->{reference_result}{arithmetic_vs_gpu}{max_abs},0,'analytic output 4 and 9 is exact');
is($r->{reference_result}{query_vs_gpu}{max_abs},0,'zero query normalization is exact');
is(scalar @{$r->{reference_result}{modes}},5,'missing FP32 K control leaves five modes');
ok(!$r->{quantization_effects}{k_storage_only_measured},'pure K-storage isolation is not inferred');
is($r->{replay}{epsilon},$epsilon,'recorded FP32 epsilon is preserved');
($rc,$r)=run(11,'--limits',"$tmp/limits.json");
is($rc,0,'exact analytic numerical limits pass');
is($r->{status},'arithmetic_limits_met','gated status is explicit');
ok(!$r->{limits}{historical_predeclaration_attested},'a profile hash does not attest historical predeclaration');
($rc,$r)=run(1,'--limits',"$tmp/limits.json");
is($rc,0,'N=1, no complete blocks, single tail token passes');
is_deeply($r->{reference_result}{selected_ids},[],'N=1 block selection is empty');
is_deeply($r->{reference_result}{attended_token_ids},[0],'N=1 still attends its one token');
put("$capture/n11/aout.f32",pack('f<',5).substr($aout,4));
($rc,$r)=run(11,'--limits',"$tmp/limits.json");
is($rc,3,'altered finite output fails declared numerical gates');
ok($r->{complete}&&@{$r->{limits}{failed}},'failed gates preserve the complete comparison');
is($r->{reference_result}{arithmetic_vs_gpu}{max_abs},1,'known one-element error is measured');
($rc,$r)=run(11);
is($rc,0,'same finite discrepancy remains diagnostic without limits');
put("$capture/n11/aout.f32",pack('f<*',(0)x6144));
put("$tmp/limits.json",encode_json({%$limits,arithmetic_max_abs=>100,
    arithmetic_rmse=>100,arithmetic_relative_l2=>1e9}));
($rc,$r)=run(11,'--limits',"$tmp/limits.json");
is($rc,3,'nonzero error against a zero reference fails any finite relative-L2 gate');
is_deeply([map { $_->{field} } @{$r->{limits}{failed}}],['arithmetic_relative_l2'],
    'undefined relative-L2 failure remains distinct from absolute and RMS gates');
put("$tmp/limits.json",encode_json($limits));
put("$capture/n11/aout.f32",$aout);
$complete->{normalized_k_available}=JSON::PP::true;
$complete->{prequant_k_source}='separate_gpu_fp32_append';
put("$capture/normalized_k.f32",$raw_k); metadata();
($rc,$r)=run(11,'--limits',"$tmp/limits.json");
is($rc,0,'optional normalized K control runs');
is(scalar @{$r->{reference_result}{modes}},7,'optional K control adds two labeled arms');
ok($r->{quantization_effects}{gpu_fp32_k_append_comparison_measured},
    'separate FP32 K append comparison is reported');
ok(!$r->{quantization_effects}{k_storage_only_measured},
    'separate append does not prove pure K-storage isolation');
delete $complete->{normalized_k_available}; delete $complete->{prequant_k_source}; metadata();
($rc,$r)=run(11);
ok($rc==2&&!defined($r),'undeclared optional normalized K is rejected');
unlink "$capture/normalized_k.f32" or die $!;
$complete->{prequant_k_source}='separate_gpu_fp32_append'; metadata();
($rc,$r)=run(11);
ok($rc==2&&!defined($r),'FP32 append declaration without its artifact is rejected');
delete $complete->{prequant_k_source};
$snapshot->{teacher_forced}=JSON::PP::false; metadata();
($rc,$r)=run(11);
ok($rc==2&&!defined($r),'contradictory snapshot teacher-forcing declaration is rejected');
delete $snapshot->{teacher_forced}; metadata();
for my $bad (
    {%$limits,unknown=>0}, {%$limits,query_max_abs=>'0'}, {%$limits,query_max_abs=>JSON::PP::false},
    {%$limits,query_max_abs=>-1}, {schema=>$limits->{schema},arithmetic_max_abs=>0}) {
    put("$tmp/limits.json",encode_json($bad));
    ($rc,$r)=run(11,'--limits',"$tmp/limits.json");
    ok($rc==2&&!defined($r),'malformed or incomplete numerical limits are rejected');
}
put("$tmp/limits.json",'{"schema":"flash-sass.qsa-attention-limits.v1","query_max_abs":0,"query_max_abs":1}');
($rc,$r)=run(11,'--limits',"$tmp/limits.json");
ok($rc==2&&!defined($r),'duplicate JSON keys cannot change a declared gate');
put("$tmp/limits.json",encode_json($limits));
put("$capture/raw_k.f32",substr($raw_k,0,-4));
($rc,$r)=run(11);
ok($rc==2&&!defined($r),'truncated raw prefix is rejected');
put("$capture/raw_k.f32",pack('V',0x7fc00000).substr($raw_k,4));
($rc,$r)=run(11);
ok($rc==2&&!defined($r),'nonfinite raw input is rejected even in an excluded token');
put("$capture/raw_k.f32",$raw_k);
put("$capture/n11/ks.f16",pack('v',0x7c00).substr($scales,2));
($rc,$r)=run(11);
ok($rc==2&&!defined($r),'nonfinite FP16 scale in excluded token is rejected');
put("$capture/n11/ks.f16",pack('v',0xbc00).substr($scales,2));
($rc,$r)=run(11);
ok($rc==2&&!defined($r),'negative cache scale is rejected');
put("$capture/n11/ks.f16",$scales);
put("$capture/n11/selected_ids.u32",pack('V',2));
($rc,$r)=run(11);
ok($rc==2&&!defined($r),'selected ID outside complete blocks is rejected');
put("$capture/n11/selected_ids.u32",pack('V',1));
$snapshot->{sparse_exclusion}='true'; metadata();
($rc,$r)=run(11);
ok($rc==2&&!defined($r),'a string cannot replace a sparse-dispatch boolean');
$snapshot->{sparse_exclusion}=JSON::PP::true;
$complete->{byte_order}='big'; metadata();
($rc,$r)=run(11);
ok($rc==2&&!defined($r),'conflicting byte order is rejected');
$complete->{byte_order}='little'; metadata();
$complete->{budget}=2; $snapshot->{budget}=2; $snapshot->{attended_keys}=11;
$snapshot->{sparse_exclusion}=JSON::PP::false; $snapshot->{scores_source}='not_dispatched';
$snapshot->{selected_ids_source}='implicit_all_blocks'; metadata();
for my $ids ([0,0],[1,0]) {
    put("$capture/n11/selected_ids.u32",pack('V*',@$ids));
    ($rc,$r)=run(11);
    ok($rc==2&&!defined($r),'dense selection rejects duplicate or reordered implicit IDs');
}
my $existing="$tmp/existing.json"; put($existing,"sentinel\n");
my $pid=fork(); defined($pid) or die 'fork';
if (!$pid) {
    open STDOUT,'>',"$tmp/stdout" or die $!;
    open STDERR,'>',"$tmp/stderr" or die $!;
    exec {$^X} $^X,"$FindBin::Bin/qsa_attention_replay.pl",$capture,11,$existing;
    die $!;
}
waitpid($pid,0); is($?>>8,2,'existing output is rejected before replay');
open my $f,'<',$existing or die $!; my $bytes=do { local $/; <$f> }; close $f;
is($bytes,"sentinel\n",'existing output remains unchanged');
done_testing();
