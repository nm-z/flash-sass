#!/usr/bin/env perl
use strict;use warnings;use FindBin;use File::Temp 'tempdir';use Getopt::Long 'GetOptionsFromArray';
use Digest::SHA 'sha256_hex';use JSON::PP;use Test::More;
use IO::Compress::Gzip qw($GzipError);
my $prompt_fixture;GetOptionsFromArray(\@ARGV,'prompt-fixture=s'=>\$prompt_fixture) or die "invalid test options\n";die "unexpected arguments: @ARGV\n" if @ARGV;
require "$FindBin::Bin/paired_depth_bench.pl";
sub clone{return JSON::PP->new->decode(JSON::PP->new->encode($_[0]));}
sub rejects{my($code,$name)=@_;my $ok=eval{$code->();1};ok(!$ok,$name);}
sub meta{return {first=>'2048',count=>'2',nph=>'2',ctas=>'1'};}
sub current_meta{return {%{meta()},status=>'completed',aborted=>'0',ended=>'0',completed=>'2050',requested=>'2050'};}
sub fixture{
 my @rows;for my $i(0..3){my $start=100+$i*50+($i>=2?10:0);my $r={position=>''.(2048+int($i/2)),phase=>''.($i%2),opcode=>$i%2?'PLEROWS':'IDXSEL',selected_keys=>''.(2049+int($i/2)),ple_wait_start_ns=>'0',ple_wait_end_ns=>'0',publication_ns=>$i%2?''.($start+50):'0'};my @k=qw(start_ns wait_start_ns wait_end_ns body_end_ns end_ns);$r->{$k[$_]}=''.($start+$_*10) for 0..4;if($i==3){$r->{ple_wait_start_ns}=''.($start+22);$r->{ple_wait_end_ns}=''.($start+26);}push @rows,$r;}return \@rows;
}
sub host_rows{return [map {{position=>''.(2048+$_),observed_ns=>''.(100+10*$_),callback_start_ns=>''.(101+10*$_),callback_end_ns=>''.(102+10*$_),ple_staging_start_ns=>'0',ple_staging_end_ns=>'0',ple_ready_ns=>'0',ple_short_reads=>'0'}}0..1];}
subtest 'partition and phase zero'=>sub{
 my $r=PairedDepthBench::phase_summary(fixture(),meta());is_deeply($r->{intervals_ns},[110],'publication interval');is($r->{tokens}[1]{phase_ns},80,'phase time');is($r->{tokens}[1]{between_phase_ns},10,'between phases');is($r->{tokens}[1]{publication_footer_ns},10,'publication footer');is_deeply($r->{op_counts},{IDXSEL=>1,PLEROWS=>1},'first publication position excluded from decode work');is($r->{op_mean_us}{PLEROWS}{ple_wait},.004,'PLE wait partition');
};
subtest '64 bit epoch'=>sub{my $r=fixture();for my $row(@$r){for my $k(keys %$row){$row->{$k}=''.($row->{$k}+4294967296) if $k=~/_ns\z/ && $row->{$k};}}is_deeply(PairedDepthBench::phase_summary($r,meta())->{intervals_ns},[110],'interval independent of absolute epoch');};
subtest 'missing phase'=>sub{my $r=fixture();pop @$r;rejects(sub{PairedDepthBench::phase_summary($r,meta())},'missing rejected');};
subtest 'duplicate phase'=>sub{my $r=fixture();$r->[1]=$r->[0];rejects(sub{PairedDepthBench::phase_summary($r,meta())},'duplicate rejected');};
subtest 'missing publication'=>sub{my $r=fixture();$r->[-1]{publication_ns}='0';rejects(sub{PairedDepthBench::phase_summary($r,meta())},'missing publication rejected');};
subtest 'wrong layout'=>sub{rejects(sub{PairedDepthBench::phase_summary(fixture(),{%{meta()},ctas=>'16'})},'16 CTA layout rejected');};
subtest 'invalid PLE wait'=>sub{my $r=fixture();$r->[-1]{ple_wait_end_ns}=$r->[-1]{publication_ns};rejects(sub{PairedDepthBench::phase_summary($r,meta())},'wait outside body rejected');};
subtest 'single publication'=>sub{my $rows=fixture();splice @$rows,2;my $r=PairedDepthBench::phase_summary($rows,{%{meta()},count=>'1'});is_deeply($r->{intervals_ns},[],'no intervals');is($r->{decode_ms},0,'zero decode duration');};
subtest 'manifest gate'=>sub{
 PairedDepthBench::matched({kv_host=>'0',layers=>'48/48'},{kv_host=>'1',layers=>'48/48'},{kv_host=>1});pass('KV placement difference allowed');
 rejects(sub{PairedDepthBench::matched({kv_host=>'0',layers=>'12/48'},{kv_host=>'1',layers=>'48/48'},{kv_host=>1})},'unexplained configuration difference rejected');
};
subtest 'failed driver query cannot pass'=>sub{
 {no warnings 'redefine';local *PairedDepthBench::run=sub{return {exit=>2,stdout=>'Failed to parse flags',stderr=>''};};rejects(sub{PairedDepthBench::driver_snapshot()},'query failure rejected');}
 {no warnings 'redefine';local *PairedDepthBench::run=sub{return {exit=>0,stdout=>'<nvidia_smi_log></nvidia_smi_log>',stderr=>''};};is(PairedDepthBench::driver_snapshot()->{exit},0,'valid XML accepted');}
};
subtest 'core launch records'=>sub{
 my $r={argv=>['flashcore','-','libengine.so','manifest'],launch_environment=>{FLASH_MAXT=>'8192',CUDA_VISIBLE_DEVICES=>'0,1',CUDA_DEVICE_ORDER=>'PCI_BUS_ID'},engine_reload_commands=>[],limitations=>[]};my $d=tempdir(CLEANUP=>1);my $p="$d/launch.json";PairedDepthBench::save($p,{host=>$r,vram=>$r});is(PairedDepthBench::core_records($p)->{host}{argv}[0],'flashcore','valid launch metadata retained');my $bad=clone($r);delete $bad->{launch_environment}{FLASH_MAXT};PairedDepthBench::save($p,{host=>$bad,vram=>$r});rejects(sub{PairedDepthBench::core_records($p)},'missing setting rejected');
};
subtest 'host short read'=>sub{
 my $r={position=>'2048',observed_ns=>'100',callback_start_ns=>'101',callback_end_ns=>'102',ple_staging_start_ns=>'1',ple_staging_end_ns=>'2',ple_ready_ns=>'3',ple_short_reads=>'1'};
 rejects(sub{PairedDepthBench::host_summary([$r],{first=>'2048',count=>'1',clock=>'CLOCK_MONOTONIC'})},'short read invalidates measurement');
};
subtest 'legacy lifecycle is analysis only'=>sub{
 my $r=PairedDepthBench::phase_summary(fixture(),meta());is_deeply($r->{lifecycle},{reported=>JSON::PP::false,status=>'legacy_unreported'},'absent historical lifecycle remains unknown');rejects(sub{PairedDepthBench::trace_lifecycle(meta(),require_current=>1)},'live lifecycle required');
};
subtest 'current lifecycle success'=>sub{
 my $m=current_meta();my $r=PairedDepthBench::trace_lifecycle($m,require_current=>1,expected_requested=>2050,allow_ended=>0);ok($r->{reported},'reported success');is(PairedDepthBench::phase_summary(fixture(),$m)->{lifecycle}{completed},2050,'completed positions retained');
};
subtest 'failed lifecycle cannot be historical success'=>sub{
 for my $status(qw(aborted argument_error image_error device_error running none)){my $m={%{meta()},status=>$status,aborted=>$status eq 'aborted'?'1':'0',ended=>'1',completed=>'2050',requested=>'2050'};rejects(sub{PairedDepthBench::phase_summary(fixture(),$m)},"$status phase rejected");rejects(sub{PairedDepthBench::host_summary(host_rows(),{%$m,clock=>'CLOCK_MONOTONIC'})},"$status host rejected");}
};
subtest 'stop analyzable but not fixed continuation'=>sub{
 my $m={%{meta()},count=>'1',status=>'completed',aborted=>'0',ended=>'1',completed=>'2049',requested=>'2050'};my $r=fixture();splice @$r,2;is(PairedDepthBench::phase_summary($r,$m)->{lifecycle}{ended},1,'stopped historical trace analyzable');rejects(sub{PairedDepthBench::trace_lifecycle($m,require_current=>1,allow_ended=>0)},'fixed continuation rejects stop');
};
subtest 'partial or malformed lifecycle'=>sub{
 rejects(sub{PairedDepthBench::trace_lifecycle({%{meta()},status=>'completed'})},'partial metadata rejected');
 for my $change({aborted=>'2'},{ended=>'false'},{completed=>'-1'},{requested=>'nan'},{completed=>'2049'},{completed=>'2051'}){rejects(sub{PairedDepthBench::trace_lifecycle({%{current_meta()},%$change})},'malformed lifecycle '.JSON::PP->new->encode($change));}
};
subtest 'wrong request or excessive window'=>sub{
 my $m=current_meta();rejects(sub{PairedDepthBench::trace_lifecycle($m,require_current=>1,expected_requested=>2051)},'wrong request rejected');rejects(sub{PairedDepthBench::trace_lifecycle({%$m,count=>'3'})},'window beyond completed positions rejected');
};
subtest 'CSV quoting and gzip'=>sub{
 is_deeply(PairedDepthBench::csv_fields('1,"a,b","a""b",'),['1','a,b','a"b',''],'CSV quoting');my $d=tempdir(CLEANUP=>1);my $p="$d/trace.csv.gz";my $text="position,opcode\n1,\"A,B\"\n";IO::Compress::Gzip::gzip(\$text=>$p) or die $GzipError;is_deeply(PairedDepthBench::read_csv_gzip($p),[{position=>'1',opcode=>'A,B'}],'compressed rows retained');
};
subtest 'subprocess capture'=>sub{
 my $r=PairedDepthBench::run([$^X,'-e','print STDOUT "out";print STDERR "err";exit 7']);is($r->{exit},7,'exit status');is($r->{stdout},'out','stdout');is($r->{stderr},'err','stderr');
};
subtest 'subprocess drains large stderr'=>sub{
 my $r=PairedDepthBench::run([$^X,'-e','print STDERR "e"x200000;print STDOUT "done"']);is($r->{exit},0,'large-output child exits');is(length($r->{stderr}),200000,'all stderr captured');is($r->{stdout},'done','stdout not blocked');
};
subtest 'exact file identity'=>sub{
 my $d=tempdir(CLEANUP=>1);my $p="$d/artifact";open my $f,'>:raw',$p or die $!;print {$f} 'artifact';close $f;my $r=PairedDepthBench::identity($p);is($r->{bytes},8,'size');is($r->{sha256},sha256_hex('artifact'),'SHA256');ok($r->{mtime_ns}=~/\A[0-9]+\z/,'integer nanosecond timestamp');
};
subtest 'committed historical prompt identity'=>sub {
 my $golden=PairedDepthBench::load_json("$FindBin::Bin/../validation/issue14/prompt-fixture-identity.json");
 my $actual=PairedDepthBench::prompt_ids($golden->{count});
 is_deeply([@$actual[0..31]],$golden->{first_32},'known first 32 IDs');
 is(sha256_hex(JSON::PP->new->canonical->encode($actual)),$golden->{canonical_array_sha256},'all 262000 historical IDs');
};
SKIP:{
 skip 'supply --prompt-fixture for historical full-array regression',3 unless defined $prompt_fixture;
 my $expected=PairedDepthBench::load_json($prompt_fixture);my $actual=PairedDepthBench::prompt_ids(scalar @$expected);my $j=JSON::PP->new->canonical;
 is(scalar(@$actual),scalar(@$expected),'historical prompt length');is_deeply([@$actual[0..31]],[@$expected[0..31]],'historical first 32 prompt IDs');is(sha256_hex($j->encode($actual)),sha256_hex($j->encode($expected)),'every historical prompt ID matches');
}
done_testing;
