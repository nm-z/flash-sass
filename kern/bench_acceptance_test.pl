#!/usr/bin/env perl
use strict;
use warnings;
use FindBin;
use lib $FindBin::Bin;
use File::Temp qw(tempdir);
use JSON::PP;
use Test::More;
use BenchAcceptance;
require "$FindBin::Bin/paired_depth_bench.pl";
sub rejects {
    my($code,$label,$pattern)=@_;
    my $ok=eval{$code->();1};
    ok(!$ok,$label);
    like($@,$pattern,"$label: reason") if $pattern;
}
sub copy { JSON::PP->new->decode(JSON::PP->new->encode($_[0])) }
sub metadata {
    return {schema=>'1',clock=>'CLOCK_MONOTONIC',request=>'2',status=>'completed',
        aborted=>'0',ended=>'0',completed=>'11',requested=>'11',prompt=>'10',
        resumed=>'9',snapshot=>'0',ple_short_reads=>'0',
        request_start_ns=>'1000',build_start_ns=>'1100',build_end_ns=>'1200',
        run_start_ns=>'1300',engine_prefix_observed_ns=>'0',
        engine_first_observed_ns=>'1400',engine_last_observed_ns=>'1500',request_end_ns=>'1600'};
}
subtest 'complete policy cannot be confused with a short experiment' => sub {
    my $a={outputs=>129,pairs=>5,warmup_outputs=>9};
    ok(BenchAcceptance::policy($a,\@BenchAcceptance::DEPTHS)->{standard_matrix},'required matrix');
    ok(!BenchAcceptance::policy({%$a,outputs=>2},\@BenchAcceptance::DEPTHS)->{standard_matrix},'two outputs are custom');
    ok(!BenchAcceptance::policy($a,[2048])->{standard_matrix},'one depth is custom');
    rejects(sub{BenchAcceptance::policy({%$a,pairs=>4},[2048])},'fewer than five pairs rejected');
};
subtest 'untraced lifecycle and timings' => sub {
    my $m=metadata();
    my $t=BenchAcceptance::requestmeta($m,10,2,1);
    is($t->{engine_observed_decode_ms},0.0001,'first-to-last host observation');
    ok(!defined($t->{prefill_device_ms}),'device prefill remains unmeasured');
    for my $change ({status=>'aborted',aborted=>'1'},{ended=>'1'},{ple_short_reads=>'1'},
                    {completed=>'10'},{requested=>'12'},{schema=>'0'},{engine_last_observed_ns=>'1399'}) {
        rejects(sub{BenchAcceptance::requestmeta({%$m,%$change},10,2,1)},'bad live metadata rejected');
    }
    rejects(sub{BenchAcceptance::requestmeta($m,10,2,2)},'stale prior request rejected',qr/stale/);
    my $missing=copy($m);delete $missing->{ple_short_reads};
    rejects(sub{BenchAcceptance::requestmeta($missing,10,2,1)},'missing I/O error counter rejected');
};
subtest 'capability gate distinguishes modes before requests' => sub {
    my $c={requestmeta_schema=>'1',tensor_schema=>'1',rev=>'abc',maxt=>'8192',layers=>'48/48',nblk=>'512',
        dies=>'6',kv_host=>'1',ple_disk=>'1',attn_dies=>'1,4,5',split=>'same',
        attn_map=>'same',cubin_fnv=>'123',trace_tokens=>'0'};
    BenchAcceptance::capabilities({host=>$c,vram=>$c},0,129);pass('untraced current metadata supported');
    rejects(sub{BenchAcceptance::capabilities({host=>$c,vram=>$c},1,129)},'missing schema 2 cannot be attribution');
    my $legacy={%$c};delete $legacy->{requestmeta_schema};
    rejects(sub{BenchAcceptance::capabilities({host=>$legacy,vram=>$c},0,129)},'legacy untraced engine cannot pass');
    my $instrumented={%$c,trace_tokens=>'129',trace_ctas=>'1',trace_schema=>'2'};
    rejects(sub{BenchAcceptance::capabilities({host=>$instrumented,vram=>$instrumented},0,129)},'instrumented image cannot be throughput');
    BenchAcceptance::capabilities({host=>$instrumented,vram=>$instrumented},1,129);pass('attribution mode requires complete capability');
};
subtest 'independent reference identity and invalid answer gate' => sub {
    my $ids=[(1)x10];
    my $ref={schema=>1,prompt_ids_sha256=>BenchAcceptance::canonical_hash($ids),
        provenance=>{method=>'independent_reference_tokens',source_sha=>'a'x40,
            artifact_sha256=>'b'x64,description=>'CPU parser fixture, not model evidence'},
        depths=>{10=>{prompt_sha256=>BenchAcceptance::canonical_hash($ids),
            generated_ids=>[3,4],valid=>JSON::PP::true}}};
    BenchAcceptance::answer_reference($ref,$ids,2);
    ok(BenchAcceptance::check_answer($ref,10,{generated_ids=>[3,4]}),'exact reference match');
    rejects(sub{BenchAcceptance::check_answer($ref,10,{generated_ids=>[3,5]})},'different output rejected');
    rejects(sub{BenchAcceptance::check_answer($ref,9,{generated_ids=>[3,4]})},'missing reference depth rejected');
    my $bad=copy($ref);$bad->{depths}{10}{valid}=JSON::PP::false;
    rejects(sub{BenchAcceptance::answer_reference($bad,$ids,2)},'invalid answer cannot be a reference');
    $bad=copy($ref);$bad->{prompt_ids_sha256}='c'x64;
    rejects(sub{BenchAcceptance::answer_reference($bad,$ids,2)},'wrong prompt identity rejected');
};
{
    package FixtureCore;
    sub new {my($class,$lines,$meta)=@_;bless{lines=>[@$lines],meta=>$meta,last_request=>1},$class}
    sub one {
        my($self,$command)=@_;
        return 'ok: snap 9' if $command eq 'snap 9';
        if($command eq 'requestmeta') {
            return 'requestmeta '.join(' ',map {"$_=$self->{meta}{$_}"} sort keys %{$self->{meta}});
        }
        die "unexpected fixture command $command\n";
    }
    sub send {$_[0]{command}=$_[1]}
    sub line {my($self)=@_;die "fixture disconnected\n" unless @{$self->{lines}};shift @{$self->{lines}}}
}
subtest 'partial raw responses survive request failures' => sub {
    my $dir=tempdir(CLEANUP=>1);
    my $core=FixtureCore->new(['tok 3','error: device fault'],metadata());
    rejects(sub{PairedDepthBench::request($core,[(1)x10],2,"$dir/error",0,6,9)},'device error propagates');
    my $r=PairedDepthBench::load_json("$dir/error.json");
    is_deeply($r->{raw},['tok 3','error: device fault'],'error line and preceding token preserved');
    is_deeply($r->{generated_ids},[3],'partial generated tokens preserved');
    is($r->{status},'failed','request excluded');
    $core=FixtureCore->new(['tok 3'],metadata());
    rejects(sub{PairedDepthBench::request($core,[(1)x10],2,"$dir/disconnect",0,6,9)},'disconnect propagates');
    $r=PairedDepthBench::load_json("$dir/disconnect.json");
    is_deeply($r->{raw},['tok 3'],'partial disconnect record retained');
};
subtest 'ordinary untraced completion still requires requestmeta' => sub {
    no warnings qw(redefine once);
    local *PairedDepthBench::gpu=sub{{exit=>0,stdout=>'CPU test fixture only'}};
    my $dir=tempdir(CLEANUP=>1);
    my $core=FixtureCore->new(['tok 3','tok 4','done 2 0 10 1.0 0.5 9 0'],metadata());
    my $r=PairedDepthBench::request($core,[(1)x10],2,"$dir/good",0,6,9);
    is($r->{status},'complete','protocol collection succeeds');
    ok($r->{lifecycle}{reported},'lifecycle checked with tracing disabled');
    is(scalar(@{$r->{intervals_ns}}),1,'two outputs give one interval');
    my $bad=metadata();$bad->{status}='device_error';
    $core=FixtureCore->new(['tok 3','tok 4','done 2 0 10 1.0 0.5 9 0'],$bad);
    rejects(sub{PairedDepthBench::request($core,[(1)x10],2,"$dir/bad",0,6,9)},'legacy done line cannot override failed metadata');
};
subtest 'aggregate excludes unvalidated answers and incomplete pairs' => sub {
    my @rows;
    for my $p(qw(host vram)) {
        push @rows,map{{depth=>2048,placement=>$p,pair=>$_,status=>'complete',
            answer_valid=>JSON::PP::true,intervals_ns=>[1000],decode_intervals=>1,client_decode_ms=>0.001}}0..4;
    }
    is(scalar(@{BenchAcceptance::summary(\@rows,[2048],5,1)}),2,'two validated placements');
    $rows[0]{answer_valid}=undef;
    rejects(sub{BenchAcceptance::summary(\@rows,[2048],5,1)},'unknown answer excluded');
    $rows[0]{answer_valid}=JSON::PP::true;pop @rows;
    rejects(sub{BenchAcceptance::summary(\@rows,[2048],5,1)},'missing pair excluded');
};
subtest 'initialization failure leaves truthful process exit and files' => sub {
    my $dir=tempdir(CLEANUP=>1);my $out="$dir/failed";
    my $r=PairedDepthBench::run([$^X,"$FindBin::Bin/paired_depth_bench.pl",'--out',$out]);
    is($r->{exit},1,'CLI exits one on required-metadata failure');
    is(PairedDepthBench::load_json("$out/exit.json")->{exit},1,'saved code matches actual process exit');
    ok(-f "$out/FAILED.json",'failure reason preserved');
    is(PairedDepthBench::load_json("$out/manifest.json")->{state},'failed','initial manifest preserved');
    ok(!-e "$out/summary.json",'no successful aggregate');
};
subtest 'all attention layers and positions must execute default QSA' => sub {
    my @keys;
    for my $p(9,10){push @keys,map{{position=>$p,layer=>4*$_+3,mode=>'qsa'}}0..11;}
    my $tr=[{metadata=>{nblk=>512},selected_keys=>\@keys}];
    is(BenchAcceptance::attention_work($tr,10,2)->{layer_positions},24,'complete sampled work');
    my $bad=copy($tr);pop @{$bad->[0]{selected_keys}};
    rejects(sub{BenchAcceptance::attention_work($bad,10,2)},'missing attention layer rejected');
    $bad=copy($tr);$bad->[0]{selected_keys}[0]{mode}='dense';
    rejects(sub{BenchAcceptance::attention_work($bad,10,2)},'dense cannot be labeled default QSA');
    $bad=copy($tr);$bad->[0]{metadata}{nblk}=4096;
    rejects(sub{BenchAcceptance::attention_work($bad,10,2)},'larger budget cannot be default QSA');
};
subtest 'separate evidence is revalidated and copied into the run' => sub {
    my $dir=tempdir(CLEANUP=>1);my $other="$dir/other";my $out="$dir/out";
    File::Path::make_path($other,$out);
    my $config={dies=>'1',nblk=>'512',kv_host=>'1',trace_tokens=>'0'};
    my $launch={launch_environment=>{FLASH_TRACE=>'0',FLASH_TRACE_CTAS=>'1',FLASH_CUBIN=>'plain.cubin'},
        engine_reload_commands=>[],ple_cache_conditions=>{mode=>'CPU fixture only'}};
    my $m={source_sha=>'a'x40,source_dirty_sha256=>'b'x64,prompt_ids_sha256=>'c'x64,
        weights=>{fixture=>'CPU only'},tensors=>{fixture=>'CPU only'},sampling=>{seed=>0},
        policy=>{outputs=>2},configs=>{host=>$config,vram=>{%$config,kv_host=>'0'}},
        provenance=>{artifacts=>{executable=>{sha256=>'d'x64},library=>{sha256=>'e'x64}},
            core_launch_records=>{host=>copy($launch),vram=>copy($launch)}}};
    my $tm=copy($m);$tm->{mode}='attribution';
    for my $p(qw(host vram)) {
        $tm->{configs}{$p}{trace_tokens}='2';
        $tm->{provenance}{core_launch_records}{$p}{launch_environment}{FLASH_TRACE}='2';
        $tm->{provenance}{core_launch_records}{$p}{launch_environment}{FLASH_CUBIN}='trace.cubin';
    }
    my @columns=qw(position phase layer opcode units flags selected_keys start_ns wait_start_ns wait_end_ns
        body_start_ns body_end_ns end_ns ple_wait_start_ns ple_wait_end_ns publication_ns cta
        send_before_ns send_after_ns exchange_generation);
    my @phase;
    for my $t(0,1) {
        for my $p(0..11) {
            my $s=10000+2000*$t+100*$p;
            push @phase,{position=>9+$t,phase=>$p,layer=>3+4*$p,opcode=>'attnpartn',
                units=>32,flags=>0,selected_keys=>10+$t,start_ns=>$s,wait_start_ns=>$s+1,
                wait_end_ns=>$s+2,body_start_ns=>$s+3,body_end_ns=>$s+7,end_ns=>$s+9,
                ple_wait_start_ns=>0,ple_wait_end_ns=>0,publication_ns=>$p==11?$s+10:0,
                cta=>0,send_before_ns=>0,send_after_ns=>0,exchange_generation=>0};
        }
    }
    my $meta={schema=>'2',nph=>'12',count=>'2',first=>'9',ctas=>'1',nblk=>'512',die=>'0',
        request=>'2',status=>'completed',aborted=>'0',ended=>'0',completed=>'11',requested=>'11'};
    my $device=PairedDepthBench::phase_summary(\@phase,$meta);
    my @rows;
    for my $p(qw(host vram)) {
        my $path="$other/$p.csv.gz";my $gz=IO::Compress::Gzip->new($path) or die $IO::Compress::Gzip::GzipError;
        print {$gz} join(',',@columns),"\n";
        for my $r(@phase){print {$gz} join(',',@$r{@columns}),"\n";}$gz->close or die 'close gzip';
        my $hostpath="$other/$p-host.csv.gz";
        $gz=IO::Compress::Gzip->new($hostpath) or die $IO::Compress::Gzip::GzipError;
        print {$gz} "position,observed_ns,callback_start_ns,callback_end_ns,ple_staging_start_ns,ple_staging_end_ns,ple_ready_ns,ple_short_reads\n";
        print {$gz} "9,1400,1401,1402,0,0,0,0\n10,1500,1501,1502,0,0,0,0\n";
        $gz->close or die 'close gzip';
        my $hm={%$meta,clock=>'CLOCK_MONOTONIC'};delete @$hm{qw(nph ctas nblk die)};
        push @rows,{depth=>10,pair=>0,placement=>$p,status=>'complete',generated_ids=>[3,4],
            client_observed_ns=>[1400,1500],intervals_ns=>[100],decode_intervals=>1,client_decode_ms=>0.0001,
            engine_observed_decode_ms=>0.0001,requestmeta=>metadata(),done=>{resumed=>9},
            device_publication_decode_ms_by_die=>[$device->{decode_ms}],
            trace_files=>[{kind=>'trace',die=>0,raw=>PairedDepthBench::identity($path),metadata=>$meta},
                {kind=>'hosttrace',raw=>PairedDepthBench::identity($hostpath),metadata=>$hm}]};
    }
    PairedDepthBench::save("$other/manifest.json",$tm);
    PairedDepthBench::save("$other/status.json",{collection_complete=>JSON::PP::true});
    PairedDepthBench::save("$other/results.json",\@rows);
    my $c=BenchAcceptance::compare_attribution($out,$m,\@rows,$other);
    is(scalar(@$c),2,'matched raw attribution accepted');
    ok(-f "$out/attribution/host.csv.gz" && -f "$out/attribution/vram-host.csv.gz",'raw files copied');
    my $bad=copy($tm);$bad->{provenance}{artifacts}{library}{sha256}='f'x64;
    PairedDepthBench::save("$other/manifest.json",$bad);
    rejects(sub{BenchAcceptance::compare_attribution($out,$m,\@rows,$other)},'different actual library rejected');
    PairedDepthBench::save("$other/manifest.json",$tm);
    my $badrows=copy(\@rows);$badrows->[0]{client_decode_ms}=99;
    PairedDepthBench::save("$other/results.json",$badrows);
    rejects(sub{BenchAcceptance::compare_attribution($out,$m,\@rows,$other)},'stored duration cannot override raw timestamps');
    PairedDepthBench::save("$other/results.json",\@rows);
    open my $f,'>>:raw',"$other/host.csv.gz" or die $!;print {$f} 'changed';close $f;
    rejects(sub{BenchAcceptance::compare_attribution($out,$m,\@rows,$other)},'changed raw trace rejected');

    my $raw=PairedDepthBench::identity("$other/vram.csv.gz");
    my $prof={schema=>1,scope=>'separate_profiler',map {$_=>$m->{$_}} qw(source_sha source_dirty_sha256 prompt_ids_sha256 weights sampling policy configs)};
    $prof->{instrumentation}={tool=>'CPU parser fixture only'};
    $prof->{commands}=[{argv=>['fixture-counter-collector'],exit=>0}];
    $prof->{raw_files}=[$raw];$prof->{counters}=[{name=>'fixture_counter',unit=>'count',value=>1}];
    BenchAcceptance::profiler_record($prof,$m,$out);
    ok(-f "$out/profiler/1-vram.csv.gz",'profiler raw file copied');
    $bad=copy($prof);$bad->{commands}[0]{exit}='not-an-exit';
    rejects(sub{BenchAcceptance::profiler_record($bad,$m)},'nonnumeric profiler exit rejected');
    $bad=copy($prof);$bad->{counters}[0]{value}='NaN';
    rejects(sub{BenchAcceptance::profiler_record($bad,$m)},'nonfinite counter rejected');
};
done_testing;
