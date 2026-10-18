#!/usr/bin/env perl
use strict;
use warnings;
use Fcntl qw(LOCK_EX LOCK_NB LOCK_UN);
use File::Path qw(make_path);
use JSON::PP;
use POSIX qw(strftime);
use lib '/home/nate/codex/flash-issue14-20261018/kern';
require '/home/nate/codex/flash-issue14-20261018/kern/paired_depth_bench.pl';

my $root='/home/nate/codex/flash-issue14-20261018';
my $out='/home/nate/codex/flash-issue14-evidence-20261018/resume/final-cpu';
die "output exists\n" if -e $out;
make_path($out);
chdir $root or die $!;
open my $lock,'>>','/tmp/flash-sass-engine-validation.lock' or die $!;
flock($lock,LOCK_EX|LOCK_NB) or die "BLOCKED: engi validation lock busy\n";
my $head=PairedDepthBench::run(['git','rev-parse','HEAD']);
my $sha=$head->{stdout};$sha=~s/\s+\z//;
my $status=PairedDepthBench::run(['git','status','--porcelain']);
die "source is not clean\n" if length($status->{stdout}) || $status->{exit};
my $record={scope=>'CPU regressions, C syntax, generated SASS and CUDA SDK-stub linkage only; no GPU or benchmark run',
    source_sha=>$sha,utc=>strftime('%Y-%m-%dT%H:%M:%SZ',gmtime),
    source_status_before=>$status,
    lock=>{path=>'/tmp/flash-sass-engine-validation.lock',acquired=>JSON::PP::true},
    commands=>[],preflight=>[],files=>{}};
my $bad=0;
sub command {
    my($name,$argv,$preflight)=@_;
    my $r=PairedDepthBench::run($argv);
    my $path="$out/$name.log";
    open my $f,'>:raw',$path or die $!;
    print {$f} 'argv: ',$PairedDepthBench::JSON->encode($argv),
        "\nexit: $r->{exit}\nstdout:\n$r->{stdout}\nstderr:\n$r->{stderr}" or die $!;
    close $f or die $!;
    push @{$record->{$preflight?'preflight':'commands'}},
        {argv=>$argv,exit=>$r->{exit},log=>"$name.log",identity=>PairedDepthBench::identity($path)};
    print "$name exit=$r->{exit}\n";
    $bad++ if !$preflight && $r->{exit};
    return $r;
}
command('hostname',['hostname'],1);
command('nvidia-smi',['bash','-c','nvidia-smi'],1);
command('processes',['ps','-eo','pid,ppid,stat,comm'],1);

command('perl-tests',['prove','-v','kern/paired_depth_bench_test.pl',
    'kern/bench_manifest_test.pl','kern/bench_trace_test.pl','kern/bench_acceptance_test.pl']);
command('requestmeta-build',['cc','-std=c11','-Wall','-Wextra','-Werror','-pedantic',
    'kern/request_meta_test.c','-o',"$out/request_meta_test"]);
command('requestmeta-test',["$out/request_meta_test"]);
command('generate-header',['make','-B','-C','kern','dieops.h']);
command('engine-syntax',['cc','-std=gnu99','-O2','-ffp-contract=off','-Wall','-Werror',
    '-fsyntax-only','-Iref','-Ikern','-I/opt/cuda/include','kern/engine.c']);
my $build=command('engine-integration-build',['make','-B','-C','kern',
    'bench_engine_trace_test','CC=cc -L/opt/cuda/lib64/stubs']);
if (!$build->{exit}) {
    make_path("$out/cuda-stub");
    symlink('/opt/cuda/lib64/stubs/libcuda.so',"$out/cuda-stub/libcuda.so.1") or die $!;
    command('engine-integration-test',['env',"LD_LIBRARY_PATH=$out/cuda-stub",
        "$root/kern/bench_engine_trace_test","$out/engine-cpu-trace.txt"]);
    my $parse=q{
        require "paired_depth_bench.pl";
        my @lines=split /\n/,PairedDepthBench::read_text($ARGV[0]);
        my $meta=PairedDepthBench::fields(shift @lines);
        die "missing footer\n" unless pop(@lines) eq "trace_end";
        my @columns=split /,/,shift @lines;
        my @rows=map {my @v=split /,/;die "columns\n" unless @v==@columns;
            my %r;@r{@columns}=@v;\%r} @lines;
        my $s=PairedDepthBench::phase_summary(\@rows,$meta);
        die "export window\n" unless $meta->{count}==129 && @rows==258;
        die "decode interval total\n" unless $s->{decode_ms}==0.128;
        print "Actual C export parsed: 129 outputs, 128 intervals, 258 phase rows; exact 0.128 ms CPU fixture span\n";
    };
    command('engine-export-parser',['perl','-Ikern','-e',$parse,"$out/engine-cpu-trace.txt"]);
}
command('trace-reconcile-tests',['ruby','kern/trace_reconcile_test.rb']);
command('generate-sass',['make','-B','-C','kern','die.sass','die_trace.sass','die_trace16.sass']);
command('engine-stub-link',['make','-B','-C','kern','libengine.so','CC=cc -L/opt/cuda/lib64/stubs']);
command('diff-check',['git','diff','--check']);
for my $path(qw(BENCHMARK.md docs/phase-trace.md kern/Makefile kern/engine.c kern/engine.h
    kern/paired_depth_bench.pl kern/BenchAcceptance.pm kern/BenchManifest.pm kern/BenchTrace.pm
    kern/bench_acceptance_test.pl kern/bench_manifest_test.pl kern/bench_trace_test.pl
    kern/request_meta.h kern/request_meta_test.c kern/bench_engine_trace_test.c
    kern/trace_host.inc kern/trace.pl kern/die.pl kern/trace_reconcile.rb kern/trace_reconcile_test.rb)) {
    $record->{files}{$path}=PairedDepthBench::identity($path);
}
for my $path(qw(kern/libengine.so kern/bench_engine_trace_test kern/die.sass kern/die_trace.sass kern/die_trace16.sass)) {
    $record->{artifacts}{$path}=PairedDepthBench::identity($path) if -f $path;
}
$record->{validator}=PairedDepthBench::identity($0);
$record->{cuda_stub}=PairedDepthBench::identity('/opt/cuda/lib64/stubs/libcuda.so');
$record->{cuda_link_mode}='SDK stub only; the C integration test mocks the exercised CUDA calls with CPU memcpy';
$record->{status}=$bad?'FAILED':'PASS_CPU_ONLY';
flock($lock,LOCK_UN) or die $!;
close $lock or die $!;
$record->{lock}{released}=JSON::PP::true;
PairedDepthBench::save("$out/cpu-validation.json",$record);
print "RESULT=$record->{status}\n";
exit($bad?1:0);
