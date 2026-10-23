#!/usr/bin/env perl
use strict; use warnings; use JSON::PP; use Digest::SHA; use File::Spec; use POSIX qw(strftime);
my $root=shift // die "runtime root required\n";
sub readall {open my $f,'<:raw',$_[0] or die "$_[0]: $!"; local $/; <$f>}
sub artifact {my($path)=@_; open my $f,'<:raw',$path or die "$path: $!"; my $s=Digest::SHA->new(256); $s->addfile($f); return {path=>$path,bytes=>0+(-s $path),sha256=>$s->hexdigest};}
sub command {my(@cmd)=@_; open my $p,'-|',@cmd or die "@cmd: $!"; local $/; my $out=<$p>; close $p; return {argv=>\@cmd,exit=>0+($?>>8),stdout=>$out};}
my @files=qw(kern/engine.c kern/prog.c kern/flashcore.c kern/Makefile kern/die.cubin kern/die_trace.cubin kern/libengine.so kern/flashcore kern/nullengine.so runtime-controls.patch maxas-executed.patch build-final.log controls-build.log);
my $r={schema=>'issue10.executed-runtime.v1',recorded_at_utc=>strftime('%Y-%m-%dT%H:%M:%SZ',gmtime),
source=>{repository=>'https://github.com/nm-z/flash-sass',commit=>'bac5b88b665ee6973488dd40b363ded3134136f6',archive_sha256=>'a569a28c73f8cf51bffa2df726372c37f3b8c2d7083a15b56fb938e0d2e0fc59',patch=>'runtime-controls.patch',note=>'Exact historical measured runtime rebuilt with the published C benchmark controls; kernel SASS bodies unchanged.'},
assembler=>{repository=>'https://github.com/NervanaSystems/maxas',commit=>'54eda7af086a46c9dae1688b691968235d560164',patch=>'maxas-executed.patch'},
builds=>[
 {cwd=>"$root/kern",argv=>['make','-B','-j2','CUDA=/opt/cuda','PTXAS11=/opt/cuda-11.4/bin/ptxas','FLASH_REV=bac5b88b665ee6973488dd40b363ded3134136f6','die.cubin','die_trace.cubin','libengine.so','flashcore','nullengine.so'],exit=>0+readall("$root/build-final-exit.txt"),log=>'build-final.log'},
 {cwd=>"$root/kern",argv=>['make','CUDA=/opt/cuda','FLASH_REV=bac5b88b665ee6973488dd40b363ded3134136f6-pr38-controls','libengine.so'],exit=>0+readall("$root/controls-build-exit.txt"),log=>'controls-build.log'}],
tools=>[command('cc','--version'),command('/opt/cuda-11.4/bin/ptxas','--version'),command('perl','-V:version'),command('uname','-a')],
artifacts=>[map {artifact("$root/$_")} @files],
dependency_artifacts=>[map {artifact($_)} qw(/usr/lib/libcuda.so.470.256.02 /opt/cuda/include/cuda.h /opt/cuda-11.4/bin/ptxas)]
};
print JSON::PP->new->canonical->pretty->encode($r);
