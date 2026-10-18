#!/usr/bin/env perl
use strict; use warnings;
use FindBin; use lib $FindBin::Bin;
use QsaEvidence qw(sha put write_text run absolute);
use Getopt::Long 'GetOptions'; use File::Path 'make_path'; use File::Temp 'tempdir'; use JSON::PP;
$|=1;
sub main {
 my $out;GetOptions('out=s'=>\$out) or die "usage: $0 --out DIRECTORY\n";
 defined($out) && !@ARGV or die "--out is required; no positional arguments accepted\n";
 make_path($out);$out=absolute($out);
 my %binaries=map {$_=>absolute($_.'_test')}qw(qsa attn);
 my @qcases=([],['unknown'],['embimg','extra'],['kv','1','512','extra'],['idxsel','0','512'],['idxsel','700','512','0','6'],['idxscore','65537'],['idxfull','0'],['gathertime','2'],['gathertime','0','511'],['gathertime','0','65524'],['idxfull','65536','512','2']);
 push @qcases,['idxscore','-1'],['idxscore','4294967295'],['idxscore','65536','0'],['idxscore','65536','4097'],['idxscore','0','0'],['idxscore','65536','512','extra'];
 my @bad=('0','-1','+1','',' 1','1 ','1x','1.0','1e3','0x10','262145','4294967295','9999999999999999999999');
 push @qcases,map {['kv',$_,'512']}@bad;push @qcases,map {['kv','1',$_]}('0','-1','4097');
 my @acases=map {[$_]}@bad;push @acases,map {['1',$_]}('-1','+1','1x','262145');push @acases,['1','0','extra'];
 my @rows;my $d=tempdir(CLEANUP=>1);my $src="$d/guard.c";my $so="$d/guard.so";
 write_text($src,"#include <unistd.h>\nint cuInit(unsigned x){(void)x;_exit(99);}\nint cuLaunchKernel(){_exit(99);}\n");
 my($buildout,$builderr,$buildrc)=run(['cc','-shared','-fPIC',$src,'-o',$so],\%ENV,60);
 write_text("$out/guard-build.log",$buildout.$builderr);$buildrc==0 or die "guard compilation failed exit=$buildrc; see guard-build.log\n";
 my %env=(%ENV,CUDA_VISIBLE_DEVICES=>'',LD_PRELOAD=>$so,DIE_CUBIN=>'/__must_not_load__.cubin');
 for my $group(['qsa',\@qcases],['attn',\@acases]) {
  my($name,$cases)=@$group;for my $case(@$cases) {
   my @cmd=($binaries{$name},@$case);my($stdout,$stderr,$rc)=run(\@cmd,\%env,10);
   push @rows,{binary=>$name,args=>$case,exit=>$rc,stdout=>$stdout,stderr=>$stderr,passed=>$rc==2?JSON::PP::true:JSON::PP::false};
  }
 }
 # Valid boundaries must reach the cuInit trap. This also proves the guard is
 # active; exit 99 establishes argument acceptance, not successful GPU work.
 my @controls;
 for my $case(['qsa','idxscore','0','1'],['qsa','idxscore','0','4096'],['qsa','idxscore','65535','512'],['qsa','idxscore','65536','1'],['qsa','idxscore','65536','512'],['qsa','idxscore','65536','4096'],['attn','1']) {
  my($name,@args)=@$case;my($stdout,$stderr,$rc)=run([$binaries{$name},@args],\%env,10);
  push @controls,{binary=>$name,args=>\@args,exit=>$rc,stdout=>$stdout,stderr=>$stderr,passed=>$rc==99?JSON::PP::true:JSON::PP::false};
 }
 my %identities=map {$_=>{path=>$binaries{$_},sha256=>sha($binaries{$_})}} keys %binaries;
 my $passed=!grep {!$_->{passed}}(@rows,@controls);
 my $report={binaries=>\%identities,guard=>'CUDA entry points exit99',cases=>\@rows,accepted_controls=>\@controls,status=>$passed?'PASS':'FAIL'};
 put("$out/cli-report.json",$report);print $report->{status},' ',scalar(@rows),' invalid-argument cases; ',scalar(@controls)," accepted boundary controls reached CUDA trap\n";return $passed?0:1;
}
my $rc=eval{main()};if($@){print STDERR $@;exit 1;}exit $rc;
