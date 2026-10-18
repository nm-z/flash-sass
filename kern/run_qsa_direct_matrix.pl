#!/usr/bin/env perl
use strict; use warnings;
use FindBin; use lib $FindBin::Bin;
use QsaEvidence qw(sha put write_text run absolute new_output utc);
use Getopt::Long 'GetOptions'; use JSON::PP; use Time::HiRes 'time';
$|=1;
sub main {
 my($binary,$out,$quick,$timeout)=('./qsa_direct_test',undef,0,180);
 GetOptions('binary=s'=>\$binary,'out=s'=>\$out,'quick'=>\$quick,'timeout=i'=>\$timeout)
  or die "usage: $0 --out NEW_DIRECTORY [--binary PATH] [--quick] [--timeout SECONDS]\n";
 defined($out) && !@ARGV or die "--out is required; no positional arguments accepted\n";
 $timeout>0 or die "--timeout must be positive\n";
 $out=new_output($out);$binary=absolute($binary);
 my $cubin=absolute($ENV{DIE_CUBIN}//'die.cubin');
 my %env=%ENV;delete $env{$_} for grep {/^QSA_/} keys %env;$env{DIE_CUBIN}=$cubin;
 my @cases;my @lengths=$quick?(1,2047,2052,3100):(1,2047,2051,2055,3100,8000,65000,262000,262144);
 for my $host(0,1) {
  for my $visible(@lengths){my $steps=262145-$visible;$steps=4 if $steps>4;push @cases,[$visible,512,$steps,4242,$host,'hash',0,undef];}
  push @cases,[65,8,4,4242,$host,'hash',1,undef];
  for my $pattern(qw(oldest newest clustered scattered equal adjacent)){push @cases,[3100,512,4,4242,$host,$pattern,0,undef];}
 }
 if(!$quick){for my $seed(0,77){for my $visible(2052,8000,65000){push @cases,[$visible,512,4,$seed,0,'hash',0,undef];}}}
 for my $injection(qw(selection gather output)){push @cases,[2052,512,1,4242,0,'hash',0,$injection];}
 my $info={schema=>'flash-sass.qsa-direct-matrix.v1',started_utc=>utc(),binary=>$binary,binary_sha256=>sha($binary),cubin=>$cubin,cubin_sha256=>sha($cubin),CUDA_VISIBLE_DEVICES=>$env{CUDA_VISIBLE_DEVICES},scope=>'synthetic-history component parity, not whole-model or checkpoint quality',cases=>[],passed=>JSON::PP::false};
 my($so,$se)=run(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.mem,temperature.gpu,memory.used,utilization.gpu','--format=csv'],\%env,30);
 write_text("$out/gpu-before.csv",$so.$se);put("$out/results.json",$info);
 for my $i(0..$#cases) {
  my($visible,$budget,$steps,$seed,$host,$pattern,$replay,$injection)=@{$cases[$i]};
  my $tag=sprintf('%03d-n%d-b%d-s%d-h%d-%s',$i,$visible,$budget,$seed,$host,$pattern);
  $tag.='-replay' if $replay;$tag.='-inject-'.$injection if defined $injection;
  my %runenv=(%env,QSA_DIRECT_SEED=>"$seed",QSA_HOSTKV=>"$host",QSA_PATTERN=>$pattern,QSA_REPLAY_PREFIX=>"$replay");
  $runenv{QSA_DIRECT_INJECT}=$injection if defined $injection;
  my @cmd=($binary,"$visible","$budget","$steps");my $started=time;
  my($stdout,$stderr,$rc)=run(\@cmd,\%runenv,$timeout);
  my $log="$out/$tag.log";write_text($log,$stdout.$stderr);
  my @rows;while($stdout=~/^visible=(\d+) complete=(\d+) tail=(\d+) selected=(\d+) keys=(\d+) guarded_allocations=(\d+) exclusion=(yes|no) (EXACT|FAIL)$/mg){push @rows,[$1,$2,$3,$4,$5,$6,$7,$8];}
  my $expected=defined($injection)?3:0;my $passed=$rc==$expected && @rows==$steps;
  for my $j(0..$#rows) {
   my($n,$complete,$tail,$selected,$keys,$guards,$excluded,$status)=@{$rows[$j]};
   my $wantselected=int($n/4)<$budget?int($n/4):$budget;
   $passed &&= $n==$visible+$j && $complete==int($n/4) && $tail==$n%4 && $selected==$wantselected && $keys==4*$selected+$tail && $guards>0 && (($excluded eq 'yes')==(int($n/4)>$budget)) && $status eq (defined($injection)?'FAIL':'EXACT');
  }
  my %qenv=map {$_=>$runenv{$_}} grep {/^QSA_/} keys %runenv;
  push @{$info->{cases}},{case=>$tag,command=>\@cmd,environment=>\%qenv,exit_code=>$rc,expected_exit_code=>$expected,seconds=>time-$started,parsed_positions=>scalar(@rows),log_sha256=>sha($log),passed=>$passed?JSON::PP::true:JSON::PP::false};
  put("$out/results.json",$info);print "$tag ",($passed?'PASS':'FAIL'),"\n";return 1 unless $passed;
 }
 $info->{passed}=JSON::PP::true;$info->{finished_utc}=utc();put("$out/results.json",$info);return 0;
}
my $rc=eval{main()};if($@){print STDERR $@;exit 1;}exit $rc;
