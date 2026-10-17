#!/usr/bin/env perl
# Summarize only complete raw cta_remap_bench.c runs. No GPU or optional inputs.
use strict;
use warnings;
use JSON::PP;
use FindBin;
use lib $FindBin::Bin;
use QsaEvidence qw(put);
my $dir=shift @ARGV;
defined($dir) && !@ARGV or die "usage: $0 WHOLE_MODEL_OUTPUT_DIRECTORY\n";
open my $status,'<',"$dir/status.txt" or die "missing status: $!\n";
my $state=<$status>;close $status;
defined($state) && $state eq "PASS\n" or die "run is not PASS\n";
open my $cfg,'<',"$dir/config.txt" or die "missing configuration: $!\n";
my %config;
while(<$cfg>){chomp;my($key,$value)=split /=/,$_,2;$config{$key}=$value;}
close $cfg;
for(qw(pairs decode_intervals dies layers nblk prompt)){
    defined($config{$_}) && $config{$_}=~/\A\d+\z/ or die "invalid configuration $_\n";
}
$config{pairs}>=2 && $config{pairs}%2==0 && $config{dies}==6 &&
$config{layers}==48 && $config{nblk}==512 or die "insufficient whole-model coverage\n";
sub stats {
    my @v=sort {$a<=>$b} @_;
    @v or die "empty distribution\n";
    my $n=@v;
    return {n=>$n,min=>$v[0],max=>$v[-1],
        median=>$n%2?$v[int($n/2)]:($v[$n/2-1]+$v[$n/2])/2,
        p95=>$v[int(.95*$n)==.95*$n?int(.95*$n)-1:int(.95*$n)]};
}
my(@runs,%by);
my @metrics=qw(attention_body_us gather_body_us indexer_body_us other_body_us
                exchange_wait_us dispatch_us grid_wait_us transition_us);
my $canonical;
for my $stage(qw(untraced traced)) {
 for my $pair(0..$config{pairs}-1) {
  for my $arm(qw(old new)) {
   my $path="$dir/$stage-$arm-pair$pair.json";
   open my $f,'<',$path or die "missing sample $path: $!\n";
   local $/;my $r=decode_json(<$f>);close $f;
   $r->{stage} eq $stage && $r->{arm} eq $arm && $r->{pair}==$pair && !$r->{warmup} &&
   $r->{token_parity} && $r->{resumed}==$config{prompt}-1 &&
   defined($r->{order}) && $r->{order}==(($pair&1)?($arm eq 'new'):($arm eq 'old')) &&
   defined($r->{stage_order}) && $r->{stage_order}==(($pair&1)?($stage eq 'traced'):($stage eq 'untraced')) &&
   $r->{decode_intervals}==$config{decode_intervals} &&
   @{$r->{token_ids}}==$config{decode_intervals}+1 &&
   @{$r->{arrival_since_first_us}}==$config{decode_intervals}+1 or die "bad coverage $path\n";
   my $token_key=join ',',@{$r->{token_ids}};
   $canonical //= $token_key;
   $canonical eq $token_key or die "token mismatch $path\n";
   my $times=$r->{arrival_since_first_us};
   $times->[0]==0 or die "bad first-token origin $path\n";
   for my $i(1..$#$times){$times->[$i]>=$times->[$i-1] or die "callback order $path\n";}
   my $mean=$times->[-1]/$config{decode_intervals};
   $mean>0 && abs($mean-$r->{host_decode_us})<.001 or die "host decode timing mismatch $path\n";
   if($stage eq 'traced') {
       @{$r->{dies}}==6 && -s "$dir/$stage-$arm-pair$pair.trace.csv" or die "missing per-die trace $path\n";
       my %seen;
       for my $die(@{$r->{dies}}) {
           my $d=$die->{die};$d>=0 && $d<6 && !$seen{$d}++ or die "bad die identity $path\n";
           my $sum=0;
           for(@metrics){defined($die->{$_}) && $die->{$_}>=0 or die "negative/missing $_ in $path\n";$sum+=$die->{$_};}
           abs($sum-$die->{device_decode_us})<.001 or die "device reconciliation mismatch $path die=$d\n";
       }
   } else {@{$r->{dies}}==0 or die "untraced sample contains trace data\n";}
   push @runs,$r;$by{$stage}{$arm}{$pair}=$r;
  }
 }
}
my %summary=(configuration=>\%config,host_units=>'microseconds per decode interval',
    host_definition=>'last minus first generated-token callback, divided by decode intervals; first token and prefill excluded',
    device_definition=>'CTA0 final phase end of last decoded position minus preceding position on each die independently',
    phase_definition=>'last decoded position; bodies include internal opcode waits and the local CTA drain barrier',
    raw_samples=>scalar(@runs));
for my $stage(qw(untraced traced)) {
 for my $arm(qw(old new)) {
    $summary{$stage}{$arm}{host_decode_us}=stats(map {$_->{host_decode_us}} values %{$by{$stage}{$arm}});
 }
 $summary{$stage}{paired_host_delta_us}=stats(map {
    $by{$stage}{new}{$_}{host_decode_us}-$by{$stage}{old}{$_}{host_decode_us}
 } 0..$config{pairs}-1);
}
for my $arm(qw(old new)) {
 $summary{instrumentation_overhead}{$arm}{paired_host_delta_us}=stats(map {
    $by{traced}{$arm}{$_}{host_decode_us}-$by{untraced}{$arm}{$_}{host_decode_us}
 } 0..$config{pairs}-1);
 for my $d(0..5) {
    my @die=map {(grep {$_->{die}==$d}@{$_->{dies}})[0]} values %{$by{traced}{$arm}};
    for my $metric('device_decode_us',@metrics) {$summary{traced}{$arm}{dies}{$d}{$metric}=stats(map {$_->{$metric}}@die);}
 }
}
put("$dir/summary.json",\%summary);
print JSON::PP->new->canonical->pretty->encode(\%summary);
