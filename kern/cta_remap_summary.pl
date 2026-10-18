#!/usr/bin/env perl
# Summarize complete raw cta_remap_bench.c runs; recompute device times from CSV.
use strict;
use warnings;
use JSON::PP;
use FindBin;
use lib $FindBin::Bin;
use QsaEvidence qw(put csv_rows);
my $dir=shift @ARGV;
defined($dir) && !@ARGV or die "usage: $0 WHOLE_MODEL_OUTPUT_DIRECTORY\n";
sub uint {
    my($v,$name,$max)=@_;$max//=4294967295;
    defined($v) && !ref($v) && $v=~/\A(?:0|[1-9][0-9]*)\z/ && $v<=$max
        or die "invalid unsigned $name\n";
    return 0+$v;
}
sub number {
    my($v,$name)=@_;
    defined($v) && !ref($v) && $v=~/\A-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\z/
        && $v>=0 && $v<1e300 or die "invalid nonnegative finite $name\n";
    return 0+$v;
}
open my $status,'<',"$dir/status.txt" or die "missing status: $!\n";
my $state=<$status>;close $status;
defined($state) && $state eq "PASS\n" or die "run is not PASS\n";
open my $cfg,'<',"$dir/config.txt" or die "missing configuration: $!\n";
my %config;
while(<$cfg>){
    chomp;my($key,$value)=split /=/,$_,2;
    defined($value) && length($key) && !exists($config{$key}) or die "malformed/duplicate configuration\n";
    $config{$key}=$value;
}
close $cfg;
for(qw(pairs decode_intervals dies layers nblk prompt)){uint($config{$_},"configuration $_");}
$config{pairs}>=2 && $config{pairs}<=100 && $config{pairs}%2==0 &&
$config{dies}==6 && $config{layers}==48 && $config{nblk}==512 &&
$config{prompt}>=2 && $config{decode_intervals}>=2 && $config{decode_intervals}<=1024
    or die "insufficient whole-model coverage\n";
for my $d(0..5){uint($config{'die'.$d.'_phases'},"die phase count",65535)>0 or die "empty phase plan\n";}
# Use this checkout's append-only descriptor ABI, not labels supplied by a CSV.
open my $header,'-|',$^X,"$FindBin::Bin/die.pl",'header' or die "open opcode header: $!\n";
my %opname;
while(<$header>){$opname{$2}=lc($1) if /^#define OP_(\w+) (\d+)$/ && $1 ne 'COUNT';}
close $header or die "opcode header generation failed\n";
%opname or die "empty opcode header\n";
my @metrics=qw(attention_body_us gather_body_us indexer_body_us other_body_us
                exchange_wait_us dispatch_us grid_wait_us transition_us);
my @columns=qw(die token phase cta opcode op wait_start_ns wait_end_ns body_start_ns
                body_end_ns phase_end_ns nonempty_units nonce);
my(@runs,%by,%phase_plan,%ranges);
my $raw_records=0;
sub stats {
    my @v=sort {$a<=>$b} @_;
    @v or die "empty distribution\n";
    my $n=@v;
    return {n=>$n,min=>$v[0],max=>$v[-1],
        median=>$n%2?$v[int($n/2)]:($v[$n/2-1]+$v[$n/2])/2,
        p95=>$v[int(.95*$n)==.95*$n?int(.95*$n)-1:int(.95*$n)]};
}
sub nonempty {
    my($token,$cta,$arm)=@_;
    my $n=$token+1;$n=4*$config{nblk}+$n%4 if int($n/4)>$config{nblk};
    my $ck=int(($n+8191)/8192)*1024;my $count=0;
    for(my $unit=$cta;$unit<32;$unit+=16){
        my $chunk=$arm eq 'old' ? $unit&7 : $unit>>2;
        ++$count if $chunk*$ck<$n;
    }
    return $count;
}
sub raw_trace {
    my($path,$sample)=@_;
    my $rows=csv_rows($path);
    my $expected=0;$expected+=32*$config{'die'.$_.'_phases'} for 0..5;
    @$rows==$expected or die "raw trace record count $path\n";
    my $first=$config{prompt}+$config{decode_intervals}-2;
    my $nonce=($sample->{pair}+2)*4+$sample->{stage_order}*2+$sample->{order}+1;
    my($index,@derived)=(0);
    for my $d(0..5){
        my $phases=$config{'die'.$d.'_phases'};
        my(@previous,@cta0,$minimum,$maximum);
        for my $t(0..1){for my $p(0..$phases-1){for my $c(0..15){
            my $r=$rows->[$index++];
            keys(%$r)==@columns && !grep {!exists $r->{$_}} @columns
                or die "raw trace columns $path\n";
            uint($r->{die},'die')==$d && uint($r->{token},'token')==$first+$t &&
            uint($r->{phase},'phase')==$p && uint($r->{cta},'CTA')==$c &&
            uint($r->{nonce},'nonce')==$nonce or die "raw trace identity/nonce $path\n";
            my $op=uint($r->{opcode},'opcode');
            exists($opname{$op}) && $r->{op} eq $opname{$op} or die "raw trace opcode $path\n";
            my $plan="$d:$p";
            $phase_plan{$plan}//=$op;
            $phase_plan{$plan}==$op or die "raw trace phase plan changed $path\n";
            my $work=$r->{op} eq 'attnpartn' ? nonempty($first+$t,$c,$sample->{arm}) : 0;
            uint($r->{nonempty_units},'work')==$work or die "raw trace work count $path\n";
            my @ts=map {uint($r->{$_},"timestamp $_",18446744073709551615)}
                qw(wait_start_ns wait_end_ns body_start_ns body_end_ns phase_end_ns);
            $ts[0]>0 && $ts[4]>$ts[0] or die "raw trace empty timestamps $path\n";
            for my $i(1..4){$ts[$i]>=$ts[$i-1] or die "raw trace timestamp order $path\n";}
            (!defined($previous[$c]) || $ts[0]>=$previous[$c])
                or die "raw trace phase/token order $path\n";
            $previous[$c]=$ts[4];
            $minimum=$ts[0] if !defined($minimum) || $ts[0]<$minimum;
            $maximum=$ts[4] if !defined($maximum) || $ts[4]>$maximum;
            $cta0[$t][$p]={op=>$r->{op},ts=>\@ts} if !$c;
        }}}
        exists($ranges{$nonce}{$d}) and die "duplicate launch nonce $path\n";
        $ranges{$nonce}{$d}=[$minimum,$maximum];
        my $begin=$cta0[0][-1]{ts}[4];my $end=$cta0[1][-1]{ts}[4];
        $end>$begin or die "raw trace decode interval $path\n";
        my %value=(die=>$d,device_decode_us=>($end-$begin)*.001,map {$_=>0} @metrics);
        my $covered=0;
        for my $p(0..$phases-1){
            my $r=$cta0[1][$p];my @ts=@{$r->{ts}};
            my $category=$r->{op}=~/\A(?:attnpartn|attnmerge)\z/ ? 0 :
                         $r->{op} eq 'kvgather' ? 1 :
                         $r->{op}=~/\A(?:idxqk|idxscore|idxsel)\z/ ? 2 : 3;
            $value{$metrics[$category]}+=($ts[3]-$ts[2])*.001;
            $value{exchange_wait_us}+=($ts[1]-$ts[0])*.001;
            $value{dispatch_us}+=($ts[2]-$ts[1])*.001;
            $value{grid_wait_us}+=($ts[4]-$ts[3])*.001;
            $covered+=$ts[4]-$ts[0];
        }
        $covered<=$end-$begin or die "raw trace interval reconciliation $path\n";
        $value{transition_us}=($end-$begin-$covered)*.001;
        push @derived,\%value;
    }
    $raw_records+=@$rows;
    return \@derived;
}
my $canonical;
for my $stage(qw(untraced traced)) {
 for my $pair(-2..$config{pairs}-1) {
  for my $arm(qw(old new)) {
   my $path="$dir/$stage-$arm-pair$pair.json";
   open my $f,'<',$path or die "missing sample $path: $!\n";
   my $r;{local $/;$r=decode_json(<$f>);}close $f;
   ref($r) eq 'HASH' && ref($r->{token_ids}) eq 'ARRAY' &&
   ref($r->{arrival_since_first_us}) eq 'ARRAY' && ref($r->{dies}) eq 'ARRAY'
       or die "bad sample structure $path\n";
   $r->{stage} eq $stage && $r->{arm} eq $arm && $r->{pair}==$pair &&
   ($r->{warmup}?1:0)==($pair<0?1:0) && $r->{token_parity} &&
   $r->{resumed}==$config{prompt}-1 &&
   defined($r->{order}) && $r->{order}==(($pair&1)?($arm eq 'new'):($arm eq 'old')) &&
   defined($r->{stage_order}) && $r->{stage_order}==(($pair&1)?($stage eq 'traced'):($stage eq 'untraced')) &&
   $r->{decode_intervals}==$config{decode_intervals} &&
   @{$r->{token_ids}}==$config{decode_intervals}+1 &&
   @{$r->{arrival_since_first_us}}==$config{decode_intervals}+1 or die "bad coverage $path\n";
   uint($_,'generated token ID') for @{$r->{token_ids}};
   my $token_key=join ',',@{$r->{token_ids}};
   $canonical //= $token_key;
   $canonical eq $token_key or die "token mismatch $path\n";
   my $times=$r->{arrival_since_first_us};
   number($_,'callback timestamp') for @$times;
   $times->[0]==0 or die "bad first-token origin $path\n";
   for my $i(1..$#$times){$times->[$i]>=$times->[$i-1] or die "callback order $path\n";}
   my $mean=$times->[-1]/$config{decode_intervals};
   $mean>0 && abs($mean-number($r->{host_decode_us},'host decode time'))<.001
       or die "host decode timing mismatch $path\n";
   if($stage eq 'traced') {
       @{$r->{dies}}==6 or die "missing per-die trace $path\n";
       my $derived=raw_trace("$dir/$stage-$arm-pair$pair.trace.csv",$r);
       my %seen;
       for my $die(@{$r->{dies}}) {
           ref($die) eq 'HASH' or die "bad die structure $path\n";
           my $d=uint($die->{die},'die',5);!$seen{$d}++ or die "bad die identity $path\n";
           for my $metric('device_decode_us',@metrics){
               abs(number($die->{$metric},$metric)-$derived->[$d]{$metric})<.001
                   or die "raw trace metric mismatch $path die=$d metric=$metric\n";
           }
       }
       # Summary arithmetic uses the raw records after checking reported values.
       $r->{dies}=$derived;
   } else {@{$r->{dies}}==0 or die "untraced sample contains trace data\n";}
   if($pair>=0){push @runs,$r;$by{$stage}{$arm}{$pair}=$r;}
  }
 }
}
# Compare only timestamps from the same die, in actual launch order.
for my $d(0..5){
    my $last=0;
    for my $nonce(sort {$a<=>$b} keys %ranges){
        my($start,$end)=@{$ranges{$nonce}{$d}};
        $start>$last or die "raw trace stale/replayed launch die=$d nonce=$nonce\n";
        $last=$end;
    }
}
my %summary=(configuration=>\%config,host_units=>'microseconds per decode interval',
    host_definition=>'last minus first generated-token callback, divided by decode intervals; first token and prefill excluded',
    device_definition=>'CTA0 final phase end of last decoded position minus preceding position on each die independently',
    phase_definition=>'last decoded position; bodies include internal opcode waits and the local CTA drain barrier',
    raw_samples=>scalar(@runs),warmup_samples=>8,raw_trace_records=>$raw_records,
    raw_traces_verified=>JSON::PP::true);
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
