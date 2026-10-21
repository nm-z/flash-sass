#!/usr/bin/env perl
# Recompute component distributions and physical CTA evidence from retained files.
use strict;
use warnings;
use JSON::PP;
use List::Util qw(sum min max);
use FindBin;
my $dir=shift @ARGV;
defined($dir) && !@ARGV or die "usage: $0 COMPONENT_DIRECTORY\n";
open my $header,'-|',$^X,"$FindBin::Bin/../../../kern/die.pl",'header' or die "opcode header: $!\n";
my %opcode;
while(<$header>){$opcode{$1}=0+$2 if /^#define OP_(ATTNPARTN|ATTNMERGE) (\d+)$/;}
close $header && keys(%opcode)==2 or die "opcode header failed\n";
sub read_json {
    my($path)=@_; open my $f,'<',$path or die "$path: $!\n";
    local $/; my $v=decode_json(<$f>); close $f; return $v;
}
sub distribution {
    my @v=sort {$a<=>$b} @_;
    @v or die "empty distribution\n";
    my $n=@v;
    return {n=>$n,min=>$v[0],max=>$v[-1],mean=>sum(@v)/$n,
        median=>$n%2?$v[int($n/2)]:($v[$n/2-1]+$v[$n/2])/2,
        p95=>$v[int(.95*$n)==.95*$n?int(.95*$n)-1:int(.95*$n)]};
}
sub positive {
    my($v)=@_;
    defined($v) && !ref($v) && $v =~ /\A(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\z/ &&
        $v>0 && $v<1e100 or die "invalid positive sample\n";
    return 0+$v;
}
my $m=read_json("$dir/manifest.json");
open my $sf,'<',"$dir/status.txt" or die "missing status\n";
<$sf> eq "PASS\n" or die "component matrix did not pass\n"; close $sf;
my @keys=split /,/,$m->{keys}; my @seeds=split /,/,$m->{seeds};
my @modes=split /,/,$m->{modes}; my $repeats=$m->{repeats};
my(%timing,%work,%body,%wait,%trace_count);
my($cases,$guard_launches,$verified_records,$raw_records)=(0,0,0,0);
for my $mode(@modes){for my $n(@keys){for my $seed(@seeds){
    my $name="$mode-keys$n-seed$seed";
    my $r=read_json("$dir/$name.json"); ++$cases;
    $r->{keys}==$n && $r->{seed}==$seed && $r->{bit_exact} &&
        $r->{boundary_canaries} && $r->{boundary_canary_words}==64 &&
        $r->{partial_values}==49536 && $r->{final_values}==6144 &&
        $r->{steps_per_sample}==64 && $r->{guard_checked_launches}==4+4*($repeats+4)
        or die "bad coverage $name\n";
    my $traced=$mode eq 'traced';
    ($r->{trace_verified}?1:0)==$traced && ($r->{timing_trace_enabled}?1:0)==$traced &&
        $r->{trace_launches}==($traced?$r->{guard_checked_launches}:0) &&
        $r->{trace_records}==($traced?256+192*($repeats+4):0)
        or die "bad trace coverage $name\n";
    $guard_launches+=$r->{guard_checked_launches}; $verified_records+=$r->{trace_records};
    for my $metric(qw(event partial_program merge)){
        for my $arm(qw(old new)){
            my $t=$r->{$arm}{$metric}; ref($t->{samples_us}) eq 'ARRAY' &&
                @{$t->{samples_us}}==$repeats or die "sample count $name $arm $metric\n";
            my @v=map {positive($_)} @{$t->{samples_us}};
            my $derived=distribution(@v);
            abs($derived->{median}-$t->{median_us})<.002 &&
                abs($derived->{p95}-$t->{p95_us})<.002 or die "reported statistic $name\n";
            push @{$timing{$mode}{$n}{$metric}{$arm}},@v;
            push @{$timing{$mode}{$n}{$metric}{"$arm-case-medians"}},$t->{median_us};
        }
        for my $i(0..$repeats-1){
            push @{$timing{$mode}{$n}{$metric}{paired_delta}},
                $r->{new}{$metric}{samples_us}[$i]-$r->{old}{$metric}{samples_us}[$i];
        }
    }
    next unless $traced;
    open my $raw,'<',"$dir/$name.stderr" or die "missing raw trace\n";
    my (%seen,%previous);
    while(<$raw>){
        /^TRACE (.*)\n$/ or die "unexpected trace stderr $name\n";
        my %v=map {split /=/,$_,2} split / /,$1;
        for my $field(qw(variant launch token phase cta opcode nonempty_units nonce wait_start wait_end body_start body_end end exchange_wait_ns dispatch_ns body_ns grid_wait_ns)){
            defined($v{$field}) && $v{$field}=~/\A\d+\z/ or die "bad raw field $name $field\n";
        }
        my $launch=$v{launch}; $launch>=1 && $launch<=4 &&
            $v{variant}==((0,1,1,0)[$launch-1]) && $v{nonce}==$launch &&
            $v{token}<2 && $v{phase}<2 && $v{cta}<16 &&
            $v{opcode}==$opcode{$v{phase}?'ATTNMERGE':'ATTNPARTN'} or die "raw trace identity $name\n";
        !$seen{"$launch:$v{token}:$v{phase}:$v{cta}"}++ or die "duplicate raw record\n";
        my $chunk_size=int(($n+8191)/8192)*1024; my $units=0;
        for(my $unit=$v{cta};$unit<32;$unit+=16){
            my $chunk=$v{variant}?($unit>>2):($unit&7);
            ++$units if !$v{phase} && $chunk*$chunk_size<$n;
        }
        $v{nonempty_units}==$units or die "raw work mismatch $name\n";
        my @t=@v{qw(wait_start wait_end body_start body_end end)};
        $t[0]>0 && $t[4]>$t[0] && (!exists($previous{$v{cta}}) || $t[0]>=$previous{$v{cta}})
            or die "empty or stale trace $name\n";
        $previous{$v{cta}}=$t[4];
        for my $i(1..4){$t[$i]>=$t[$i-1] or die "timestamp order $name\n";}
        $v{exchange_wait_ns}==$t[1]-$t[0] && $v{dispatch_ns}==$t[2]-$t[1] &&
            $v{body_ns}==$t[3]-$t[2] && $v{grid_wait_ns}==$t[4]-$t[3]
            or die "raw interval arithmetic $name\n";
        ++$raw_records;
        if(!$v{phase}){
            my $arm=$v{variant}?'new':'old'; my $c=$v{cta};
            defined($work{$n}{$arm}{$c}) && $work{$n}{$arm}{$c}!=$units and die "work varies\n";
            $work{$n}{$arm}{$c}=$units;
            push @{$body{$n}{$arm}{$c}},$v{body_ns}*.001;
            push @{$wait{$n}{$arm}{$c}},$v{grid_wait_ns}*.001;
        }
    }
    close $raw; keys(%seen)==256 or die "incomplete ABBA trace $name\n";
}}}
my %result=(harness_source=>$m->{harness_source},gpu=>$m->{gpu},whole_model=>JSON::PP::false,
    cases=>$cases,guard_checked_launches=>$guard_launches,verified_trace_records=>$verified_records,
    retained_raw_trace_records=>$raw_records,keys=>\@keys,seeds=>\@seeds,repeats=>$repeats,
    component_units=>'microseconds per fixed-input persistent-dispatch position',
    event_definition=>'CUDA event interval for 64 attention-plus-merge positions divided by 64',
    partial_program_definition=>'separate 64-position attention-only program; includes dispatcher overhead',
    merge_definition=>'last-position CTA0 phase-end difference; includes dispatch and synchronization',
    physical_trace_scope=>'initial ABBA correctness launches; body/wait distributions are not warmed production latency',
    across_modes=>'sequential mode batches; traced/untraced differences are descriptive, not balanced instrumentation-overhead estimates');
for my $mode(@modes){for my $n(@keys){for my $metric(qw(event partial_program merge)){
    my $v=$timing{$mode}{$n}{$metric};
    for my $field(keys %$v){$result{timing}{$mode}{$n}{$metric}{$field}=distribution(@{$v->{$field}});}
    $result{timing}{$mode}{$n}{$metric}{new_faster_pairs}=scalar(grep {$_<0} @{$v->{paired_delta}});
}}}
for my $n(keys %work){for my $arm(qw(old new)){
    $result{physical_cta}{$n}{$arm}=[map {{cta=>$_,nonempty_units=>$work{$n}{$arm}{$_},
        body_us=>distribution(@{$body{$n}{$arm}{$_}}),
        grid_wait_us=>distribution(@{$wait{$n}{$arm}{$_}})}} 0..15];
}}
my (%telemetry,@utc); my $samples=0; my $other_max=0;
for my $path(glob "$dir/telemetry-*.gpu.csv"){
    open my $f,'<',$path or die "$path: $!\n";
    my $selected=0;
    while(<$f>){
        chomp; my @v=split /,\s*/;
        @v==9 or die "telemetry columns $path\n";
        $v[6]=~/\A\d+\z/ or die "unknown utilization $path\n";
        if($v[0] ne $m->{gpu}){$other_max=max($other_max,$v[6]);next;}
        ++$selected;
        for my $item([3,'temperature_c'],[4,'sm_mhz'],[5,'memory_mhz'],[6,'gpu_utilization_percent'],[7,'memory_used_mib'],[8,'power_w']){
            my($index,$label)=@$item;
            $v[$index]=~/\A\d+(?:\.\d+)?\z/ or die "unknown telemetry $label $path\n";
            push @{$telemetry{$label}},0+$v[$index];
        }
        push @{$telemetry{active_sm_mhz}},0+$v[4] if $v[6]>0;
    }
    close $f; $selected==1 or die "target GPU missing $path\n"; ++$samples;
    (my $utc_path=$path)=~s/\.gpu\.csv$/.utc/;
    open my $u,'<',$utc_path or die "missing telemetry time\n";
    chomp(my $time=<$u>); close $u; push @utc,$time;
}
$samples>0 && $other_max==0 or die "missing telemetry or competing GPU activity\n";
$result{telemetry}={samples=>$samples,first_utc=>(sort @utc)[0],last_utc=>(sort @utc)[-1],
    maximum_other_gpu_utilization_percent=>$other_max,
    map {$_=>distribution(@{$telemetry{$_}})} keys %telemetry};
if(-e "$dir/uninstrumented-rejection.stderr"){
    open my $f,'<',"$dir/uninstrumented-rejection.stderr" or die "rejection stderr\n";
    my $text=do{local $/;<$f>}; close $f;
    $text =~ /^trace rejected variant=0 launch=1 record=0: launch nonce / &&
        $text !~ /undefined symbol|CUDA_ERROR|->/ or die "incorrect negative result\n";
    open my $exits,'<',"$dir/exits.tsv" or die "missing exits\n";
    my @negative=grep {/^uninstrumented-rejection\t/} <$exits>; close $exits;
    @negative==1 && $negative[0] eq "uninstrumented-rejection\t1\n" or die "incorrect negative exit\n";
    $result{uninstrumented_image_rejection}={passed=>JSON::PP::true,exit_code=>1,reason=>'missing launch nonce'};
}
print JSON::PP->new->canonical->pretty->encode(\%result);
