package BenchTrace;
use strict;
use warnings;
use Config;
sub need { die "$_[1]\n" unless $_[0] }
sub uint {
    my($v,$name)=@_;
    need(defined($v)&&!ref($v)&&$v=~/\A[0-9]+\z/,"invalid $name");
    need(length($v)<20 || (length($v)==20 && $v le '18446744073709551615'),"overflow $name");
    return 0+$v;
}
sub phase_summary {
    my($rows,$m)=@_;
    need($Config{ivsize}>=8,'64-bit Perl required');
    need(ref($rows) eq 'ARRAY' && ref($m) eq 'HASH','invalid trace input');
    my %d; $d{$_}=uint($m->{$_},$_) for qw(schema nph count first ctas nblk);
    my($nph,$count,$first,$ctas)=@d{qw(nph count first ctas)};
    need($d{schema}==2 && $nph>0 && $nph<=4096 && $count>0
        && ($ctas==1||$ctas==16) && $count*$ctas<=256 && $d{nblk}>=1 && $d{nblk}<=4096,'bad schema2 dimensions');
    need(@$rows==$nph*$count*$ctas,'incomplete trace');
    my @numeric=qw(position phase units flags selected_keys start_ns wait_start_ns wait_end_ns
        body_start_ns body_end_ns end_ns ple_wait_start_ns ple_wait_end_ns publication_ns cta
        send_before_ns send_after_ns exchange_generation);
    my(@lanes,@previous_publication,@last_end);
    my(%ops,%counts,%all_counts,%selected,%phase_identity);
    for my $i(0..$#$rows) {
        need(ref($rows->[$i]) eq 'HASH','bad trace row');my %r=%{$rows->[$i]};
        $r{$_}=uint($r{$_},$_) for @numeric;
        need(defined($r{layer})&&!ref($r{layer})&&$r{layer}=~/\A-?[0-9]+\z/,'invalid layer');
        need(defined($r{opcode})&&!ref($r{opcode})&&$r{opcode}=~/\A[a-zA-Z0-9_]+\z/,'invalid opcode');
        my($t,$p,$c)=(int($i/($nph*$ctas)),int($i/$ctas)%$nph,$i%$ctas);
        need($r{position}==$first+$t && $r{phase}==$p && $r{cta}==$c,'missing/duplicate/out-of-order phase');
        my $id=join ',',@r{qw(layer opcode units flags)};
        if(exists $phase_identity{$p}){need($phase_identity{$p} eq $id,'phase identity changed');}
        else{$phase_identity{$p}=$id;}
        my @z=@r{qw(start_ns wait_start_ns wait_end_ns body_start_ns body_end_ns end_ns)};
        need($z[0]>0,'zero phase start');
        for(1..5){need($z[$_]>=$z[$_-1],'nonmonotonic phase');}
        need(!defined($last_end[$c])||$z[0]>=$last_end[$c],'overlapping local phases');
        my($ps,$pe,$pub,$sb,$se)=@r{qw(ple_wait_start_ns ple_wait_end_ns publication_ns send_before_ns send_after_ns)};
        need((!$ps&&!$pe)||($z[3]<=$ps&&$ps<=$pe&&$pe<=$z[4]),'bad PLE wait');
        need(($r{flags}&1)?($sb>0&&$z[4]<=$sb&&$sb<=$se&&$se<=$z[5]):(!$sb&&!$se),'bad send bracket');
        need($p==$nph-1?$pub>=$z[5]:!$pub,'bad publication');
        if(!$p) {
            $lanes[$c][$t]={position=>$r{position},cta=>$c,start_ns=>$z[0],phase_ns=>0,
                buckets_ns=>{},previous_publication_missing=>$t?0:1};
        }
        my $token=$lanes[$c][$t];my $b=$token->{buckets_ns};
        my %v=(dispatch_setup=>$z[1]-$z[0],opcode_dispatch=>$z[3]-$z[2],
            body_excluding_ple_wait=>$z[4]-$z[3]-($pe-$ps),ple_wait=>$pe-$ps,
            barrier_send=>$z[5]-$z[4]);
        $v{($r{flags}&2)?'exchange_wait':'dispatch_no_wait'}=$z[2]-$z[1];
        $b->{$_}+=$v{$_} for keys %v;
        $b->{phase_gap}+=$p?$z[0]-$last_end[$c]:0;
        $token->{phase_ns}+=$z[5]-$z[0];$token->{end_ns}=$z[5];$last_end[$c]=$z[5];
        if(!$c) {
            my $op=$r{opcode};$all_counts{$op}++;
            if($t){$counts{$op}++;$ops{$op}{$_}+=$v{$_} for keys %v;}
            # Actual partial-attention opcodes in prog.c/die.pl. Count one
            # layer execution; q/k projection, gather and merge are not extra
            # attention executions. Preserve the first output too.
            if($r{layer}>=0 && ($op eq 'attnpart' || $op eq 'attnpartn')) {
                my $key="$r{position},$r{layer}";my $mode=$op eq 'attnpartn'?'qsa':'dense';
                my $visible=$r{position}+1;
                my $expected=int($visible/4)<=$d{nblk}?$visible:4*$d{nblk}+$visible%4;
                need($r{selected_keys}==$expected,'runtime QSA-budget key field disagrees with nblk/position');
                my $effective=$mode eq 'dense'?$visible:$r{selected_keys};
                need(!exists($selected{$key}),'duplicate attention execution for layer/position');
                $selected{$key}={position=>$r{position},layer=>0+$r{layer},mode=>$mode,opcode=>$op,
                    selected_keys=>$effective,runtime_selected_keys=>$r{selected_keys},nblk=>$d{nblk},
                    capped=>$mode eq 'qsa' && $effective<$visible?1:0};
            }
        }
        if($p==$nph-1) {
            $token->{publication_ns}=$pub;$b->{publication_tail}=$pub-$z[5];
            my $sum=0;$sum+=$_ for values %$b;
            $token->{residual_ns}=$pub-$token->{start_ns}-$sum;
            need(!$token->{residual_ns},'nonzero local residual');
            if($t) {
                my $transition=$token->{start_ns}-$previous_publication[$c];
                need($transition>=0,'token starts before prior publication');$b->{token_transition}=$transition;
                $token->{interpublication_ns}=$pub-$previous_publication[$c];
                need($token->{interpublication_ns}>0,'bad publication interval');
                $token->{period_residual_ns}=$token->{interpublication_ns}-$sum-$transition;
                need(!$token->{period_residual_ns},'nonzero period residual');
            }
            $previous_publication[$c]=$pub;
        }
    }
    my @tokens=@{$lanes[0]};my @intervals=map {$tokens[$_]{interpublication_ns}}1..$#tokens;
    my %means;
    for my $op(keys %ops) {
        for my $k(qw(dispatch_setup dispatch_no_wait exchange_wait opcode_dispatch body_excluding_ple_wait ple_wait barrier_send)) {
            $means{$op}{$k}=($ops{$op}{$k}//0)/1000/($count>1?$count-1:1);
        }
    }
    my $sum=0;$sum+=$_ for @intervals;
    my @selected=sort {$a->{position}<=>$b->{position}||$a->{layer}<=>$b->{layer}} values %selected;
    return {metadata=>$m,tokens=>\@tokens,intervals_ns=>\@intervals,decode_ms=>$sum/1e6,
        op_counts=>\%counts,op_counts_all_outputs=>\%all_counts,op_mean_us=>\%means,selected_keys=>\@selected,
        local_cta_reconciliation=>\@lanes,opcode_cta=>0,mean_denominator_intervals=>$count-1,
        note=>'CTA0 opcode categories per decode interval, excluding first output; selected-key and all-output counts include it. Categories are disjoint. CTA and die durations overlap. Dense keys use position+1; runtime_selected_keys retains the QSA-budget field. No cross-die critical path inferred.'};
}
1;
