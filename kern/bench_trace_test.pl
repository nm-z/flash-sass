#!/usr/bin/env perl
use strict;
use warnings;
use FindBin;
use lib $FindBin::Bin;
use Test::More;
use BenchTrace;
sub fixture {
    my($ctas,$epoch,$first,$budget)=@_;$first//=9;$budget//=512;
    my @rows;
    for my $t(0..2) {
        for my $p(0..1) {
            for my $c(0..$ctas-1) {
                my $s=$epoch+$t*1000+$p*100+$c;my $visible=$first+$t+1;
                push @rows,{position=>$first+$t,phase=>$p,layer=>3,
                    opcode=>$p?'attnpartn':'other',units=>1,flags=>3,
                    selected_keys=>int($visible/4)<=$budget?$visible:4*$budget+$visible%4,
                    start_ns=>$s,wait_start_ns=>$s+10,wait_end_ns=>$s+20,
                    body_start_ns=>$s+30,body_end_ns=>$s+70,end_ns=>$s+90,
                    ple_wait_start_ns=>$s+40,ple_wait_end_ns=>$s+50,publication_ns=>$p?$s+100:0,
                    cta=>$c,send_before_ns=>$s+75,send_after_ns=>$s+80,exchange_generation=>$t*2+$p};
            }
        }
    }
    return (\@rows,{schema=>2,nph=>2,count=>3,first=>$first,ctas=>$ctas,nblk=>$budget,
        die=>0,request=>1,status=>'completed',aborted=>0,ended=>0,completed=>$first+3,requested=>$first+3});
}
for my $ctas(1,16) {
    subtest "$ctas CTA exact sampled local timing" => sub {
        my($r,$m)=fixture($ctas,9007199254740993);my $s=BenchTrace::phase_summary($r,$m);
        is_deeply($s->{intervals_ns},[1000,1000],'exact periods above 2^53');
        is(scalar(@{$s->{local_cta_reconciliation}}),$ctas,'independent lanes retained');
        is($s->{op_counts}{attnpartn},2,'decode counts exclude first output and CTA duplicates');
        is($s->{op_counts_all_outputs}{attnpartn},3,'all-output execution count retained');
        is(scalar(@{$s->{selected_keys}}),3,'one attention count per layer/position including first');
        is($s->{op_mean_us}{attnpartn}{opcode_dispatch},0.01,'dispatch separate from body');
        is($s->{op_mean_us}{attnpartn}{body_excluding_ple_wait},0.03,'body excludes nested wait');
        ok(!exists($s->{op_mean_us}{attnpartn}{dispatch}),'no overlapping aggregate category');
        for my $lane(@{$s->{local_cta_reconciliation}}) {
            for my $token(@$lane) {
                is($token->{residual_ns},0,'exact local partition');
                is($token->{period_residual_ns},0,'exact period partition') if exists $token->{period_residual_ns};
            }
        }
    };
}
sub rejected {
    my($label,$change)=@_;my($r,$m)=fixture(1,10000);$change->($r,$m);
    my $ok=eval{BenchTrace::phase_summary($r,$m);1};ok(!$ok,$label);
}
rejected('missing row',sub{pop @{$_[0]}});
rejected('duplicate phase',sub{$_[0][1]{phase}=0});
rejected('body dispatch reversed',sub{$_[0][0]{body_start_ns}=$_[0][0]{wait_end_ns}-1});
rejected('PLE outside body',sub{$_[0][0]{ple_wait_start_ns}=$_[0][0]{body_start_ns}-1});
rejected('nonterminal publication',sub{$_[0][0]{publication_ns}=$_[0][0]{end_ns}+1});
rejected('missing publication',sub{$_[0][1]{publication_ns}=0});
rejected('overlapping positions',sub{$_[0][1]{publication_ns}=$_[0][2]{start_ns}+1});
rejected('wrong CTA order',sub{$_[0][0]{cta}=1});
rejected('phase identity changes',sub{$_[0][2]{opcode}='changed'});
rejected('malformed integer',sub{$_[0][0]{start_ns}='1e12'});
rejected('unsupported schema',sub{$_[1]{schema}=1});
rejected('incomplete CTA dimensions',sub{$_[1]{ctas}=16});
rejected('incorrect runtime selected-key field',sub{$_[0][1]{selected_keys}=999});
subtest 'dense and capped QSA cannot share a label' => sub {
    my($r,$m)=fixture(1,10000,7999,512);
    my $s=BenchTrace::phase_summary($r,$m);
    is($s->{selected_keys}[0]{selected_keys},2048,'default QSA caps at 512 blocks');
    ok($s->{selected_keys}[0]{capped},'capped at long context');
    $_->{opcode}='attnpart' for grep {$_->{phase}==1} @$r;
    my $dense=BenchTrace::phase_summary($r,$m);
    is($dense->{selected_keys}[0]{selected_keys},8000,'dense actual keys use visible length');
    is($dense->{selected_keys}[0]{runtime_selected_keys},2048,'QSA budget metadata retained separately');
    is($dense->{selected_keys}[0]{mode},'dense','dense mode identified from executed opcode');
    ($r,$m)=fixture(1,10000,7999,4096);$s=BenchTrace::phase_summary($r,$m);
    is($s->{selected_keys}[0]{selected_keys},8000,'larger budget remains uncapped');
    ok(!$s->{selected_keys}[0]{capped},'uncapped is explicit');
};
done_testing;
