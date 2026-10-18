#!/usr/bin/env perl
# CPU-only malformed-evidence tests. Every value below is synthetic.
use strict;
use warnings;
use FindBin;
use lib $FindBin::Bin;
use QsaEvidence qw(put write_text run);
use File::Temp qw(tempdir);
use JSON::PP;
my $root=tempdir('cta-remap-summary-test-XXXXXX',TMPDIR=>1,CLEANUP=>1);
my @columns=qw(die token phase cta opcode op wait_start_ns wait_end_ns body_start_ns
                body_end_ns phase_end_ns nonempty_units nonce);
sub fixture {
    my ($label,$mutation)=@_;
    my $dir="$root/$label";mkdir($dir) or die $!;
    write_text("$dir/status.txt","PASS\n");
    write_text("$dir/config.txt","pairs=2\ndecode_intervals=2\ndies=6\nlayers=48\nnblk=512\nprompt=2048\n".
        join('',map {"die".$_."_phases=4\n"} 0..5));
    for my $stage(qw(untraced traced)) {
     for my $arm(qw(old new)) {
      for my $pair(-2..1) {
        my $us=1000+($arm eq 'new'?20:0)+($stage eq 'traced'?40:0)+10*$pair;
        my $r={stage=>$stage,arm=>$arm,pair=>$pair,warmup=>$pair<0?JSON::PP::true:JSON::PP::false,
            order=>($pair&1)?($arm eq 'new'?1:0):($arm eq 'old'?1:0),
            stage_order=>($pair&1)?($stage eq 'traced'?1:0):($stage eq 'untraced'?1:0),
            token_parity=>JSON::PP::true,resumed=>2047,decode_intervals=>2,
            token_ids=>[11,22,33],arrival_since_first_us=>[0,$us,2*$us],
            host_decode_us=>$us,dies=>[]};
        if($stage eq 'traced') {
            my $nonce=($pair+2)*4+$r->{stage_order}*2+$r->{order}+1;
            my @lines=(join(',',@columns));
            my @op=(67,66,70,0);my @name=qw(attnpartn kvgather idxscore embrow);
            for my $d(0..5) {
                # Different die clocks deliberately have unrelated origins.
                my $base=(6-$d)*1_000_000_000_000+$nonce*10_000_000;
                push @{$r->{dies}},{die=>$d,device_decode_us=>800,
                    attention_body_us=>50,gather_body_us=>50,indexer_body_us=>50,other_body_us=>50,
                    exchange_wait_us=>100,dispatch_us=>100,grid_wait_us=>200,transition_us=>200};
                for my $t(0..1){for my $p(0..3){for my $c(0..15){
                    my $start=$base+$t*800_000+$p*200_000+$c;
                    # At compact counts 2049/2050 the old mapping uses six CTAs
                    # with two units; the remap uses twelve CTAs with one unit.
                    my $work=$p ? 0 : $arm eq 'old' ? ($c%8<3?2:0) : ($c<12?1:0);
                    push @lines,join(',',$d,2048+$t,$p,$c,$op[$p],$name[$p],
                        $start,$start+25_000,$start+50_000,$start+100_000,$start+150_000,$work,$nonce);
                }}}
            }
            write_text("$dir/$stage-$arm-pair$pair.trace.csv",join("\n",@lines)."\n");
        }
        $mutation->($r) if $mutation && $stage eq 'traced' && $arm eq 'new' && $pair==1;
        put("$dir/$stage-$arm-pair$pair.json",$r);
      }
     }
    }
    return $dir;
}
sub text_edit {
    my($path,$edit)=@_;open my $f,'<',$path or die $!;local $/;my $text=<$f>;close $f;
    $edit->(\$text);write_text($path,$text);
}
sub trace_edit {
    my($dir,$edit)=@_;
    text_edit("$dir/traced-new-pair1.trace.csv",sub{
        my($text)=@_;my @lines=split /\n/,$$text;
        $edit->(\@lines);$$text=join("\n",@lines)."\n";
    });
}
sub field_edit {
    my($dir,$row,$column,$edit)=@_;
    trace_edit($dir,sub{
        my($lines)=@_;my @v=split /,/,$lines->[$row];
        $v[$column]=$edit->($v[$column]);$lines->[$row]=join(',',@v);
    });
}
my @cases=(
    ['complete',undef,undef,0],
    ['missing_sample',undef,sub{unlink "$_[0]/untraced-old-pair1.json" or die $!;},1],
    ['missing_warmup',undef,sub{unlink "$_[0]/untraced-old-pair-1.json" or die $!;},1],
    ['missing_trace',undef,sub{unlink "$_[0]/traced-new-pair1.trace.csv" or die $!;},1],
    ['changed_tokens',sub{$_[0]{token_ids}[2]=99;},undef,1],
    ['fractional_token',sub{$_[0]{token_ids}[2]=33.5;},undef,1],
    ['first_token_bias',sub{$_[0]{host_decode_us}+=10;},undef,1],
    ['nonfinite_callback',sub{$_[0]{arrival_since_first_us}[1]='Infinity';},undef,1],
    ['bad_order',sub{$_[0]{order}=0;},undef,1],
    ['bad_reconciliation',sub{$_[0]{dies}[4]{device_decode_us}+=1;},undef,1],
    ['coherent_json_tampering',sub{$_[0]{dies}[4]{attention_body_us}+=1;$_[0]{dies}[4]{transition_us}-=1;},undef,1],
    ['missing_die',sub{pop @{$_[0]{dies}};},undef,1],
    ['failed_status',undef,sub{write_text("$_[0]/status.txt","FAIL\n");},1],
    ['arbitrary_nonempty_trace',undef,sub{write_text("$_[0]/traced-new-pair1.trace.csv","synthetic_fixture\n");},1],
    ['missing_raw_record',undef,sub{trace_edit($_[0],sub{pop @{$_[0]};});},1],
    ['duplicate_raw_cta',undef,sub{trace_edit($_[0],sub{$_[0][2]=$_[0][1];});},1],
    ['wrong_nonce',undef,sub{field_edit($_[0],1,12,sub{$_[0]+1});},1],
    ['wrong_cta15_work',undef,sub{field_edit($_[0],16,11,sub{$_[0]+1});},1],
    ['wrong_opcode_name',undef,sub{field_edit($_[0],1,5,sub{'attnmerge'});},1],
    ['backwards_raw_time',undef,sub{field_edit($_[0],1,8,sub{$_[0]-100_000});},1],
    ['nonfinite_raw_time',undef,sub{field_edit($_[0],1,6,sub{'NaN'});},1],
    ['stale_raw_launch',undef,sub{trace_edit($_[0],sub{
        my($lines)=@_;
        for my $i(1..$#$lines){my @v=split /,/,$lines->[$i];$v[$_]-=50_000_000 for 6..10;$lines->[$i]=join(',',@v);}
    });},1],
    ['wrong_phase_count',undef,sub{text_edit("$_[0]/config.txt",sub{my($ref)=@_;$$ref=~s/die0_phases=4/die0_phases=5/;});},1],
    ['duplicate_configuration',undef,sub{text_edit("$_[0]/config.txt",sub{my($ref)=@_;$$ref.="prompt=2048\n";});},1]
);
for my $case(@cases) {
    my($name,$mutation,$change,$want_fail)=@$case;
    my $dir=fixture($name,$mutation);$change->($dir) if $change;
    my($stdout,$stderr,$rc)=run([$^X,"$FindBin::Bin/cta_remap_summary.pl",$dir],\%ENV,30);
    (($rc!=0)==$want_fail) or die "$name unexpected rc=$rc\n$stdout\n$stderr";
    if(!$want_fail) {
        my $s=decode_json($stdout);
        $s->{raw_samples}==8 && $s->{warmup_samples}==8 &&
        $s->{raw_trace_records}==6144 && $s->{raw_traces_verified} &&
        $s->{untraced}{paired_host_delta_us}{median}==20 &&
        $s->{instrumentation_overhead}{old}{paired_host_delta_us}{median}==40 &&
        $s->{traced}{new}{dies}{0}{attention_body_us}{median}==50 &&
        $s->{traced}{new}{dies}{0}{device_decode_us}{median}==800
            or die "synthetic distribution or raw-trace arithmetic incorrect\n";
    }
    print "$name: PASS (", $want_fail ? "rejected malformed synthetic evidence" : "accepted complete synthetic evidence",")\n";
}
print scalar(@cases)," CPU-only summary checks passed; no GPU results generated\n";
