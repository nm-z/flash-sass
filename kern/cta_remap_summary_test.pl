#!/usr/bin/env perl
# CPU-only malformed-evidence tests. All values below are synthetic.
use strict;
use warnings;
use FindBin;
use lib $FindBin::Bin;
use QsaEvidence qw(put write_text run);
use File::Temp qw(tempdir);
use JSON::PP;
my $root=tempdir('cta-remap-summary-test-XXXXXX',TMPDIR=>1,CLEANUP=>1);
sub fixture {
    my ($label,$mutation)=@_;
    my $dir="$root/$label";mkdir($dir) or die $!;
    write_text("$dir/status.txt","PASS\n");
    write_text("$dir/config.txt","pairs=2\ndecode_intervals=2\ndies=6\nlayers=48\nnblk=512\nprompt=2048\n");
    for my $stage(qw(untraced traced)) {
     for my $arm(qw(old new)) {
      for my $pair(0..1) {
        my $us=1000+($arm eq 'new'?20:0)+($stage eq 'traced'?40:0)+10*$pair;
        my $r={stage=>$stage,arm=>$arm,pair=>$pair,warmup=>JSON::PP::false,
            order=>($pair&1)?($arm eq 'new'?1:0):($arm eq 'old'?1:0),
            stage_order=>($pair&1)?($stage eq 'traced'?1:0):($stage eq 'untraced'?1:0),
            token_parity=>JSON::PP::true,resumed=>2047,decode_intervals=>2,
            token_ids=>[11,22,33],arrival_since_first_us=>[0,$us,2*$us],
            host_decode_us=>$us,dies=>[]};
        if($stage eq 'traced') {
            for my $d(0..5) {
                push @{$r->{dies}},{die=>$d,device_decode_us=>800,
                    map {$_=>100} qw(attention_body_us gather_body_us indexer_body_us other_body_us exchange_wait_us dispatch_us grid_wait_us transition_us)};
            }
            write_text("$dir/$stage-$arm-pair$pair.trace.csv","synthetic_fixture\n");
        }
        $mutation->($r) if $mutation && $stage eq 'traced' && $arm eq 'new' && $pair==1;
        put("$dir/$stage-$arm-pair$pair.json",$r);
      }
     }
    }
    return $dir;
}
my @cases=(
    ['complete',undef,undef,0],
    ['missing_sample',undef,sub{unlink "$_[0]/untraced-old-pair1.json" or die $!;},1],
    ['missing_trace',undef,sub{unlink "$_[0]/traced-new-pair1.trace.csv" or die $!;},1],
    ['changed_tokens',sub{$_[0]{token_ids}[2]=99;},undef,1],
    ['first_token_bias',sub{$_[0]{host_decode_us}+=10;},undef,1],
    ['bad_order',sub{$_[0]{order}=0;},undef,1],
    ['bad_reconciliation',sub{$_[0]{dies}[4]{device_decode_us}+=1;},undef,1],
    ['missing_die',sub{pop @{$_[0]{dies}};},undef,1],
    ['failed_status',undef,sub{write_text("$_[0]/status.txt","FAIL\n");},1]
);
for my $case(@cases) {
    my($name,$mutation,$change,$want_fail)=@$case;
    my $dir=fixture($name,$mutation);$change->($dir) if $change;
    my($stdout,$stderr,$rc)=run([$^X,"$FindBin::Bin/cta_remap_summary.pl",$dir],\%ENV,10);
    (($rc!=0)==$want_fail) or die "$name unexpected rc=$rc\n$stdout\n$stderr";
    if(!$want_fail) {
        my $s=decode_json($stdout);
        $s->{raw_samples}==8 &&
        $s->{untraced}{paired_host_delta_us}{median}==20 &&
        $s->{instrumentation_overhead}{old}{paired_host_delta_us}{median}==40
            or die "synthetic distribution arithmetic incorrect\n";
    }
    print "$name: PASS (", $want_fail ? "rejected malformed synthetic evidence" : "accepted complete synthetic evidence",")\n";
}
print "9 CPU-only summary checks passed; no GPU results generated\n";
