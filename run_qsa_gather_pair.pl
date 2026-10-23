#!/usr/bin/env perl
use strict; use warnings;
use FindBin; use lib $FindBin::Bin;
use QsaEvidence qw(sha put write_text run absolute new_output csv_rows);
use Getopt::Long 'GetOptions'; use JSON::PP;
$|=1;
sub main {
 my($binary,$out,$nv,$iterations,$seed,$matrix)=('./qsa_gather_bench',undef,5000,32,4242,0);
 GetOptions('binary=s'=>\$binary,'out=s'=>\$out,'nv=i'=>\$nv,'iterations=i'=>\$iterations,'seed=i'=>\$seed,'matrix'=>\$matrix) or die "invalid arguments\n";
 defined($out) && !@ARGV or die "--out is required; no positional arguments accepted\n";
 1<=$nv && $nv<=65536 && 1<=$iterations && $iterations<=10000 && 0<=$seed && $seed<=4294967295 or die "invalid nv/iterations/seed\n";
 $out=new_output($out);$binary=absolute($binary);my $here=$FindBin::Bin;
 my $cubin=absolute($ENV{DIE_CUBIN}//'die.cubin');
 my %identities;for my $f($binary,$cubin,"$here/qsa_gather_bench.c","$here/qsa_test.c","$here/../ref/qsa.c","$here/../ref/attn.c","$here/../ref/fmath.c"){$identities{$f}=sha($f);}
 my($so,$se)=run(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total,memory.free','--format=csv'],\%ENV,30);
 write_text("$out/nvidia-smi.txt",$so.$se);
 my @cases=([$nv,4*$nv+1<=262144?1:0,'changing']);
 if($matrix){@cases=();my %seen;for my $v(sort {$a<=>$b} grep {!$seen{$_}++}(128,$nv)){for my $t(0..3){for my $pat(qw(clustered even random changing captured)){push @cases,[$v,$t,$pat] if 4*$v+$t<=262144;}}}}
 my $report={identities_sha256=>\%identities,CUDA_VISIBLE_DEVICES=>$ENV{CUDA_VISIBLE_DEVICES},seed=>$seed,iterations=>$iterations,hardware_counter_gate=>'not_run',true_dram_bandwidth_validated=>JSON::PP::false,pairs=>[]};
 put("$out/provenance.json",$report);
 for my $case(@cases) {
  my($v,$tail,$pat)=@$case;my $k=$v<512?$v:512;my $tag="nv$v-tail$tail-$pat";
  my %env=%ENV;$env{DIE_CUBIN}=$cubin;delete $env{GATHER_IDS};
  if($pat eq 'captured') {
   my $capture="$out/$tag.ids.u32";open my $f,'>:raw',$capture or die "write $capture: $!\n";
   # Portable fixture: select k distinct IDs, then sort each captured row.
   my $state=$seed;
   for(1..$iterations) {
    my @pool=(0..$v-1);my @row;
    for my $i(0..$k-1){$state=(1664525*$state+1013904223)&0xffffffff;my $j=$i+$state%($v-$i);@pool[$i,$j]=@pool[$j,$i];push @row,$pool[$i];}
    @row=sort {$a<=>$b}@row;print {$f} pack('V*',@row) or die "write $capture: $!\n";
   }
   close $f or die "close $capture: $!\n";$env{GATHER_IDS}=$capture;
  }
  my(%arms,%rows);
  for my $host(0,1) {
   my $raw="$out/$tag-host$host.csv";
   my @cmd=($binary,"$v","$tail","$iterations",$pat,"$host",$raw,"$seed");
   my($stdout,$stderr,$rc)=run(\@cmd,\%env,180);
   write_text("$out/$tag-host$host.log",$stdout.$stderr);
   $rc==0 or die "benchmark failed exit=$rc: @cmd; see log\n";
   my $result=JSON::PP->new->decode($stdout);
   ref($result) eq 'HASH' && $result->{preflight_verified_all} or die "missing preflight verification\n";
   $rows{$host}=csv_rows($raw);
   @{$rows{$host}}==$iterations or die "missing raw timing rows\n";
   $arms{$host}={command=>\@cmd,result=>$result,raw_csv_sha256=>sha($raw)};
  }
  for my $field(qw(cache_fnv1a64 selections_fnv1a64 logical_keys physical_gathered_slots selection_changes verified_output_signature_changes)) {
   exists($arms{0}{result}{$field}) && exists($arms{1}{result}{$field}) or die "missing result field: $field\n";
   defined($arms{0}{result}{$field}) && defined($arms{1}{result}{$field}) && "$arms{0}{result}{$field}" eq "$arms{1}{result}{$field}" or die "host/VRAM input or output identity differs: $field\n";
  }
  for my $i(0..$iterations-1) {
   for my $field(qw(iteration selection_fnv1a64 verified_output_fnv1a64)) {
    exists($rows{0}[$i]{$field}) && exists($rows{1}[$i]{$field}) or die "missing raw CSV field: $field\n";
    $rows{0}[$i]{$field} eq $rows{1}[$i]{$field} or die "raw pair mismatch: $field\n";
   }
  }
  exists($arms{0}{result}{gpu_us_p50}) && exists($arms{1}{result}{gpu_us_p50}) or die "missing gpu_us_p50\n";
  my $den=$arms{0}{result}{gpu_us_p50};my $ratio=$den?$arms{1}{result}{gpu_us_p50}/$den:undef;
  my $pair={case=>$tag,arms=>\%arms,host_to_vram_p50_ratio=>$ratio};
  if($pat eq 'captured'){$pair->{captured_ids_sha256}=sha("$out/$tag.ids.u32");$pair->{captured_ids_origin}='generated fixture, not model capture';$pair->{captured_ids_generator}='lcg32-1664525-1013904223-partial-fisher-yates-v1';}
  push @{$report->{pairs}},$pair;put("$out/report.json",$report);
  print "$tag verified ",(defined($ratio)?$ratio:'null'),"\n";
 }
 return 0;
}
my $rc=eval{main()};if($@){print STDERR $@;exit 1;}exit $rc;
