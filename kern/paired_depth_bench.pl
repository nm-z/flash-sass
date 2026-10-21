#!/usr/bin/env perl
package PairedDepthBench;
use strict;use warnings;
use B qw(svref_2object SVp_POK);use Cwd 'abs_path';use Digest::SHA;
use File::Path 'make_path';use Getopt::Long 'GetOptionsFromArray';
use IO::Compress::Gzip qw($GzipError);use IO::Uncompress::Gunzip qw($GunzipError);
use IO::Select;use IO::Socket::UNIX;use IPC::Open3;use JSON::PP;
use POSIX qw(ceil strftime);use Socket 'SOCK_STREAM';use Symbol 'gensym';
use Time::HiRes qw(clock_gettime CLOCK_MONOTONIC);
use FindBin;use lib $FindBin::Bin;
use Fcntl qw(LOCK_EX LOCK_NB LOCK_UN O_CREAT O_RDWR);
use BenchManifest;use BenchTrace;use BenchAcceptance;
# Connect only to supplied cores. Never reload a core or weight holder.
# Fixed synthetic tokens exercise the checkpoint, not answer quality.
# Historical absent lifecycle metadata stays unknown; live traces require it.
our $JSON=JSON::PP->new->canonical->allow_nonref;
our $PRETTY=JSON::PP->new->canonical->pretty->allow_nonref;
our @GPU_QUERY=qw(index uuid driver_version pci.bus_id memory.used memory.free utilization.gpu temperature.gpu clocks.current.sm clocks.current.memory ecc.mode.current pcie.link.gen.current pcie.link.width.current clocks_throttle_reasons.active clocks_throttle_reasons.sw_power_cap clocks_throttle_reasons.sw_thermal_slowdown power.draw power.limit);
sub read_text{my($p)=@_;open my $f,'<:raw',$p or die "open $p: $!\n";local $/;my $s=<$f>;close $f or die "close $p: $!\n";return defined($s)?$s:'';}
sub save{my($p,$v)=@_;my $tmp="$p.tmp.$$";open my $f,'>:raw',$tmp or die "open $tmp: $!\n";print {$f} $PRETTY->encode($v) or die "write $tmp: $!\n";close $f or die "close $tmp: $!\n";rename $tmp,$p or die "rename $p: $!\n";}
sub load_json{return $JSON->decode(read_text($_[0]));}
sub monotonic_ns{return int(clock_gettime(CLOCK_MONOTONIC)*1_000_000_000);}
sub sum{my $n=0;$n+=$_ for @{$_[0]};return $n;}
sub median{my($x)=@_;die "empty median\n" unless @$x;my @s=sort {$a<=>$b}@$x;my $m=int(@s/2);return @s%2?$s[$m]:($s[$m-1]+$s[$m])/2;}
sub p95{my($x)=@_;die "empty percentile\n" unless @$x;my @s=sort {$a<=>$b}@$x;return $s[ceil(.95*@s)-1];}
# Drain stdout and stderr concurrently.
sub run{
 my($argv)=@_;my $err=gensym;my $pid=open3(my $in,my $out,$err,@$argv);close $in;binmode $out;binmode $err;
 my $select=IO::Select->new($out,$err);my %which=(fileno($out)=>'stdout',fileno($err)=>'stderr');my %text=(stdout=>'',stderr=>'');
 while($select->count){for my $f($select->can_read){my $n=sysread($f,my $buf,65536);if(!defined $n){next if $!{EINTR};die "read subprocess: $!\n";}if(!$n){$select->remove($f);close $f;}else{$text{$which{fileno($f)}}.=$buf;}}}
 waitpid($pid,0)==$pid or die "waitpid: $!\n";my $s=$?;my $exit=($s&127)?-($s&127):$s>>8;return {argv=>[@$argv],exit=>$exit,%text};
}
sub sha{my($p)=@_;open my $f,'<:raw',$p or die "open $p: $!\n";my $h=Digest::SHA->new(256);$h->addfile($f);close $f or die "close $p: $!\n";return $h->hexdigest;}
# GNU stat retains fractional nanoseconds; floating stat epochs do not.
sub file_stamp{
 my($p)=@_;my $r=run(['stat','-L','-c',"%s\n%Y\n%y",'--',$p]);die "stat $p failed: ".$JSON->encode($r)."\n" if $r->{exit};
 my @v=split /\n/,$r->{stdout};die "invalid stat response for $p\n" unless @v>=3 && $v[0]=~/\A[0-9]+\z/ && $v[1]=~/\A-?[0-9]+\z/ && $v[2]=~/\.([0-9]{9}) [+-][0-9]{4}\z/;
 my $fraction=$1;return {bytes=>0+$v[0],mtime_ns=>$v[1]*1_000_000_000+$fraction};
}
sub identity{
 my($p)=@_;my $path=abs_path($p);die "resolve $p: $!\n" unless defined $path;
 my $a=file_stamp($path);my $hash=sha($path);my $b=file_stamp($path);
 die "file changed while hashing\n" if $a->{bytes}!=$b->{bytes} || $a->{mtime_ns}!=$b->{mtime_ns};return {path=>$path,%$b,sha256=>$hash};
}
sub gpu{my $r=run(['nvidia-smi','--query-gpu='.join(',',@GPU_QUERY),'--format=csv']);die "GPU metadata query failed: ".$JSON->encode($r)."\n" if $r->{exit} || $r->{stdout}!~/GPU-/;return $r;}
sub driver_snapshot{my $r=run(['nvidia-smi','-q','-x']);die "driver metadata query failed: ".$JSON->encode($r)."\n" if $r->{exit} || index($r->{stdout},'<nvidia_smi_log>')<0;return $r;}
sub is_string{return defined($_[0]) && !ref($_[0]) && (svref_2object(\$_[0])->FLAGS & SVp_POK);}
sub core_records{
 my($p)=@_;my $data=load_json($p);die "core records must name exactly host and vram\n" unless ref($data) eq 'HASH' && join(',',sort keys %$data) eq 'host,vram';
 for my $name(qw(host vram)){my $r=$data->{$name};die "invalid core record $name\n" unless ref($r) eq 'HASH';my $argv=$r->{argv};my $env=$r->{launch_environment};
  die "missing core argv $name\n" unless ref($argv) eq 'ARRAY' && @$argv && !(grep {!is_string($_)||!length($_)}@$argv);
  die "invalid launch environment $name\n" unless ref($env) eq 'HASH' && !(grep {!is_string($env->{$_})}keys %$env);
  die "engine reload commands must be an explicit list $name\n" unless ref($r->{engine_reload_commands}) eq 'ARRAY';
  for my $k(qw(FLASH_MAXT CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER)){die "missing launch setting $name.$k\n" unless exists $env->{$k};}
  die "explicit limitations list required $name\n" unless ref($r->{limitations}) eq 'ARRAY';
 }return $data;
}
sub fields{my($line)=@_;my @w=split /\s+/,$line;shift @w;my %r;for(@w){next unless index($_,'=')>=0;my($k,$v)=split /=/,$_,2;$r{$k}=$v;}return \%r;}
sub comparison_value {
 my($v)=@_;
 return {map {$_=>comparison_value($v->{$_})}keys %$v} if ref($v) eq 'HASH';
 return [map {comparison_value($_)}@$v] if ref($v) eq 'ARRAY';
 return $v if ref($v) || !defined($v);
 # Protocol scalars are text. Numeric inspection/JSON round trips can change
 # Perl's internal scalar flags without changing their reported value.
 return "$v";
}
sub matched{
 my($a,$b,$allow)=@_;my %allow=ref($allow) eq 'HASH'?%$allow:map {$_=>1}@$allow;
 my %keys=map {$_=>1}(keys %$a,keys %$b);my %diff;
 for my $k(keys %keys){next if $allow{$k};my($av,$bv)=($a->{$k},$b->{$k});my $same=!defined($av)&&!defined($bv);$same||=defined($av)&&defined($bv)&&$JSON->encode(comparison_value($av)) eq $JSON->encode(comparison_value($bv));$diff{$k}=[$av,$bv] unless $same;}
 die "unexplained manifest differences: ".$JSON->encode(\%diff)."\n" if keys %diff;
}
sub trace_lifecycle{
 my($m,%o)=@_;my @required=qw(status aborted ended completed requested);my @present=grep {exists $m->{$_}}@required;
 if(!@present){die "live trace lacks lifecycle metadata\n" if $o{require_current};return {reported=>JSON::PP::false,status=>'legacy_unreported'};}
 die "partial trace lifecycle metadata\n" if @present!=@required;
 for my $k(qw(aborted ended)){die "invalid trace lifecycle boolean\n" unless defined($m->{$k}) && !ref($m->{$k}) && $m->{$k}=~/\A[01]\z/;}
 for my $k(qw(completed requested)){die "invalid trace lifecycle count $k\n" unless defined($m->{$k}) && !ref($m->{$k}) && $m->{$k}=~/\A[0-9]+\z/;}
 my($c,$r,$ended)=map {0+$m->{$_}}qw(completed requested ended);
 die "unsuccessful trace request: $m->{status}\n" unless $m->{status} eq 'completed' && $m->{aborted} eq '0';
 die "impossible trace lifecycle counts\n" unless 1<=$c && $c<=$r;
 die "incomplete trace request\n" if !$ended && $c!=$r;
 die "stopped trace is not a fixed-continuation benchmark\n" if $ended && exists($o{allow_ended}) && !$o{allow_ended};
 die "trace belongs to different requested positions\n" if defined($o{expected_requested}) && $r!=$o{expected_requested};
 if(exists($m->{first})&&exists($m->{count})){my($first,$count)=map {0+$m->{$_}}qw(first count);die "trace window exceeds completed positions\n" if $first<0 || $count<1 || $first+$count>$c;}
 return {reported=>JSON::PP::true,status=>'completed',aborted=>0,ended=>$ended,completed=>$c,requested=>$r};
}
sub phase_summary{
 my($rows,$m)=@_;my $lifecycle=trace_lifecycle($m);my($nph,$count,$first)=map {0+$m->{$_}}qw(nph count first);
 if(($m->{schema}//0)==2){my $s=BenchTrace::phase_summary($rows,$m);$s->{lifecycle}=$lifecycle;return $s;}
 die "bad trace dimensions\n" if $nph<1 || $count<1 || $m->{ctas}!=1;
 my($seen,$previous)=(0,0);my(@tokens,%ops,%keys,%opcounts);
 for my $row(@$rows){
  my($pos,$phase)=map {0+$row->{$_}}qw(position phase);
  die "missing/duplicate/out-of-order phase\n" unless $pos==$first+int($seen/$nph) && $phase==$seen%$nph;
  my @z=map {0+$row->{$_}}qw(start_ns wait_start_ns wait_end_ns body_end_ns end_ns);
  die "nonmonotonic trace\n" if !$z[0] || $z[0]<$previous || (grep {$z[$_]<$z[$_-1]}1..4);
  my($ps,$pe,$pub)=map {0+$row->{$_}}qw(ple_wait_start_ns ple_wait_end_ns publication_ns);
  die "bad PLE wait interval\n" if ($ps||$pe) && !($z[2]<=$ps && $ps<=$pe && $pe<=$z[3]);
  push @tokens,{position=>$pos,start_ns=>$z[0],phase_ns=>0} if $phase==0;
  my $t=$tokens[-1];$t->{phase_ns}+=$z[4]-$z[0];$t->{end_ns}=$z[4];
  if($phase==$nph-1){die "missing publication\n" if $pub<$z[4];$t->{publication_ns}=$pub;}elsif($pub){die "publication on a nonterminal phase\n";}
  if($pos>$first){my $op=$row->{opcode};$ops{$op}//=[(0)x5];my @v=($z[1]-$z[0],$z[2]-$z[1],$z[3]-$z[2]-($pe-$ps),$z[4]-$z[3],$pe-$ps);$ops{$op}[$_]+=$v[$_] for 0..4;$opcounts{$op}++;}
  $keys{"$pos,".(0+$row->{selected_keys})}=[$pos,0+$row->{selected_keys}];$previous=$z[4];$seen++;
 }
 die "incomplete trace\n" unless $seen==$count*$nph;
 for my $t(@tokens){$t->{between_phase_ns}=$t->{end_ns}-$t->{start_ns}-$t->{phase_ns};$t->{publication_footer_ns}=$t->{publication_ns}-$t->{end_ns};}
 my @intervals=map {$tokens[$_]{publication_ns}-$tokens[$_-1]{publication_ns}}1..$#tokens;die "bad device interval\n" if grep {$_<=0}@intervals;
 my @labels=qw(dispatch exchange_wait body_excluding_ple_wait barrier_send ple_wait);my %means;
 for my $op(keys %ops){my %v;$v{$labels[$_]}=$ops{$op}[$_]/1000/($count>1?$count-1:1) for 0..4;$means{$op}=\%v;}
 my @selected=sort {$a->[0]<=>$b->[0] || $a->[1]<=>$b->[1]}values %keys;
 return {metadata=>$m,lifecycle=>$lifecycle,tokens=>\@tokens,intervals_ns=>\@intervals,decode_ms=>sum(\@intervals)/1e6,op_mean_us=>\%means,op_counts=>\%opcounts,selected_keys=>\@selected,note=>'Local clock per die; these die totals overlap and cannot be summed.'};
}
sub host_summary{
 my($rows,$m)=@_;trace_lifecycle($m);my($first,$count)=map {0+$m->{$_}}qw(first count);
 die "bad host trace dimensions\n" unless @$rows==$count && $m->{clock} eq 'CLOCK_MONOTONIC';my $previous=0;
 for my $i(0..$#$rows){my $r=$rows->[$i];die "bad host position\n" unless $r->{position}==$first+$i;
  my($ob,$cs,$ce)=map {0+$r->{$_}}qw(observed_ns callback_start_ns callback_end_ns);
  die "bad host callback ordering\n" unless $ob>=$previous && 0<$ob && $ob<=$cs && $cs<=$ce;
  my($ps,$pe,$ready)=map {0+$r->{$_}}qw(ple_staging_start_ns ple_staging_end_ns ple_ready_ns);
  die "bad host staging order\n" if ($ps||$pe||$ready) && !(0<$ps && $ps<=$pe && $pe<=$ready);
  die "PLE short read invalidates benchmark\n" if $r->{ple_short_reads};$previous=$ob;
 }return $rows;
}
sub csv_fields{
 my($line)=@_;$line=~s/\r?\n\z//;my @v;pos($line)=0;
 while(1){my $x;if($line=~/\G"((?:[^"]|"")*)"/gc){$x=$1;$x=~s/""/"/g;}elsif($line=~/\G([^",]*)/gc){$x=$1;}else{die "malformed trace CSV\n";}push @v,$x;last if pos($line)==length($line);die "malformed trace CSV delimiter\n" unless $line=~/\G,/gc;if(pos($line)==length($line)){push @v,'';last;}}return \@v;
}
sub read_csv_gzip{
 my($p)=@_;my $f=IO::Uncompress::Gunzip->new($p) or die "gunzip $p: $GunzipError\n";my $h=<$f>;die "empty CSV $p\n" unless defined $h;my $names=csv_fields($h);my @rows;
 while(my $line=<$f>){my $v=csv_fields($line);die "wrong CSV column count\n" unless @$v==@$names;my %r;@r{@$names}=@$v;push @rows,\%r;}
 $f->close or die "close gzip $p: $GunzipError\n";return \@rows;
}
sub capture{
 my($core,$command,$prefix,$kind)=@_;$core->send($command);my $meta=$core->line;die "bad trace response $meta\n" unless index($meta,$kind.'_begin ')==0;my $p="$prefix.csv.gz";my $f=IO::Compress::Gzip->new($p) or die "gzip $p: $GzipError\n";
 while(1){my $x=$core->line;last if $x eq $kind.'_end';print {$f} "$x\n" or die "write gzip $p: $GzipError\n";}$f->close or die "close gzip $p: $GzipError\n";return($p,fields($meta));
}
sub capture_tensors {
 my($core,$path,$weights,$dies)=@_;
 $core->send('tensormeta');my $begin=$core->line;
 die "missing live tensor inventory\n" unless $begin=~/\Atensormeta_begin /;
 my $meta=fields($begin);my @lines;
 while(1){my $line=$core->line;last if $line eq 'tensormeta_end';push @lines,$line;
  die "unreasonably large tensor inventory\n" if @lines>1_000_000;}
 save($path,{begin=>$begin,lines=>\@lines});
 die "tensor die count differs from config\n" unless ($meta->{dies}//0)==$dies;
 return BenchManifest::tensor_inventory(\@lines,$meta,$weights);
}
sub request{
 my($core,$ids,$outputs,$prefix,$trace,$dies,$expected_resume)=@_;
 my(@raw,@tokens,@observed);
 my $text='chat '.($outputs-1).' '.join(' ',@$ids);
 my $row={request=>$text,status=>'incomplete',generated_ids=>\@tokens,
          client_observed_ns=>\@observed,raw=>\@raw,
          visible_positions=>[map {scalar(@$ids)+$_} 0..$outputs-1]};
 save("$prefix.json",$row);
 my $ok=eval {
  $row->{snapshot_command}='snap '.(@$ids-1);
  $row->{snapshot_reply}=$core->one($row->{snapshot_command});
  $row->{request_start_ns}=monotonic_ns();
  $core->send($text);
  while(1){
   my $line=$core->line;my $now=monotonic_ns();push @raw,$line;
   if($line=~/\Atok ([0-9]+)\z/){
    die "generated token outside model vocabulary\n" if $1>=248320;
    push @tokens,0+$1;push @observed,$now;
    die "too many generated outputs\n" if @tokens>$outputs;
   } elsif($line=~/\Adone ([0-9]+) ([01]) ([0-9]+) ([0-9]+(?:\.[0-9]+)?) ([0-9]+(?:\.[0-9]+)?) ([0-9]+) ([0-9]+)\z/){
    $row->{done}={generated=>0+$1,ended=>0+$2,prompt=>0+$3,run_ms=>0+$4,first_ms=>0+$5,resumed=>0+$6,snapshot=>0+$7};
    $row->{request_end_ns}=$now;last;
   } else {die "unexpected request response $line\n";}
  }
  # The response is checkpointed before post-request validation/metadata calls.
  save("$prefix.json",$row);
  my $done=$row->{done};
  die "incomplete or stopped request\n" unless @tokens==$outputs && $done->{generated}==$outputs
       && !$done->{ended} && $done->{prompt}==@$ids;
  die "prefix snapshot was not restored\n" if defined($expected_resume) && $done->{resumed}!=$expected_resume;
  my @intervals=map {$observed[$_]-$observed[$_-1]}1..$#observed;
  die "invalid client token times\n" if grep {$_<=0}@intervals;
  $row->{intervals_ns}=\@intervals;$row->{decode_intervals}=scalar @intervals;
  $row->{client_decode_ms}=sum(\@intervals)/1e6;
  $row->{request_ttft_ms}=($observed[0]-$row->{request_start_ns})/1e6;
  $row->{request_completion_ms}=($row->{request_end_ns}-$row->{request_start_ns})/1e6;
  $row->{reported_post_first_ms}=$done->{run_ms}-$done->{first_ms};
  $row->{requestmeta_raw}=$core->one('requestmeta');
  die "unexpected requestmeta response\n" unless $row->{requestmeta_raw}=~/\Arequestmeta /;
  my $meta=fields($row->{requestmeta_raw});$row->{requestmeta}=$meta;
  my $timing=BenchAcceptance::requestmeta($meta,scalar(@$ids),$outputs,$core->{last_request});
  $core->{last_request}=0+$meta->{request};$row->{timing}=$timing;
  die "requestmeta and done disagree on prefix state\n"
      unless $meta->{resumed}==$done->{resumed} && $meta->{snapshot}==$done->{snapshot};
  $row->{lifecycle}=trace_lifecycle($meta,require_current=>1,expected_requested=>@$ids+$outputs-1,allow_ended=>0);
  $row->{engine_observed_decode_ms}=$timing->{engine_observed_decode_ms};
  $row->{client_minus_engine_interval_ms}=$row->{client_decode_ms}-$row->{engine_observed_decode_ms};
  if($trace){
   my(@traces,$request_id);$row->{trace_files}=[];
   for my $die(0..$dies-1){
    my($p,$m)=capture($core,"trace $die","$prefix-die-$die",'trace');
    trace_lifecycle($m,require_current=>1,expected_requested=>@$ids+$outputs-1,allow_ended=>0);
    die "wrong trace schema or request dimensions\n" unless ($m->{schema}//0)==2
        && $m->{count}==$outputs && $m->{first}==@$ids-1 && $m->{die}==$die;
    $request_id=$m->{request} unless defined $request_id;
    die "different die trace requests\n" unless $m->{request} eq $request_id;
    my $s=phase_summary(read_csv_gzip($p),$m);save("$p.summary.json",$s);
    push @{$row->{trace_files}},{kind=>'trace',die=>$die,metadata=>$m,raw=>identity($p),summary=>identity("$p.summary.json")};
    push @traces,$s;
   }
   die "stale trace request\n" if defined($core->{last_trace_request}) && $request_id<=$core->{last_trace_request};
   $core->{last_trace_request}=0+$request_id;
   my($p,$m)=capture($core,'hosttrace',"$prefix-host",'hosttrace');
   trace_lifecycle($m,require_current=>1,expected_requested=>@$ids+$outputs-1,allow_ended=>0);
   die "different host trace request\n" unless $m->{request} eq $request_id
       && $m->{count}==$outputs && $m->{first}==@$ids-1 && ($m->{schema}//0)==2;
   my $host=host_summary(read_csv_gzip($p),$m);
   push @{$row->{trace_files}},{kind=>'hosttrace',metadata=>$m,raw=>identity($p)};
   $row->{device_publication_decode_ms_by_die}=[map {$_->{decode_ms}}@traces];
   $row->{attention_execution}=BenchAcceptance::attention_work(\@traces,scalar(@$ids),$outputs);
   $row->{trace_op_counts_by_die}=[map {$_->{op_counts}}@traces];
   $row->{trace_op_counts_all_outputs_by_die}=[map {$_->{op_counts_all_outputs}}@traces];
   $row->{trace_op_mean_us_by_die}=[map {$_->{op_mean_us}}@traces];
   $row->{trace_selected_keys_by_die}=[map {$_->{selected_keys}}@traces];
   $row->{trace_host_observed_decode_ms}=($host->[-1]{observed_ns}-$host->[0]{observed_ns})/1e6;
   $row->{engine_minus_gpu0_interval_ms}=$row->{engine_observed_decode_ms}-$traces[0]{decode_ms};
  }
  $row->{gpu_after}=gpu();$row->{status}='complete';1;
 };
 if(!$ok){my $error=$@;$row->{status}='failed';$row->{error}="$error";save("$prefix.json",$row);die $error;}
 save("$prefix.json",$row);return $row;
}
# Preserve the historical integer-seeded MT19937/choice fixture exactly.
sub prompt_ids{
 my($count)=@_;my @mt=(19650218);for my $i(1..623){$mt[$i]=(1812433253*($mt[$i-1]^($mt[$i-1]>>30))+$i)&0xffffffff;}my $i=1;
 for(1..624){$mt[$i]=(($mt[$i]^(($mt[$i-1]^($mt[$i-1]>>30))*1664525))+4242)&0xffffffff;$i++;if($i>=624){$mt[0]=$mt[623];$i=1;}}
 for(1..623){$mt[$i]=(($mt[$i]^(($mt[$i-1]^($mt[$i-1]>>30))*1566083941))-$i)&0xffffffff;$i++;if($i>=624){$mt[0]=$mt[623];$i=1;}}
 $mt[0]=0x80000000;my $index=624;
 my $word=sub{if($index>=624){for my $k(0..623){my $y=($mt[$k]&0x80000000)|($mt[($k+1)%624]&0x7fffffff);$mt[$k]=($mt[($k+397)%624]^($y>>1)^(($y&1)?0x9908b0df:0))&0xffffffff;}$index=0;}my $y=$mt[$index++];$y^=$y>>11;$y^=($y<<7)&0x9d2c5680;$y^=($y<<15)&0xefc60000;$y^=$y>>18;return $y&0xffffffff;};
 my @vocab=(198,220,11,13,279,315,323,374,499,264,311,389);my @ids;
 for(1..$count){my $r;do{$r=$word->()>>28;}while $r>=@vocab;push @ids,$vocab[$r];}return \@ids;
}
sub main{
 my @args=@_;my @original=($^X,$0,@args);
 my %a=(depths=>'2048,2051,2052,3100,8000,65000,262000',outputs=>129,pairs=>5,
        warmup_outputs=>9,timeout=>18000,trace=>0,artifact=>[],source_dir=>"$FindBin::Bin/..");
 GetOptionsFromArray(\@args,'host-socket=s'=>\$a{host_socket},'vram-socket=s'=>\$a{vram_socket},
  'out=s'=>\$a{out},'depths=s'=>\$a{depths},'outputs=i'=>\$a{outputs},'pairs=i'=>\$a{pairs},
  'warmup-outputs=i'=>\$a{warmup_outputs},'trace'=>\$a{trace},'timeout=f'=>\$a{timeout},
  'source-sha=s'=>\$a{source_sha},'source-dir=s'=>\$a{source_dir},'dirty-patch=s'=>\$a{dirty_patch},
  'artifact=s@'=>\$a{artifact},'weight-manifest=s'=>\$a{weight_manifest},
  'build-record=s'=>\$a{build_record},'core-records=s'=>\$a{core_records},
  'prompt-file=s'=>\$a{prompt_file},'answer-reference=s'=>\$a{answer_reference},
  'attribution-run=s'=>\$a{attribution_run},'profiler-record=s'=>\$a{profiler_record}) or die "invalid benchmark options\n";
 die "required option missing: out\n" unless defined($a{out}) && length($a{out});
 die "output already exists: $a{out}\n" if -e $a{out};
 make_path($a{out});my $out=abs_path($a{out});$a{out}=$out;
 my $manifest={schema=>2,argv=>\@original,utc=>strftime('%Y-%m-%dT%H:%M:%SZ',gmtime),
               mode=>$a{trace}?'attribution':'throughput',state=>'initializing'};
 save("$out/manifest.json",$manifest);
 my(%cores,@results,$lock);my $exitcode=1;
 local $SIG{INT}=sub {die "interrupted by SIGINT\n"};
 local $SIG{TERM}=sub {die "interrupted by SIGTERM\n"};
 my $ok=eval{
  die "unexpected arguments: @args\n" if @args;
  for my $k(qw(host_socket vram_socket source_sha weight_manifest build_record core_records)){
   die "required option missing: $k\n" unless defined($a{$k}) && length($a{$k});
  }
  die "host and VRAM must use distinct sockets\n" if $a{host_socket} eq $a{vram_socket};
  die "positive timeout required\n" unless $a{timeout}>0;
  my @depths=split /,/,$a{depths},-1;
  die "unique ascending positive depths required\n" unless @depths
      && !(grep {$_!~/\A[0-9]+\z/ || $_<1 || $_>262144} @depths);
  for my $i(1..$#depths){die "unique ascending positive depths required\n" if $depths[$i]<=$depths[$i-1];}
  @depths=map {0+$_}@depths;
  $manifest->{policy}=BenchAcceptance::policy(\%a,\@depths);
  # One lock covers preflight, all warmups/measurements and final metadata capture.
  sysopen($lock,'/tmp/flash-sass-engine-validation.lock',O_RDWR|O_CREAT,0600)
      or die "open shared validation lock: $!\n";
  flock($lock,LOCK_EX|LOCK_NB) or die "shared validation lock is busy; no requests sent\n";
  $manifest->{lock}={path=>'/tmp/flash-sass-engine-validation.lock',pid=>0+$$,acquired_utc=>strftime('%Y-%m-%dT%H:%M:%SZ',gmtime)};
  my $ps=run(['ps','-eo','pid,ppid,stat,comm']);
  save("$out/processes-before.json",$ps);die "process preflight failed\n" if $ps->{exit};
  my $smi=run(['nvidia-smi']);save("$out/nvidia-smi-before.json",$smi);
  die "nvidia-smi preflight failed\n" if $smi->{exit};
  $manifest->{gpu_before}=gpu();$manifest->{driver_counters_before}=driver_snapshot();
  $manifest->{host}=run(['uname','-a']);save("$out/manifest.json",$manifest);
  my $provenance=BenchManifest::capture(\%a);
  $manifest->{provenance}=$provenance;
  $manifest->{source_sha}=$a{source_sha};
  $manifest->{source_dirty_sha256}=$provenance->{source}{diff_sha256};
  $manifest->{weights}=$provenance->{weights};
  $manifest->{runner}=identity(__FILE__);
  my $ids=defined($a{prompt_file})?load_json($a{prompt_file}):prompt_ids($depths[-1]);
  die "prompt must contain enough post-template token IDs in the fixed model vocabulary\n"
      unless ref($ids) eq 'ARRAY' && @$ids>=$depths[-1]
      && !(grep {!BenchAcceptance::uint($_) || $_>=248320} @$ids);
  $ids=[map {0+$_} @$ids];save("$out/prompt-ids.json",$ids);
  $manifest->{prompt}=identity("$out/prompt-ids.json");
  $manifest->{prompt_ids_sha256}=BenchAcceptance::canonical_hash($ids);
  $manifest->{fixture}=defined($a{prompt_file})?'supplied post-template token IDs':'historical synthetic token fixture; diagnostic only';
  $manifest->{sampling}={seed=>0,temperature=>0,top_k=>0,top_p=>1,stop_ids=>[]};
  $manifest->{instrumentation}={phase_trace=>$a{trace}?JSON::PP::true:JSON::PP::false,
      profiler_counters=>'not collected in throughput runs',
      client_clock=>'CLOCK_MONOTONIC',raw_response_checkpoint=>'after completion and on caught failure; no per-token disk write',
      requestmeta=>'always enabled host boundary observations; settings must match across placements'};
  my $reference;
  if(defined($a{answer_reference})){
   $reference=BenchAcceptance::answer_reference(load_json($a{answer_reference}),$ids,$a{outputs});
   save("$out/answer-reference.json",$reference);$manifest->{answer_reference}=identity("$out/answer-reference.json");
  }
  my %configs;
  for my $item(['host',$a{host_socket}],['vram',$a{vram_socket}]){
   my($name,$sock)=@$item;
   my $c=PairedDepthBench::Core->new($sock,$a{timeout},"$out/$name-commands.jsonl");$cores{$name}=$c;
   die "socket peer does not match verified $name process\n"
       unless $c->{peer_pid}==$provenance->{core_launch_records}{$name}{pid};
   $manifest->{socket_peers}{$name}={map {$_=>$c->{$_}}qw(peer_pid peer_uid peer_gid)};
   my $reply=$c->one('config');die "unexpected config response\n" unless $reply=~/\Aconfig /;
   $configs{$name}=fields($reply);
  }
  $manifest->{configs}=\%configs;save("$out/manifest.json",$manifest);
  die "host and VRAM sockets refer to the same process\n" if $cores{host}{peer_pid}==$cores{vram}{peer_pid};
  matched($configs{host},$configs{vram},{kv_host=>1});
  die "wrong realized KV placement\n" unless $configs{host}{kv_host} eq '1' && $configs{vram}{kv_host} eq '0';
  BenchAcceptance::capabilities(\%configs,$a{trace},$a{outputs});
  BenchManifest::validate_runtime($provenance,\%configs);
  for my $name(qw(host vram)) {
   $manifest->{tensors}{$name}=capture_tensors($cores{$name},"$out/$name-tensors.json",$provenance->{weights},$configs{$name}{dies});
  }
  matched($manifest->{tensors}{host},$manifest->{tensors}{vram},{});
  save("$out/manifest.json",$manifest);
  # Capability/provenance failures above happen before changing sampling or snapshots.
  for my $c(values %cores){$c->one('stop');$c->one('sample 0 0 0 1');}
  my($cap,$dies)=map {0+$configs{host}{$_}}qw(maxt dies);
  for my $depth(@depths){
   my $needed=$depth+($a{outputs}>$a{warmup_outputs}?$a{outputs}:$a{warmup_outputs})-1;
   if($needed>$cap){
    my $r={depth=>$depth,status=>'capacity_skip',required_positions=>$needed,capacity=>$cap,
           reason=>'reported equal host/VRAM engine capacity'};
    push @results,$r;save("$out/results.json",\@results);print $JSON->encode($r),"\n";next;
   }
   die "missing validated answer for supported depth $depth\n" if $reference && !$reference->{depths}{$depth};
   my @prompt=@$ids[0..$depth-1];
   for my $name(qw(host vram)){
    my $r=request($cores{$name},\@prompt,$a{warmup_outputs},"$out/warmup-$depth-$name",0,$dies,undef);
    $r->{scope}='warmup; excluded from aggregates';save("$out/warmup-$depth-$name.json",$r);
   }
   for my $pair(0..$a{pairs}-1){
    my @order=$pair%2?qw(vram host):qw(host vram);my %paired;
    for my $name(@order){
     my $prefix="$out/depth-$depth-pair-$pair-$name";
     my $r=request($cores{$name},\@prompt,$a{outputs},$prefix,$a{trace},$dies,$depth-1);
     @$r{qw(depth pair placement order)}=($depth,$pair,$name,[@order]);
     my $answer_ok=eval{$r->{answer_valid}=BenchAcceptance::check_answer($reference,$depth,$r);1};
     if(!$answer_ok){my $error=$@;$r->{answer_valid}=JSON::PP::false;$r->{status}='invalid_answer';$r->{error}="$error";save("$prefix.json",$r);die $error;}
     my $after=fields($cores{$name}->one('config'));
     matched($configs{$name},$after,{});$r->{config_after}=$after;
     save("$prefix.json",$r);push @results,$r;$paired{$name}=$r;
     # Checkpoint each side; if its partner fails, the first side remains raw evidence.
     save("$out/results.json",\@results);
     print $JSON->encode({depth=>$depth,pair=>$pair,placement=>$name,
       tok_s=>1000*$r->{decode_intervals}/$r->{client_decode_ms},resumed=>$r->{done}{resumed},
       answer_valid=>$r->{answer_valid},scope=>$manifest->{mode}}),"\n";
    }
    die "matched placements generated different tokens\n"
        unless $JSON->encode($paired{host}{generated_ids}) eq $JSON->encode($paired{vram}{generated_ids});
   }
  }
  my $runs=scalar(grep {$_->{status} eq 'complete'} @results);
  die "no supported depths were measured\n" unless $runs;
  my @gaps;
  push @gaps,'custom depth/output policy does not satisfy the complete issue-14 matrix' unless $manifest->{policy}{standard_matrix};
  push @gaps,'synthetic prompt fixture is diagnostic only' unless defined($a{prompt_file});
  push @gaps,'independent answer-reference evidence absent' unless $reference;
  if($a{trace}){push @gaps,'instrumented attribution is not authoritative throughput';}
  elsif(defined($a{attribution_run})){
   BenchAcceptance::compare_attribution($out,$manifest,\@results,$a{attribution_run});
  }else{push @gaps,'matched separately collected attribution and trace-overhead evidence absent';}
  if(defined($a{profiler_record})){
   my $profiler=BenchAcceptance::profiler_record(load_json($a{profiler_record}),$manifest,$out);
   save("$out/profiler-record.json",$profiler);$manifest->{profiler_record}=identity("$out/profiler-record.json");
  }else{push @gaps,'separately collected profiler counters absent';}
  my $summary=BenchAcceptance::summary(\@results,\@depths,$a{pairs},!@gaps);
  save("$out/".(@gaps?'diagnostic-summary.json':'summary.json'),$summary);
  $manifest->{driver_counters_after}=driver_snapshot();$manifest->{gpu_after}=gpu();
  $manifest->{state}='collected';save("$out/manifest.json",$manifest);
  save("$out/status.json",{status=>@gaps?'COLLECTED_INCOMPLETE':'PASS',collection_complete=>JSON::PP::true,
      acceptance_complete=>@gaps?JSON::PP::false:JSON::PP::true,runs=>$runs,gaps=>\@gaps});
  $exitcode=@gaps?2:0;1;
 };
 my $error=$@;
 if(!$ok){
  $manifest->{state}='failed';save("$out/manifest.json",$manifest);
  save("$out/FAILED.json",{error=>"$error"});
  save("$out/status.json",{status=>'FAILED',collection_complete=>JSON::PP::false,
      acceptance_complete=>JSON::PP::false,error=>"$error"});
 }
 for my $c(values %cores){eval{$c->close};}
 if($lock){flock($lock,LOCK_UN);close $lock;$manifest->{lock_released_utc}=strftime('%Y-%m-%dT%H:%M:%SZ',gmtime) if $manifest->{lock};}
 save("$out/manifest.json",$manifest);
 save("$out/results.json",\@results);
 save("$out/exit.json",{exit=>$exitcode,argv=>\@original});
 if(!$ok){print STDERR $error;return $exitcode;}
 return $exitcode;
}
package PairedDepthBench::Core;
use strict;use warnings;use IO::Select;use IO::Socket::UNIX;use Socket qw(SOCK_STREAM SOL_SOCKET SO_PEERCRED);
sub new{my($class,$path,$timeout,$journal)=@_;my $s=IO::Socket::UNIX->new(Type=>SOCK_STREAM,Peer=>$path,Timeout=>$timeout) or die "connect $path: $!\n";binmode $s;my $cred=getsockopt($s,SOL_SOCKET,SO_PEERCRED);die "cannot read socket peer credentials\n" unless defined($cred) && length($cred)==12;my($pid,$uid,$gid)=unpack('i3',$cred);return bless {socket=>$s,timeout=>$timeout,buffer=>'',journal=>$journal,peer_pid=>$pid,peer_uid=>$uid,peer_gid=>$gid},$class;}
sub record{my($self,$kind,$text)=@_;return unless $self->{journal};open my $f,'>>:raw',$self->{journal} or die "open command journal: $!\n";print {$f} $PairedDepthBench::JSON->encode({kind=>$kind,text=>$text,monotonic_ns=>PairedDepthBench::monotonic_ns()}),"\n" or die "write command journal: $!\n";close $f or die "close command journal: $!\n";}
sub send{
 my($self,$text)=@_;my $bytes="$text\n";my $at=0;my $select=IO::Select->new($self->{socket});local $SIG{PIPE}='IGNORE';
 $self->record('command',$text);
 while($at<length $bytes){die "core send timeout\n" unless $select->can_write($self->{timeout});my $n=syswrite($self->{socket},$bytes,length($bytes)-$at,$at);if(!defined $n){next if $!{EINTR};die "core send failed: $!\n";}die "core disconnected\n" unless $n;$at+=$n;}
}
sub line{
 my($self)=@_;my $select=IO::Select->new($self->{socket});
 while(index($self->{buffer},"\n")<0){die "core read timeout\n" unless $select->can_read($self->{timeout});my $n=sysread($self->{socket},my $buf,65536);if(!defined $n){next if $!{EINTR};die "core read failed: $!\n";}die "core disconnected\n" unless $n;$self->{buffer}.=$buf;}
 my $at=index($self->{buffer},"\n");my $line=substr($self->{buffer},0,$at+1,'');$line=~s/\n\z//;return $line;
}
sub one{my($self,$text)=@_;$self->send($text);my $line=$self->line;$self->record('response',$line);die "$line\n" if index($line,'error')==0;return $line;}
sub close{my($self)=@_;if($self->{socket}){close $self->{socket};delete $self->{socket};}}
package PairedDepthBench;
unless(caller){$|=1;exit main(@ARGV);}1;
