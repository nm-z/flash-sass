#!/usr/bin/env perl
package PairedDepthBench;
use strict;use warnings;
use B qw(svref_2object SVp_POK);use Cwd 'abs_path';use Digest::SHA;
use File::Path 'make_path';use Getopt::Long 'GetOptionsFromArray';
use IO::Compress::Gzip qw($GzipError);use IO::Uncompress::Gunzip qw($GunzipError);
use IO::Select;use IO::Socket::UNIX;use IPC::Open3;use JSON::PP;
use POSIX qw(ceil strftime);use Socket 'SOCK_STREAM';use Symbol 'gensym';
use Time::HiRes qw(clock_gettime CLOCK_MONOTONIC);
# Connect only to supplied cores. Never reload a core or weight holder.
# Fixed synthetic tokens exercise the checkpoint, not answer quality.
# Historical absent lifecycle metadata stays unknown; live traces require it.
our $JSON=JSON::PP->new->canonical->allow_nonref;
our $PRETTY=JSON::PP->new->canonical->pretty->allow_nonref;
our @GPU_QUERY=qw(index uuid driver_version pci.bus_id memory.used memory.free utilization.gpu temperature.gpu clocks.current.sm clocks.current.memory ecc.mode.current pcie.link.gen.current pcie.link.width.current clocks_throttle_reasons.active clocks_throttle_reasons.sw_power_cap clocks_throttle_reasons.sw_thermal_slowdown power.draw power.limit);
sub read_text{my($p)=@_;open my $f,'<:raw',$p or die "open $p: $!\n";local $/;my $s=<$f>;close $f or die "close $p: $!\n";return defined($s)?$s:'';}
sub save{my($p,$v)=@_;open my $f,'>:raw',$p or die "open $p: $!\n";print {$f} $PRETTY->encode($v) or die "write $p: $!\n";close $f or die "close $p: $!\n";}
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
sub gpu{return run(['nvidia-smi','--query-gpu='.join(',',@GPU_QUERY),'--format=csv']);}
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
sub matched{
 my($a,$b,$allow)=@_;my %allow=ref($allow) eq 'HASH'?%$allow:map {$_=>1}@$allow;
 my %keys=map {$_=>1}(keys %$a,keys %$b);my %diff;
 for my $k(keys %keys){next if $allow{$k};my($av,$bv)=($a->{$k},$b->{$k});my $same=!defined($av)&&!defined($bv);$same||=defined($av)&&defined($bv)&&$JSON->encode($av) eq $JSON->encode($bv);$diff{$k}=[$av,$bv] unless $same;}
 die "unexplained manifest differences: ".$JSON->encode(\%diff)."\n" if keys %diff;
}
sub trace_lifecycle{
 my($m,%o)=@_;my @required=qw(status aborted ended completed requested);my @present=grep {exists $m->{$_}}@required;
 if(!@present){die "live trace lacks lifecycle metadata\n" if $o{require_current};return {reported=>JSON::PP::false,status=>'legacy_unreported'};}
 die "partial trace lifecycle metadata\n" if @present!=@required;
 for my $k(qw(aborted ended)){die "invalid trace lifecycle boolean\n" unless defined($m->{$k}) && !ref($m->{$k}) && $m->{$k}=~/\A[01]\z/;}
 for my $k(qw(completed requested)){die "invalid trace lifecycle count $k\n" unless is_string($m->{$k}) && $m->{$k}=~/\A[0-9]+\z/;}
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
sub request{
 my($core,$ids,$outputs,$prefix,$trace,$dies,$expected_resume)=@_;$core->one('snap '.(@$ids-1));my $text='chat '.($outputs-1).' '.join(' ',@$ids);my $started=monotonic_ns();$core->send($text);my(@raw,@tokens,@observed,$done,$now);
 while(1){my $line=$core->line;$now=monotonic_ns();push @raw,$line;
  if(index($line,'tok ')==0){my @x=split /\s+/,$line;push @tokens,0+$x[1];push @observed,$now;}
  elsif(index($line,'done ')==0){my @x=split /\s+/,$line;die "malformed done response\n" unless @x>=8;$done={generated=>0+$x[1],ended=>0+$x[2],prompt=>0+$x[3],run_ms=>0+$x[4],first_ms=>0+$x[5],resumed=>0+$x[6],snapshot=>0+$x[7]};last;}
  else{die "unexpected request response $line\n";}
 }
 my $row={request=>$text,request_start_ns=>$started,request_end_ns=>$now,generated_ids=>\@tokens,client_observed_ns=>\@observed,raw=>\@raw,done=>$done};save("$prefix.json",$row);
 die "incomplete or stopped request\n" unless @tokens==$outputs && $done->{generated}==$outputs && !$done->{ended} && $done->{prompt}==@$ids;
 die "prefix snapshot was not restored\n" if defined($expected_resume) && $done->{resumed}!=$expected_resume;
 my @intervals=map {$observed[$_]-$observed[$_-1]}1..$#observed;die "invalid client token times\n" if grep {$_<=0}@intervals;
 $row->{intervals_ns}=\@intervals;$row->{decode_intervals}=scalar @intervals;$row->{client_decode_ms}=sum(\@intervals)/1e6;
 $row->{request_ttft_ms}=($observed[0]-$started)/1e6;$row->{request_completion_ms}=($now-$started)/1e6;$row->{reported_post_first_ms}=$done->{run_ms}-$done->{first_ms};
 if($trace){my(@traces,$request_id);
  for my $die(0..$dies-1){my($p,$m)=capture($core,"trace $die","$prefix-die-$die",'trace');trace_lifecycle($m,require_current=>1,expected_requested=>@$ids+$outputs-1,allow_ended=>0);
   die "wrong trace request dimensions\n" unless $m->{count}==$outputs && $m->{first}==@$ids-1 && $m->{die}==$die;
   $request_id=$m->{request} unless defined $request_id;die "different die trace requests\n" unless $m->{request} eq $request_id;
   my $s=phase_summary(read_csv_gzip($p),$m);save("$p.summary.json",$s);push @traces,$s;
  }
  my($p,$m)=capture($core,'hosttrace',"$prefix-host",'hosttrace');$row->{lifecycle}=trace_lifecycle($m,require_current=>1,expected_requested=>@$ids+$outputs-1,allow_ended=>0);
  die "different host trace request\n" unless $m->{request} eq $request_id && $m->{count}==$outputs;
  my $host=host_summary(read_csv_gzip($p),$m);$row->{device_publication_decode_ms_by_die}=[map {$_->{decode_ms}}@traces];
  $row->{engine_observed_decode_ms}=($host->[-1]{observed_ns}-$host->[0]{observed_ns})/1e6;
  $row->{client_minus_engine_interval_ms}=$row->{client_decode_ms}-$row->{engine_observed_decode_ms};$row->{engine_minus_gpu0_interval_ms}=$row->{engine_observed_decode_ms}-$traces[0]{decode_ms};
 }
 $row->{gpu_after}=gpu();$row->{status}='complete';save("$prefix.json",$row);return $row;
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
 my @args=@_;my @original=($0,@args);my %a=(depths=>'2048,2051,2052,3100,8000,65000,262000',outputs=>129,pairs=>5,warmup_outputs=>9,timeout=>18000,trace=>0,artifact=>[]);
 GetOptionsFromArray(\@args,'host-socket=s'=>\$a{host_socket},'vram-socket=s'=>\$a{vram_socket},'out=s'=>\$a{out},'depths=s'=>\$a{depths},'outputs=i'=>\$a{outputs},'pairs=i'=>\$a{pairs},'warmup-outputs=i'=>\$a{warmup_outputs},'trace'=>\$a{trace},'timeout=f'=>\$a{timeout},'source-sha=s'=>\$a{source_sha},'dirty-patch=s'=>\$a{dirty_patch},'artifact=s@'=>\$a{artifact},'weight-manifest=s'=>\$a{weight_manifest},'build-record=s'=>\$a{build_record},'core-records=s'=>\$a{core_records}) or die "invalid benchmark options\n";
 die "unexpected arguments: @args\n" if @args;
 for my $k(qw(host_socket vram_socket out source_sha weight_manifest build_record core_records)){die "required option missing: $k\n" unless defined($a{$k})&&length($a{$k});}
 die "at least one --artifact is required\n" unless @{$a{artifact}};
 die "source-sha must be a full SHA\n" unless $a{source_sha}=~/\A[0-9a-f]{40}\z/;
 die "outputs2..256, pairs>=5, warmup>=2\n" if $a{outputs}<2 || $a{outputs}>256 || $a{pairs}<5 || $a{warmup_outputs}<2;
 my @depths=split /,/,$a{depths},-1;die "unique ascending positive depths required\n" unless @depths && !(grep {$_!~/\A[0-9]+\z/||$_<1}@depths);
 for my $i(1..$#depths){die "unique ascending positive depths required\n" if $depths[$i]<=$depths[$i-1];}@depths=map {0+$_}@depths;
 my $launch=core_records($a{core_records});die "output already exists: $a{out}\n" if -e $a{out};make_path($a{out});my $out=$a{out};
 my $ids=prompt_ids($depths[-1]);save("$out/prompt-ids.json",$ids);my $weights=load_json($a{weight_manifest});
 for my $w(@$weights){my $s=file_stamp($w->{path});die "stale weight identity\n" if $s->{bytes}!=$w->{bytes} || $s->{mtime_ns}!=$w->{mtime_ns};}
 my %environment=map {$_=>$ENV{$_}}grep {/\A(?:FLASH_|CUDA_|SPLIT_|ATTN_|DIE_)/}keys %ENV;
 my $manifest={argv=>\@original,source_sha=>$a{source_sha},dirty_patch=>defined($a{dirty_patch})?read_text($a{dirty_patch}):undef,runner=>identity(__FILE__),artifacts=>[map {identity($_)}@{$a{artifact}}],weights=>$weights,build=>load_json($a{build_record}),core_launch_records=>$launch,core_records_identity=>identity($a{core_records}),prompt=>identity("$out/prompt-ids.json"),sampling=>{seed=>0,temperature=>0,stop_ids=>[]},warmup_outputs=>$a{warmup_outputs},pairs=>$a{pairs},outputs=>$a{outputs},fixture=>'synthetic fixed token IDs; real checkpoint forward; no answer-quality acceptance',utc=>strftime('%Y-%m-%dT%H:%M:%SZ',gmtime),gpu_before=>gpu(),host=>run(['uname','-a']),environment=>\%environment,driver_counters_before=>driver_snapshot()};
 my(%cores,@results);my $exitcode=1;my $ok=eval{
  my %configs;
  for my $item(['host',$a{host_socket}],['vram',$a{vram_socket}]){my($name,$sock)=@$item;my $c=PairedDepthBench::Core->new($sock,$a{timeout});$cores{$name}=$c;$configs{$name}=fields($c->one('config'));$c->one('stop');$c->one('sample 0 0 0 1');}
  matched($configs{host},$configs{vram},{kv_host=>1});die "wrong realized KV placement\n" unless $configs{host}{kv_host} eq '1' && $configs{vram}{kv_host} eq '0';
  my $cfg=$configs{host};die "requires full48-layer budget512 QSA\n" unless $cfg->{layers} eq '48/48' && $cfg->{nblk} eq '512';
  die "insufficient trace capacity\n" if $a{trace}&&($cfg->{trace_tokens}//0)<$a{outputs};
  $manifest->{configs}=\%configs;save("$out/manifest.json",$manifest);my($cap,$dies)=map {0+$cfg->{$_}}qw(maxt dies);
  for my $depth(@depths){if($depth+$a{outputs}-1>$cap){my $r={depth=>$depth,status=>'capacity_skip',capacity=>$cap};push @results,$r;print $JSON->encode($r),"\n";next;}
   my @prompt=@$ids[0..$depth-1];for my $name(qw(host vram)){my $r=request($cores{$name},\@prompt,$a{warmup_outputs},"$out/warmup-$depth-$name",0,$dies,undef);save("$out/warmup-$depth-$name.json",$r);}
   for my $pair(0..$a{pairs}-1){my @order=$pair%2?qw(vram host):qw(host vram);my %paired;
    for my $name(@order){my $prefix="$out/depth-$depth-pair-$pair-$name";my $r=request($cores{$name},\@prompt,$a{outputs},$prefix,$a{trace},$dies,$depth-1);@$r{qw(depth pair placement order)}=($depth,$pair,$name,[@order]);save("$prefix.json",$r);push @results,$r;$paired{$name}=$r;print $JSON->encode({depth=>$depth,pair=>$pair,placement=>$name,tok_s=>1000*$r->{decode_intervals}/$r->{client_decode_ms},resumed=>$r->{done}{resumed}}),"\n";}
    die "matched placements generated different tokens\n" unless $JSON->encode($paired{host}{generated_ids}) eq $JSON->encode($paired{vram}{generated_ids});save("$out/results.json",\@results);
   }
  }
  my @summary;for my $depth(@depths){for my $name(qw(host vram)){my @rr=grep {defined($_->{depth})&&$_->{depth}==$depth && defined($_->{placement})&&$_->{placement} eq $name && $_->{status} eq 'complete'}@results;next unless @rr;my @raw=map {map {$_/1e6}@{$_->{intervals_ns}}}@rr;my @means=map {$_->{client_decode_ms}/$_->{decode_intervals}}@rr;my $med=median(\@means);push @summary,{depth=>$depth,placement=>$name,runs=>scalar(@rr),intervals=>scalar(@raw),median_run_ms_per_token=>$med,p95_interval_ms=>p95(\@raw),median_tok_s=>1000/$med};}}
  $manifest->{driver_counters_after}=driver_snapshot();save("$out/manifest.json",$manifest);save("$out/summary.json",\@summary);save("$out/status.json",{status=>'PASS',runs=>scalar(grep {$_->{status} eq 'complete'}@results)});$exitcode=0;1;
 };
 my $error=$@;save("$out/FAILED.json",{error=>"$error"}) unless $ok;for my $c(values %cores){eval{$c->close};}save("$out/exit.json",{exit=>$exitcode});save("$out/results.json",\@results);die $error unless $ok;return 0;
}
package PairedDepthBench::Core;
use strict;use warnings;use IO::Select;use IO::Socket::UNIX;use Socket 'SOCK_STREAM';
sub new{my($class,$path,$timeout)=@_;my $s=IO::Socket::UNIX->new(Type=>SOCK_STREAM,Peer=>$path,Timeout=>$timeout) or die "connect $path: $!\n";binmode $s;return bless {socket=>$s,timeout=>$timeout,buffer=>''},$class;}
sub send{
 my($self,$text)=@_;my $bytes="$text\n";my $at=0;my $select=IO::Select->new($self->{socket});local $SIG{PIPE}='IGNORE';
 while($at<length $bytes){die "core send timeout\n" unless $select->can_write($self->{timeout});my $n=syswrite($self->{socket},$bytes,length($bytes)-$at,$at);if(!defined $n){next if $!{EINTR};die "core send failed: $!\n";}die "core disconnected\n" unless $n;$at+=$n;}
}
sub line{
 my($self)=@_;my $select=IO::Select->new($self->{socket});
 while(index($self->{buffer},"\n")<0){die "core read timeout\n" unless $select->can_read($self->{timeout});my $n=sysread($self->{socket},my $buf,65536);if(!defined $n){next if $!{EINTR};die "core read failed: $!\n";}die "core disconnected\n" unless $n;$self->{buffer}.=$buf;}
 my $at=index($self->{buffer},"\n");my $line=substr($self->{buffer},0,$at+1,'');$line=~s/\n\z//;die "$line\n" if index($line,'error')==0;return $line;
}
sub one{my($self,$text)=@_;$self->send($text);return $self->line;}
sub close{my($self)=@_;if($self->{socket}){close $self->{socket};delete $self->{socket};}}
package PairedDepthBench;
unless(caller){$|=1;exit main(@ARGV);}1;
