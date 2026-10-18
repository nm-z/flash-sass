#!/usr/bin/env perl
# Exact parent/remap evidence, with an exclusive and freshly checked GPU host.
use strict;
use warnings;
use FindBin;
use lib $FindBin::Bin;
use QsaEvidence qw(sha put run absolute new_output utc);
use Getopt::Long 'GetOptions';
use Fcntl qw(LOCK_EX LOCK_NB);
use File::Path 'make_path';
use File::Basename 'dirname';
use Time::HiRes qw(sleep alarm clock_gettime CLOCK_MONOTONIC);
use JSON::PP;
$|=1;

sub result {
 my($argv,$env,$timeout)=@_;
 my($stdout,$stderr,$exit)=run($argv,$env//\%ENV,$timeout//60);
 return {argv=>$argv,exit=>$exit,stdout=>$stdout,stderr=>$stderr};
}
sub image {
 my($p)=@_;$p=absolute($p);my @s=stat($p);@s or die "stat $p: $!\n";
 return {path=>$p,sha256=>sha($p),bytes=>$s[7]};
}
sub gpu {
 my($uuid)=@_;
 return result(['nvidia-smi','-i',$uuid,'--query-gpu=uuid,name,driver_version,pci.bus_id,temperature.gpu,clocks.current.sm,clocks.current.memory,utilization.gpu,memory.used,ecc.mode.current,pcie.link.gen.current,pcie.link.width.current','--format=csv'],\%ENV,30);
}
sub snapshot {
 my($uuid,$a)=@_;
 # Default: any resident compute context blocks. The explicit idle-resident
 # mode also checks all-device load and the local request queue; it permits
 # known resident contexts only on OTHER devices, never the measured device.
 my $s={utc=>utc(),
  compute_apps=>result(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],\%ENV,30),
  ps=>result(['ps','-eo','pid=,ppid=,stat=,comm=,args='],\%ENV,30),gpu=>gpu($uuid)};
 if($a && $a->{idle_resident_queue}) {
  $s->{all_gpu_load}=result(['nvidia-smi','--query-gpu=uuid,utilization.gpu,utilization.memory','--format=csv,noheader,nounits'],\%ENV,30);
  $s->{queue}=result(['curl','--fail','--silent','--show-error','--noproxy','*','--max-time','10',$a->{idle_resident_queue}],\%ENV,15);
 }
 return $s;
}
sub cpu_configured_llama {
 my($comm,$args)=@_;
 return 0 unless $comm eq 'llama-server' && $args=~m{\A(?:\S*/)?llama-server(?:\s|$)};
 # Require an explicit zero-offload configuration, rejecting conflicting
 # repeated flags. This only classifies a process with NO reported GPU app;
 # it does not override compute-app inventory or all-device load checks.
 my @layers=($args=~/(?:\A|\s)(?:--n-gpu-layers|--gpu-layers|-ngl)(?:=|\s+)(\S+)/g);
 return 0 unless @layers && !grep {$_ ne '0'} @layers;
 return 1;
}
sub busy_reasons {
 my($s,$a,$ignore_target_utilization)=@_;my $residents=$a && $a->{idle_resident_queue};
 for my $name(qw(compute_apps ps gpu),$residents?qw(all_gpu_load queue):()) {
  ref($s->{$name}) eq 'HASH' && defined($s->{$name}{exit}) && $s->{$name}{exit}==0
   or die "$name query failed; GPU idle state is unknown\n";
  defined($s->{$name}{stdout}) or die "$name query has no stdout\n";
 }
 $s->{ps}{stdout}=~/\S/ or die "ps returned no process inventory\n";
 $s->{gpu}{stdout}=~/GPU-[0-9a-f-]+/i or die "GPU metadata query returned no UUID\n";
 my(@busy,%resident_pid,%load_seen,%compute_pid,@cpu_workers);my $queue_port;
 if($residents) {
  ($queue_port)=$a->{idle_resident_queue}=~m{\Ahttp://127\.0\.0\.1:(\d+)/queue\z};
  defined($queue_port) && $queue_port>0 && $queue_port<=65535
   or die "idle-resident-queue must be http://127.0.0.1:PORT/queue\n";
  $a->{gpu}=~/\AGPU-[0-9a-f-]+\z/i or die "idle resident mode requires a GPU UUID, not an index\n";
  my $q=eval {JSON::PP->new->decode($s->{queue}{stdout})};
  ref($q) eq 'HASH' or die "invalid queue JSON\n";
  for my $key(qw(running waiting)) {
   defined($q->{$key}) && !ref($q->{$key}) && $q->{$key}=~/\A\d+\z/
    or die "invalid queue $key count\n";
   push @busy,"request queue $key=$q->{$key}" if $q->{$key};
  }
  for my $line(split /\n/,$s->{all_gpu_load}{stdout}) {
   next unless $line=~/\S/;
   my($id,$compute,$memory)=$line=~/\A\s*(GPU-[0-9a-f-]+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\z/i;
   defined($id) && $compute<=100 && $memory<=100 or die "invalid all-GPU utilization row\n";
   !$load_seen{lc $id}++ or die "duplicate GPU utilization row\n";
   push @busy,"GPU activity: $id compute=$compute memory=$memory"
    if ($compute || $memory) && !($ignore_target_utilization && lc($id) eq lc($a->{gpu}));
  }
  $load_seen{lc $a->{gpu}} or die "target GPU missing from utilization inventory\n";
 }
 for my $line(split /\n/,$s->{compute_apps}{stdout}) {
  next unless $line=~/\S/;
  my($id,$pid,$program)=$line=~/^\s*(GPU-[0-9a-f-]+)\s*,\s*(\d+)\s*,\s*([^,]+),/i;
  defined($id) or die "unrecognized nvidia-smi compute-app output\n";
  $compute_pid{$pid}=1;
  !$residents || $load_seen{lc $id} or die "compute-app GPU missing from utilization inventory\n";
  $program=~s/\s+$//;
  if($residents && lc($id) ne lc($a->{gpu}) && $program=~m{(?:\A|/)(hold|fence|flashcore)\z}) {
   my $name=$1;
   !exists($resident_pid{$pid}) || $resident_pid{$pid} eq $name or die "conflicting resident process identity\n";
   $resident_pid{$pid}=$name;
   next;
  }
  push @busy,"compute app: $line";
 }
 my %verified_resident;
 for my $line(split /\n/,$s->{ps}{stdout}) {
  next unless $line=~/\S/;
  my($pid,$ppid,$stat,$comm,$args)=$line=~/^\s*(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s*(.*)$/;
  defined($pid) or die "unrecognized ps process row\n";
  if(exists($resident_pid{$pid})) {
   my $name=$resident_pid{$pid};
   $comm eq $name && $args=~m{\A(?:\S*/)?\Q$name\E(?:\s|$)} && $stat!~/Z/
    or die "resident process identity mismatch: pid=$pid\n";
   $verified_resident{$pid}=1;
   next;
  }
  # Only the daemon serving the checked loopback queue may remain resident.
  if($residents && $comm eq 'flashd' && $stat!~/Z/ &&
     $args=~m{\A(?:\S*/)?flashd\s+\S+\s+\S+\s+127\.0\.0\.1:\Q$queue_port\E\s*\z}) {next;}
  if($residents && !$compute_pid{$pid} && $stat!~/Z/ && cpu_configured_llama($comm,$args)) {
   push @cpu_workers,{pid=>0+$pid,comm=>$comm,reason=>'explicit zero GPU layers; no reported compute app'};
   next;
  }
  # Conservatively catch known workers before their CUDA context is visible.
  # Inspect only the executable/script, not arguments of this runner or a shell
  # that merely mentions another program. Persist process identity, not argv.
  if($comm=~/^(?:flashcore|flashd|hold|fence|attnpartn_regre|qsa_oracle|qsa_test|model_test|attn_test|die_test|memtest|gemv.*test|llama-server|llama-cli|llama-perplexit)/
   || $args=~m{^(?:(?:\S*/)?(?:perl|ruby|bash|sh)(?:\s+-[^ \t]+)*\s+)?(?:\S*/)?(?:flashcore[^\s/]*|flashd|hold(?:[-_.][^\s/]*)?|fence(?:[-_.][^\s/]*)?|attnpartn_regress|attnpartn_model_compare|qsa_test|model_test|attn_test|die_test|memtest|gemv[^\s/]*test|llama-server|llama-cli|llama-perplexity)(?:\s|$)}) {
   push @busy,"worker process: pid=$pid ppid=$ppid stat=$stat comm=$comm";
  }
 }
 for my $pid(keys %resident_pid) {
  $verified_resident{$pid} or die "resident process missing from ps inventory: pid=$pid\n";
 }
 $s->{allowed_idle_residents}=[map {{pid=>0+$_,comm=>$resident_pid{$_}}} sort {$main::a<=>$main::b} keys %resident_pid] if $residents;
 $s->{cpu_configured_workers}=\@cpu_workers if $residents;
 return \@busy;
}
sub redact_process_args {
 my($s)=@_;
 if(ref($s->{ps}) eq 'HASH' && defined($s->{ps}{stdout})) {
  $s->{ps}{stdout}=~s/^(\s*\d+\s+\d+\s+\S+\s+\S+).*$/$1/mg;
  $s->{ps}{retained_fields}='pid ppid stat comm; arguments inspected but not retained';
 }
}
sub wait_idle {
 my($a,$out,$case,$collect)=@_;$collect//=\&snapshot;
 my $start=clock_gettime(CLOCK_MONOTONIC);my $attempt=0;
 while(1) {
  my $s=$collect->($a->{gpu},$a);$s->{case}=$case;$s->{attempt}=++$attempt;
  my $busy=busy_reasons($s,$a);
  # Full argv is needed only to recognize interpreter-launched GPU workers.
  # Do not save unrelated workers' command arguments in public evidence.
  redact_process_args($s);
  $s->{busy_reasons}=$busy;
  put("$out/gate-$case-$attempt.json",$s) if defined($out);
  return $s unless @$busy;
  print STDERR "Waiting for idle GPU host: ",join('; ',@$busy),"\n";
  my $left=$a->{wait_seconds}-(clock_gettime(CLOCK_MONOTONIC)-$start);
  die "BLOCKED_BUSY: host still has other GPU workers\n" if $left<=0;
  sleep($left<$a->{poll_seconds}?$left:$a->{poll_seconds});
 }
}
sub post_case_idle {
 my($a,$out,$case,$collect)=@_;$collect//=\&snapshot;
 my $start=clock_gettime(CLOCK_MONOTONIC);my @samples;
 my $limit=$a->{settle_seconds}//5;
 while(1) {
  my $s=$collect->($a->{gpu},$a);my($busy,$foreign,$error);
  eval {
   $busy=busy_reasons($s,$a);
   # The owned child has exited. Only its target's utilization counters may
   # lag behind that exit; target contexts, other-device activity, and queued
   # requests are still failures, including during the bounded settling wait.
   $foreign=busy_reasons($s,$a,1);
   1;
  } or $error=$@||"post-run idle inventory failed\n";
  $s->{case}=$case;$s->{post_run_sample}=1+@samples;
  $s->{busy_reasons}=$busy if defined($busy);
  $s->{inventory_error}=$error if defined($error);
  redact_process_args($s);
  push @samples,$s;
  put("$out/gate-$case-after-".scalar(@samples).".json",$s) if defined($out);
  die $error if defined($error);
  @$foreign and die "post-run contention: ".join('; ',@$foreign)."\n";
  return {status=>'IDLE',settle_limit_seconds=>$limit,snapshots=>\@samples} unless @$busy;
  my $left=$limit-(clock_gettime(CLOCK_MONOTONIC)-$start);
  die "post-run target utilization did not settle within ${limit}s\n" if $left<=0;
  sleep($left<0.25?$left:0.25);
 }
}
sub acquire_lock {
 my($path,$a)=@_;make_path(dirname($path));
 open my $lock,'>>',$path or die "open lock $path: $!\n";
 if(!flock($lock,LOCK_EX|LOCK_NB)) {
  $!{EWOULDBLOCK}||$!{EAGAIN}||$!{EINTR} or die "flock $path: $!\n";
  die "BLOCKED_BUSY: shared lock held: $path\n" if $a->{wait_seconds}<=0;
  print STDERR "Queued for shared GPU lock: $path (timeout $a->{wait_seconds}s)\n";
  # Block in flock rather than repeatedly racing other queued validators.
  # SIGALRM bounds this one wait; no GPU work starts before lock ownership.
  local $SIG{ALRM}=sub {die "BLOCKED_BUSY: shared lock held: $path\n";};
  my $error;
  eval {
   alarm($a->{wait_seconds});
   while(!flock($lock,LOCK_EX)) {next if $!{EINTR};die "flock $path: $!\n";}
   alarm(0);1;
  } or $error=$@||"lock wait failed\n";
  alarm(0);
  die $error if defined($error);
 }
 return $lock;
}
sub inherited_host_lock {
 my($fd,$path)=@_;
 $fd=~/\A\d+\z/ && $fd>=3 or die "invalid inherited host-lock descriptor\n";
 open my $lock,'>>&',$fd or die "duplicate inherited host-lock descriptor: $!\n";
 my @actual=stat($lock);my @expected=stat($path);
 @actual && @expected && $actual[0]==$expected[0] && $actual[1]==$expected[1]
  or die "inherited host-lock descriptor names a different file\n";
 open my $info,'<',"/proc/self/fdinfo/$fd" or die "read inherited lock metadata: $!\n";
 my $owned=grep {/^lock:\s+\d+:\s+FLOCK\s+ADVISORY\s+WRITE\s/} <$info>;
 close $info;
 $owned or die "inherited host-lock descriptor has no exclusive flock\n";
 # Duplicate the same open file description, not a second open of the path.
 # This confirms the caller's lease without recursively waiting on itself.
 flock($lock,LOCK_EX|LOCK_NB) or die "BLOCKED_BUSY: inherited host-lock verification failed\n";
 return $lock;
}
sub median {
 my @v=sort {$a<=>$b}@_;@v or die "no values for median\n";
 my $i=int(@v/2);return @v%2?$v[$i]:($v[$i-1]+$v[$i])/2;
}
sub validate_preflight {
 my($r)=@_;
 ref($r) eq 'HASH' && defined($r->{exit}) && $r->{exit}==0
  or die "eager driver-ABI self-test failed before GPU reservation\n";
 my $v=eval {JSON::PP->new->decode($r->{stdout}//'')};
 ref($v) eq 'HASH' && ($v->{self_test}//'') eq 'pass' && $v->{cpu_only}
  && ($v->{negative_canary_cases}//0)==16 && ($v->{negative_trace_cases}//0)==14
  && ($v->{mapping_cases}//0)==64
  or die "incomplete CPU-only driver-ABI self-test evidence\n";
 return $v;
}
sub validate {
 my($v,$trace)=@_;ref($v) eq 'HASH' or die "harness result is not an object\n";
 $v->{bit_exact} && $v->{boundary_canaries}
  && defined($v->{boundary_canary_words}) && $v->{boundary_canary_words}==64
  && defined($v->{partial_values}) && $v->{partial_values}==49536
  && defined($v->{final_values}) && $v->{final_values}==6144
  && defined($v->{steps_per_sample}) && $v->{steps_per_sample}==64
  or die "missing exact comparison, boundary canaries, or timing coverage\n";
 for my $arm(qw(old new)) {
  my $m=$v->{$arm}{event}{median_us};
  defined($m) && !ref($m) && $m=~/\A(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\z/ && $m>0
   or die "missing positive $arm event median\n";
 }
 defined($v->{trace_verified}) && defined($v->{trace_launches}) or die "missing trace mode metadata\n";
 if($trace) {
  $v->{trace_verified} && $v->{trace_launches}>0
   or die "trace requested but harness did not verify fresh physical-CTA traces\n";
 } else {
  !$v->{trace_verified} && $v->{trace_launches}==0
   or die "untraced run unexpectedly contains trace instrumentation\n";
 }
 return $v;
}
sub self_test {
 my $ok=0;
 my $expect=sub {
  my($name,$test,$want)=@_;my $error;
  eval {$test->();1} or $error=$@||'unknown failure';
  if($want) {defined($error)&&$error=~$want or die "self-test $name: expected rejection\n";}
  else {die "self-test $name: $error" if defined($error);}
  print 'ok ',++$ok," - $name\n";
 };
 my $clean=sub {return {compute_apps=>{exit=>0,stdout=>''},
  ps=>{exit=>0,stdout=>"1 0 S init /sbin/init\n"},
  gpu=>{exit=>0,stdout=>"uuid, name\nGPU-01234567-89ab-cdef-0123-456789abcdef, Tesla M60\n"}};};
 $expect->('idle inventory accepted',sub {@{busy_reasons($clean->())}==0 or die 'busy';});
 $expect->('failed smi query rejects',sub {my $s=$clean->();$s->{compute_apps}{exit}=9;busy_reasons($s);},qr/query failed/);
 $expect->('failed ps query rejects',sub {my $s=$clean->();$s->{ps}{exit}=1;busy_reasons($s);},qr/query failed/);
 $expect->('malformed smi query rejects',sub {my $s=$clean->();$s->{compute_apps}{stdout}="unexpected\n";busy_reasons($s);},qr/unrecognized/);
 $expect->('resident hold process blocks',sub {my $s=$clean->();$s->{ps}{stdout}.="44 1 S perl perl /tmp/hold.pl\n";@{busy_reasons($s)}==1 or die 'missed hold';});
 $expect->('runner harness argument does not block itself',sub {my $s=$clean->();$s->{ps}{stdout}.="45 1 S perl perl kern/cta_remap_matrix.pl --harness ./attnpartn_regress\n";@{busy_reasons($s)}==0 or die 'matched runner argument';});
 $expect->('busy host waits then releases',sub {
  my $n=0;wait_idle({gpu=>'mock',wait_seconds=>1,poll_seconds=>0.001},undef,'mock',sub {
   my $s=$clean->();$s->{compute_apps}{stdout}="GPU-aaaa, 123, flashcore, 100\n" unless $n++;return $s;
  });$n==2 or die 'did not recheck';
 });
 $expect->('busy host ends BLOCKED_BUSY',sub {
  wait_idle({gpu=>'mock',wait_seconds=>0,poll_seconds=>0.001},undef,'mock',sub {
   my $s=$clean->();$s->{compute_apps}{stdout}="GPU-aaaa, 123, flashcore, 100\n";return $s;
  });
 },qr/BLOCKED_BUSY/);
 my $target='GPU-00000000-0000-0000-0000-000000000007';
 my $other='GPU-00000000-0000-0000-0000-000000000000';
 my $policy={gpu=>$target,idle_resident_queue=>'http://127.0.0.1:18090/queue'};
 my $idle=sub {
  my $s=$clean->();
  $s->{gpu}{stdout}="uuid, name\n$target, Tesla M60\n";
  $s->{all_gpu_load}={exit=>0,stdout=>"$target, 0, 0\n$other, 0, 0\n"};
  $s->{queue}={exit=>0,stdout=>'{"running":0,"waiting":0,"max":8}'};
  $s->{compute_apps}{stdout}="$other, 44, ./hold, 7000\n$other, 45, ./flashcore, 200\n$other, 46, ./fence, 100\n";
  $s->{ps}{stdout}.="44 1 S hold ./hold weights 0\n45 1 S flashcore ./flashcore ple library manifest\n46 1 S fence ./fence 0\n47 1 S flashd ./flashd weights /tmp/flash.sock 127.0.0.1:18090\n";
  return $s;
 };
 $expect->('explicit idle off-device residents accepted',sub {my $s=$idle->();@{busy_reasons($s,$policy)}==0 && @{$s->{allowed_idle_residents}}==3 or die 'blocked idle residents';});
 $expect->('default policy still blocks resident contexts',sub {@{busy_reasons($idle->())}>0 or die 'default policy relaxed';});
 $expect->('target compute context still blocks',sub {my $s=$idle->();$s->{compute_apps}{stdout}.="$target, 45, ./flashcore, 200\n";@{busy_reasons($s,$policy)}>0 or die 'target context allowed';});
 $expect->('unrecognized off-device compute worker blocks',sub {my $s=$idle->();$s->{compute_apps}{stdout}.="$other, 48, ./gemv_test, 100\n";@{busy_reasons($s,$policy)}>0 or die 'unknown worker allowed';});
 $expect->('off-device compute activity blocks',sub {my $s=$idle->();$s->{all_gpu_load}{stdout}=~s/\Q$other\E, 0, 0/$other, 1, 0/;@{busy_reasons($s,$policy)}>0 or die 'compute activity allowed';});
 $expect->('off-device memory activity blocks',sub {my $s=$idle->();$s->{all_gpu_load}{stdout}=~s/\Q$other\E, 0, 0/$other, 0, 1/;@{busy_reasons($s,$policy)}>0 or die 'memory activity allowed';});
 for my $field(qw(running waiting)) {
  $expect->("queue $field blocks",sub {my $s=$idle->();$s->{queue}{stdout}=~s/"$field":0/"$field":1/;@{busy_reasons($s,$policy)}>0 or die 'nonempty queue allowed';});
 }
 for my $field(qw(queue all_gpu_load)) {
  $expect->("failed $field query rejects",sub {my $s=$idle->();$s->{$field}{exit}=1;busy_reasons($s,$policy);},qr/query failed/);
 }
 $expect->('malformed queue rejects',sub {my $s=$idle->();$s->{queue}{stdout}='not JSON';busy_reasons($s,$policy);},qr/invalid queue JSON/);
 $expect->('fractional queue count rejects',sub {my $s=$idle->();$s->{queue}{stdout}='{"running":0.5,"waiting":0}';busy_reasons($s,$policy);},qr/invalid queue running/);
 $expect->('unknown utilization rejects',sub {my $s=$idle->();$s->{all_gpu_load}{stdout}=~s/, 0, 0/, N\/A, 0/;busy_reasons($s,$policy);},qr/invalid all-GPU/);
 $expect->('missing target utilization rejects',sub {my $s=$idle->();$s->{all_gpu_load}{stdout}="$other, 0, 0\n";busy_reasons($s,$policy);},qr/target GPU missing/);
 $expect->('duplicate utilization rejects',sub {my $s=$idle->();$s->{all_gpu_load}{stdout}.="$target, 0, 0\n";busy_reasons($s,$policy);},qr/duplicate GPU/);
 $expect->('unknown compute-app GPU rejects',sub {my $s=$idle->();$s->{compute_apps}{stdout}.="GPU-abcd, 44, ./hold, 100\n";busy_reasons($s,$policy);},qr/compute-app GPU missing/);
 $expect->('resident identity mismatch rejects',sub {my $s=$idle->();$s->{ps}{stdout}=~s/44 1 S hold/44 1 S evil/;busy_reasons($s,$policy);},qr/identity mismatch/);
 $expect->('resident disappearing from ps rejects',sub {my $s=$idle->();$s->{ps}{stdout}=~s/^44 .*\n//m;busy_reasons($s,$policy);},qr/missing from ps/);
 $expect->('different daemon queue still blocks',sub {my $s=$idle->();$s->{ps}{stdout}=~s/:18090/:18091/;@{busy_reasons($s,$policy)}>0 or die 'other daemon allowed';});
 $expect->('worker without context still blocks',sub {my $s=$idle->();$s->{ps}{stdout}.="48 1 S model_test ./model_test weights\n";@{busy_reasons($s,$policy)}>0 or die 'worker before CUDA allowed';});
 for my $flag('--n-gpu-layers 0','-ngl 0','--gpu-layers=0') {
  $expect->("CPU-configured llama accepted: $flag",sub {
   my $s=$idle->();$s->{ps}{stdout}.="48 1 S llama-server /tmp/llama-server --model weights $flag\n";
   @{busy_reasons($s,$policy)}==0 && @{$s->{cpu_configured_workers}}==1 or die 'zero-offload process blocked';
  });
 }
 for my $flags('--n-gpu-layers 1','--n-gpu-layers 0 -ngl 5','--n-gpu-layers auto','--model weights') {
  $expect->("ambiguous/GPU llama still blocks: $flags",sub {
   my $s=$idle->();$s->{ps}{stdout}.="48 1 S llama-server /tmp/llama-server $flags\n";
   @{busy_reasons($s,$policy)}>0 or die 'nonzero or unspecified offload accepted';
  });
 }
 $expect->('CPU flag cannot hide a reported GPU context',sub {
  my $s=$idle->();$s->{ps}{stdout}.="48 1 S llama-server /tmp/llama-server -ngl 0\n";
  $s->{compute_apps}{stdout}.="$other, 48, /tmp/llama-server, 100\n";
  @{busy_reasons($s,$policy)}>0 && @{$s->{cpu_configured_workers}}==0 or die 'CPU flag hid compute app';
 });
 $expect->('strict policy still blocks CPU-configured llama',sub {
  my $s=$clean->();$s->{ps}{stdout}.="48 1 S llama-server /tmp/llama-server -ngl 0\n";
  @{busy_reasons($s)}>0 or die 'strict policy unexpectedly relaxed';
 });
 my $recent_target=sub {my $s=$idle->();$s->{all_gpu_load}{stdout}=~s/\Q$target\E, 0, 0/$target, 70, 30/;return $s;};
 $expect->('initial target utilization still blocks',sub {@{busy_reasons($recent_target->(),$policy)}>0 or die 'initial target load ignored';});
 $expect->('owned target utilization settles without false rejection',sub {
  my $n=0;my $r=post_case_idle({%$policy,settle_seconds=>1},undef,'mock',sub {$n++?$idle->():$recent_target->()});
  @{$r->{snapshots}}==2 && @{$r->{snapshots}[0]{busy_reasons}}>0 && @{$r->{snapshots}[1]{busy_reasons}}==0
   or die 'immediate/settled telemetry not retained';
 });
 $expect->('target utilization settling is bounded',sub {
  post_case_idle({%$policy,settle_seconds=>0},undef,'mock',$recent_target);
 },qr/target utilization did not settle/);
 $expect->('target context still rejects during settling',sub {
  post_case_idle($policy,undef,'mock',sub {my $s=$recent_target->();$s->{compute_apps}{stdout}.="$target, 45, ./flashcore, 200\n";return $s;});
 },qr/post-run contention/);
 $expect->('off-device activity still rejects during settling',sub {
  post_case_idle($policy,undef,'mock',sub {my $s=$recent_target->();$s->{all_gpu_load}{stdout}=~s/\Q$other\E, 0, 0/$other, 1, 0/;return $s;});
 },qr/post-run contention/);
 $expect->('new request rejects while target settles',sub {
  my $n=0;post_case_idle({%$policy,settle_seconds=>1},undef,'mock',sub {my $s=$recent_target->();$s->{queue}{stdout}='{"running":1,"waiting":0}' if $n++;return $s;});
 },qr/post-run contention/);
 $expect->('failed settling query fails closed',sub {
  post_case_idle($policy,undef,'mock',sub {my $s=$recent_target->();$s->{all_gpu_load}{exit}=1;return $s;});
 },qr/query failed/);
 require File::Temp;
 my $locks=File::Temp::tempdir(CLEANUP=>1);
 for my $release(0,1) {
  $expect->($release?'queued lock acquires after release':'queued lock times out without stealing',sub {
   my $path="$locks/test-$release.lock";
   open my $held,'>>',$path or die "open fixture lock: $!";
   flock($held,LOCK_EX) or die "fixture flock: $!";
   my $pid=fork();defined($pid) or die "fork fixture: $!";
   if(!$pid) {
    close $held;
    my $got=eval {acquire_lock($path,{wait_seconds=>$release?2:0.05})};
    my $error=$@;
    POSIX::_exit($release ? ($got?0:1) : (!$got && $error=~/BLOCKED_BUSY/?0:1));
   }
   if($release){sleep(0.05);close $held;}
   waitpid($pid,0)==$pid && $?==0 or die 'queued lock fixture failed';
   close $held unless $release;
  });
 }
 $expect->('inherited reservation is shared without nested locking',sub {
  my $path="$locks/inherited.lock";open my $held,'>>',$path or die $!;
  flock($held,LOCK_EX) or die $!;
  my $borrowed=inherited_host_lock(fileno($held),$path);close $borrowed;
  open my $other,'>>',$path or die $!;
  !flock($other,LOCK_EX|LOCK_NB) or die 'closing borrowed descriptor released caller lock';
  close $other;close $held;
 });
 $expect->('different inherited file rejects',sub {
  my $path="$locks/wrong.lock";open my $held,'>>',$path or die $!;flock($held,LOCK_EX) or die $!;
  inherited_host_lock(fileno($held),"$locks/inherited.lock");
 },qr/different file/);
 $expect->('unlocked inherited descriptor rejects',sub {
  my $path="$locks/unlocked.lock";open my $held,'>>',$path or die $!;
  inherited_host_lock(fileno($held),$path);
 },qr/no exclusive flock/);
 $expect->('independent descriptor cannot impersonate caller lock',sub {
  my $path="$locks/independent.lock";open my $held,'>>',$path or die $!;flock($held,LOCK_EX) or die $!;
  open my $other,'>>',$path or die $!;
  inherited_host_lock(fileno($other),$path);
 },qr/no exclusive flock/);
 my $preflight=sub {return {exit=>0,stdout=>'{"self_test":"pass","cpu_only":true,"negative_canary_cases":16,"negative_trace_cases":14,"mapping_cases":64}'};};
 $expect->('eager CPU-only preflight accepted',sub {validate_preflight($preflight->());});
 $expect->('driver ABI loader failure rejects before lock',sub {my $r=$preflight->();$r->{exit}=127;validate_preflight($r);},qr/driver-ABI self-test failed/);
 $expect->('invalid eager preflight JSON rejects',sub {my $r=$preflight->();$r->{stdout}='';validate_preflight($r);},qr/incomplete CPU-only/);
 $expect->('non-CPU-only preflight rejects',sub {my $r=$preflight->();$r->{stdout}=~s/true/false/;validate_preflight($r);},qr/incomplete CPU-only/);
 my $valid=sub {return {bit_exact=>JSON::PP::true,boundary_canaries=>JSON::PP::true,boundary_canary_words=>64,
  partial_values=>49536,final_values=>6144,steps_per_sample=>64,
  old=>{event=>{median_us=>1}},new=>{event=>{median_us=>2}},trace_verified=>JSON::PP::false,trace_launches=>0};};
 $expect->('untraced complete result accepted',sub {validate($valid->(),0);});
 $expect->('missing canaries rejects',sub {my $v=$valid->();delete $v->{boundary_canaries};validate($v,0);},qr/boundary canaries/);
 $expect->('missing requested trace rejects',sub {validate($valid->(),1);},qr/trace requested/);
 $expect->('unexpected traced timing rejects',sub {my $v=$valid->();$v->{trace_verified}=1;$v->{trace_launches}=1;validate($v,0);},qr/untraced/);
 $expect->('verified traced result accepted',sub {my $v=$valid->();$v->{trace_verified}=1;$v->{trace_launches}=1;validate($v,1);});
 $expect->('child stdout and failure retained',sub {
  my($o,$e,$rc)=run([$^X,'-e','print "out"; print STDERR "err"; exit 7'],\%ENV,2);
  $o eq 'out' && $e eq 'err' && $rc==7 or die 'capture/exit';
 });
 $expect->('only own child is timed out',sub {
  my(undef,$e,$rc)=run([$^X,'-e','select undef,undef,undef,5'],\%ENV,0.05);
  $rc==124 && $e=~/RUNNER TIMEOUT/ or die 'timeout not reported';
 });
 print JSON::PP->new->canonical->encode({status=>'PASS',tests=>$ok,gpu_executed=>JSON::PP::false}),"\n";
 return 0;
}
sub main {
 my %a=(old_source=>'c6d3de8180d8b87caf0d2f61fa52edc9247daded',new_source=>'66725432f86cc193b75cd51230216aed44c60d3f',
  harness=>'./attnpartn_regress',keys=>'1,1023,1024,1025,2047,2048,2049,2050,2051,2052,4095,4096,4097,8191,8192,8193',
  seeds=>'0,77,4242',repeats=>32,instrumentation=>'none',trace=>0,
  lock_dir=>$ENV{FLASH_GPU_LOCK_DIR}//'/tmp/flash-sass-gpu-locks',
  host_lock=>$ENV{FLASH_GPU_HOST_LOCK}//'/tmp/flash-sass-engine-validation.lock',wait_seconds=>60,poll_seconds=>5,settle_seconds=>5);
 GetOptions('old=s'=>\$a{old},'new=s'=>\$a{new},'old-library=s'=>\$a{old_library},'new-library=s'=>\$a{new_library},
  'old-source=s'=>\$a{old_source},'new-source=s'=>\$a{new_source},'harness=s'=>\$a{harness},'harness-source=s'=>\$a{harness_source},
  'gpu=s'=>\$a{gpu},'out=s'=>\$a{out},'keys=s'=>\$a{keys},'seeds=s'=>\$a{seeds},'repeats=i'=>\$a{repeats},
  'instrumentation=s'=>\$a{instrumentation},'trace!'=>\$a{trace},'lock-dir=s'=>\$a{lock_dir},'host-lock=s'=>\$a{host_lock},
  'idle-resident-queue=s'=>\$a{idle_resident_queue},
  'settle-seconds=f'=>\$a{settle_seconds},
  'host-lock-fd=i'=>\$a{host_lock_fd},
  'wait-seconds=f'=>\$a{wait_seconds},'poll-seconds=f'=>\$a{poll_seconds},'self-test'=>\$a{self_test},'check-only'=>\$a{check_only})
  or die "invalid arguments\n";
 !@ARGV or die "unexpected positional arguments\n";
 return self_test() if $a{self_test};
 for my $key(qw(gpu out)) {defined($a{$key}) or die "--$key is required\n";}
 if(!$a{check_only}) {
  for my $key(qw(old new old_library new_library harness_source)) {defined($a{$key}) or die "--$key is required\n";}
  for(qw(old_source new_source harness_source)) {$a{$_}=~/\A[0-9a-f]{40}\z/ or die "source identities must be full 40-character git commit IDs\n";}
 }
 $a{gpu}=~/\A(?:GPU-[0-9a-f-]+|\d+)\z/i or die "--gpu must be a GPU UUID or numeric index\n";
 if(defined($a{idle_resident_queue})) {
  $a{idle_resident_queue}=~m{\Ahttp://127\.0\.0\.1:([1-9]\d{0,4})/queue\z} && $1<=65535
   or die "--idle-resident-queue must name a loopback HTTP queue\n";
  $a{gpu}=~/\AGPU-/i or die "idle resident mode requires a GPU UUID, not an index\n";
 }
 $a{wait_seconds}>=0 && $a{poll_seconds}>0 && $a{poll_seconds}<=60 or die "wait-seconds >=0 and poll-seconds in (0,60] required\n";
 $a{settle_seconds}>=0 && $a{settle_seconds}<=30 or die "settle-seconds in [0,30] required\n";
 !defined($a{host_lock_fd}) || $a{host_lock_fd}>=3 or die "host-lock-fd must be >=3\n";
 ($a{trace}?$a{instrumentation} ne 'none':$a{instrumentation} eq 'none')
  or die "--trace requires --instrumentation; untraced timing requires instrumentation=none\n";
 my @counts=split /,/,$a{keys},-1;my @seeds=split /,/,$a{seeds},-1;
 for my $list(\@counts,\@seeds) {
  my %seen;@$list or die "keys and seeds must be nonempty\n";
  for(@$list) {/\A\d+\z/ or die "invalid unsigned integer list\n";$_=0+$_;$seen{$_}++ and die "duplicate key/seed would overwrite evidence\n";}
 }
 for(@counts) {$_>=1&&$_<=16384 or die "keys 1..16384 required\n";}
 for(@seeds) {$_<=4294967295 or die "32-bit unsigned seed required\n";}
 $a{repeats}>=4&&$a{repeats}<=10000&&$a{repeats}%2==0 or die "even repetition count in 4..10000 required\n";
 my $out=new_output($a{out});my %env=(%ENV,CUDA_DEVICE_ORDER=>'PCI_BUS_ID',CUDA_VISIBLE_DEVICES=>$a{gpu});
 delete $env{ATTN_TRACE_CTAS};
 if($a{trace}) {$env{ATTN_TRACE_CHECK}='1';} else {delete $env{ATTN_TRACE_CHECK};}
 my(@rows,@locks);my($rc,$launches)=(0,0);
 eval {
  my %traceenv=map {$_=>$env{$_}} grep {/\A(?:CUDA_|ATTN_|DIE_|FLASH_GPU_|LD_LIBRARY_PATH\z|LD_PRELOAD\z|LD_BIND_NOW\z)/} keys %env;
  my $manifest={arguments=>\%a,utc=>utc(),runner=>image($0),helper=>image("$FindBin::Bin/QsaEvidence.pm"),
   environment=>\%traceenv,trace_requested=>$a{trace}?JSON::PP::true:JSON::PP::false,
   measurement_mode=>$a{check_only}?'check-only':$a{trace}?'physical-cta-trace':'untraced-performance',
   locking=>{host=>$a{host_lock},gpu_dir=>$a{lock_dir},inherited_host_fd=>$a{host_lock_fd}}};
  $manifest->{resident_policy}=$a{idle_resident_queue}
   ? 'Explicit off-device idle residents: target has no reported compute app; all-device utilization and loopback queue counts must be zero before each case and after bounded settling of the owned target. Immediate post-run telemetry is retained; target contexts, other-device activity, and queued requests always reject. This does not authorize whole-model core reuse.'
   : 'Strict: any compute context or recognized GPU worker on the host blocks.';
  if(!$a{check_only}) {
   $manifest->{old_cubin}=image($a{old});$manifest->{new_cubin}=image($a{new});
   $manifest->{old_library}=image($a{old_library});$manifest->{new_library}=image($a{new_library});
   $manifest->{harness}=image($a{harness});$manifest->{compiler}=result(['cc','--version']);
   $manifest->{ptxas}=result([$ENV{PTXAS11}//'/opt/cuda-11.4/bin/ptxas','--version']);
   $manifest->{timing}='64 positions per persistent launch, four warmup pairs, balanced alternating old/new order; one-phase partial and two-phase partial+merge programs; GPU events exclude allocation/upload and CPU checks';
   $manifest->{input}='finite fixed-seed FP32 values; seed0 has zero query; padding finite zero';
  }
  put("$out/manifest.json",$manifest);
  if(!$a{check_only}) {
   # The harness returns before cuInit in this mode. Eager binding catches
   # missing legacy-driver symbols before waiting for the shared GPU lease.
   my %preflight_env=(%env,LD_BIND_NOW=>'1');
   my $preflight=result([absolute($a{harness}),'--self-test'],\%preflight_env,30);
   $preflight->{LD_BIND_NOW}=1;
   put("$out/driver-abi-preflight.json",$preflight);
   validate_preflight($preflight);
  }
  put("$out/status.json",{status=>'WAITING',cases=>0,gpu_executed=>JSON::PP::false});
  push @locks,defined($a{host_lock_fd})
   ? inherited_host_lock($a{host_lock_fd},$a{host_lock}) : acquire_lock($a{host_lock},\%a);
  push @locks,acquire_lock("$a{lock_dir}/$a{gpu}.lock",\%a);
  if($a{check_only}) {
   wait_idle(\%a,$out,'check-only');
   put("$out/status.json",{status=>'IDLE',check_only=>JSON::PP::true,gpu_executed=>JSON::PP::false});
  } else {
   for my $n(@counts) {for my $seed(@seeds) {
    my $case="keys-$n-seed-$seed";my $before=wait_idle(\%a,$out,$case);
    my @args=(absolute($a{harness}),absolute($a{old}),absolute($a{new}),"$n","$seed","$a{repeats}");
    ++$launches;my $r=result(\@args,\%env,180);
    $r->{gpu_before}=$before;$r->{gpu_after}=gpu($a{gpu});
    my $after_error;
    if($a{idle_resident_queue}) {
     eval {
      $r->{host_after}=post_case_idle(\%a,$out,$case);
      1;
     } or $after_error=$@||"post-run idle check failed\n";
     $r->{post_run_error}=$after_error if defined($after_error);
    }
    put("$out/$case.json",$r);
    $r->{exit}==0 or die "GPU comparison failed: n=$n, seed=$seed, exit=$r->{exit}\n";
    die $after_error if defined($after_error);
    $r->{gpu_after}{exit}==0 or die "post-run GPU metadata query failed\n";
    my $v=validate(JSON::PP->new->decode($r->{stdout}),$a{trace});
    defined($v->{keys})&&$v->{keys}==$n or die "harness reported wrong key count\n";
    defined($v->{seed})&&$v->{seed}==$seed or die "harness reported wrong seed\n";
    push @rows,$v;
    print JSON::PP->new->canonical->encode({keys=>$n,seed=>$seed,exact=>JSON::PP::true,
     boundary_canaries=>JSON::PP::true,trace_verified=>$v->{trace_verified},
     old_us=>$v->{old}{event}{median_us},new_us=>$v->{new}{event}{median_us}}),"\n";
   }}
   my @summary;
   for my $n(@counts) {
    my @cases=grep {$_->{keys}==$n}@rows;
    my $old=median(map {$_->{old}{event}{median_us}}@cases);my $new=median(map {$_->{new}{event}{median_us}}@cases);
    push @summary,{keys=>$n,seeds=>scalar(@cases),old_partial_merge_us=>$old,new_partial_merge_us=>$new,delta_us=>$new-$old,speedup=>$old/$new};
   }
   put("$out/summary.json",\@summary);
   put("$out/status.json",{status=>'PASS',cases=>scalar(@rows),gpu_launches=>$launches,trace_requested=>$a{trace}?JSON::PP::true:JSON::PP::false});
  }
 };
 if($@) {
  my $error=$@;my $busy=$error=~/^BLOCKED_BUSY:/;
  put("$out/status.json",{status=>$busy?'BLOCKED_BUSY':'FAIL',completed=>scalar(@rows),gpu_launches=>$launches,
   error=>$error,gpu_executed=>$launches?JSON::PP::true:JSON::PP::false});
  print STDERR $error;$rc=$busy?75:1;
 }
 close $_ for reverse @locks;
 return $rc;
}
my $rc=eval {main()};if($@) {print STDERR $@;exit 1;}exit $rc;
