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
use Time::HiRes qw(sleep clock_gettime CLOCK_MONOTONIC);
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
 my($uuid)=@_;
 # ALL devices are checked. An idle selected die does not authorize overlap
 # with a different worker's experiment on this shared host.
 return {utc=>utc(),
  compute_apps=>result(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],\%ENV,30),
  ps=>result(['ps','-eo','pid=,ppid=,stat=,comm=,args='],\%ENV,30),gpu=>gpu($uuid)};
}
sub busy_reasons {
 my($s)=@_;
 for my $name(qw(compute_apps ps gpu)) {
  ref($s->{$name}) eq 'HASH' && defined($s->{$name}{exit}) && $s->{$name}{exit}==0
   or die "$name query failed; GPU idle state is unknown\n";
  defined($s->{$name}{stdout}) or die "$name query has no stdout\n";
 }
 $s->{ps}{stdout}=~/\S/ or die "ps returned no process inventory\n";
 $s->{gpu}{stdout}=~/GPU-[0-9a-f-]+/i or die "GPU metadata query returned no UUID\n";
 my @busy;
 for my $line(split /\n/,$s->{compute_apps}{stdout}) {
  next unless $line=~/\S/;
  $line=~/^\s*GPU-[0-9a-f-]+\s*,\s*\d+\s*,/i or die "unrecognized nvidia-smi compute-app output\n";
  push @busy,"compute app: $line";
 }
 for my $line(split /\n/,$s->{ps}{stdout}) {
  next unless $line=~/\S/;
  my($pid,$ppid,$stat,$comm,$args)=$line=~/^\s*(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s*(.*)$/;
  defined($pid) or die "unrecognized ps process row\n";
  # Conservatively catch known workers before their CUDA context is visible.
  # Inspect only the executable/script, not arguments of this runner or a shell
  # that merely mentions another program. Persist process identity, not argv.
  if($comm=~/^(?:flashcore|flashd|hold|fence|attnpartn_regre|qsa_oracle|qsa_test|model_test|attn_test|die_test|memtest|gemv.*test|llama-server|llama-cli|llama-perplexit)/
   || $args=~m{^(?:(?:\S*/)?(?:perl|ruby|bash|sh)(?:\s+-[^ \t]+)*\s+)?(?:\S*/)?(?:flashcore[^\s/]*|flashd|hold(?:[-_.][^\s/]*)?|fence(?:[-_.][^\s/]*)?|attnpartn_regress|attnpartn_model_compare|qsa_test|model_test|attn_test|die_test|memtest|gemv[^\s/]*test|llama-server|llama-cli|llama-perplexity)(?:\s|$)}) {
   push @busy,"worker process: pid=$pid ppid=$ppid stat=$stat comm=$comm";
  }
 }
 return \@busy;
}
sub wait_idle {
 my($a,$out,$case,$collect)=@_;$collect//=\&snapshot;
 my $start=clock_gettime(CLOCK_MONOTONIC);my $attempt=0;
 while(1) {
  my $s=$collect->($a->{gpu});$s->{case}=$case;$s->{attempt}=++$attempt;
  my $busy=busy_reasons($s);
  # Full argv is needed only to recognize interpreter-launched GPU workers.
  # Do not save unrelated workers' command arguments in public evidence.
  $s->{ps}{stdout}=~s/^(\s*\d+\s+\d+\s+\S+\s+\S+).*$/$1/mg;
  $s->{ps}{retained_fields}='pid ppid stat comm; arguments inspected but not retained';
  $s->{busy_reasons}=$busy;
  put("$out/gate-$case-$attempt.json",$s) if defined($out);
  return $s unless @$busy;
  print STDERR "Waiting for idle GPU host: ",join('; ',@$busy),"\n";
  my $left=$a->{wait_seconds}-(clock_gettime(CLOCK_MONOTONIC)-$start);
  die "BLOCKED_BUSY: host still has other GPU workers\n" if $left<=0;
  sleep($left<$a->{poll_seconds}?$left:$a->{poll_seconds});
 }
}
sub acquire_lock {
 my($path,$a)=@_;make_path(dirname($path));
 open my $lock,'>>',$path or die "open lock $path: $!\n";
 my $start=clock_gettime(CLOCK_MONOTONIC);
 while(!flock($lock,LOCK_EX|LOCK_NB)) {
  $!{EWOULDBLOCK}||$!{EAGAIN}||$!{EINTR} or die "flock $path: $!\n";
  my $left=$a->{wait_seconds}-(clock_gettime(CLOCK_MONOTONIC)-$start);
  die "BLOCKED_BUSY: shared lock held: $path\n" if $left<=0;
  print STDERR "Waiting for shared GPU lock: $path\n";
  sleep($left<$a->{poll_seconds}?$left:$a->{poll_seconds});
 }
 return $lock;
}
sub median {
 my @v=sort {$a<=>$b}@_;@v or die "no values for median\n";
 my $i=int(@v/2);return @v%2?$v[$i]:($v[$i-1]+$v[$i])/2;
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
  host_lock=>$ENV{FLASH_GPU_HOST_LOCK}//'/tmp/flash-sass-engine-validation.lock',wait_seconds=>60,poll_seconds=>5);
 GetOptions('old=s'=>\$a{old},'new=s'=>\$a{new},'old-library=s'=>\$a{old_library},'new-library=s'=>\$a{new_library},
  'old-source=s'=>\$a{old_source},'new-source=s'=>\$a{new_source},'harness=s'=>\$a{harness},'harness-source=s'=>\$a{harness_source},
  'gpu=s'=>\$a{gpu},'out=s'=>\$a{out},'keys=s'=>\$a{keys},'seeds=s'=>\$a{seeds},'repeats=i'=>\$a{repeats},
  'instrumentation=s'=>\$a{instrumentation},'trace!'=>\$a{trace},'lock-dir=s'=>\$a{lock_dir},'host-lock=s'=>\$a{host_lock},
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
 $a{wait_seconds}>=0 && $a{poll_seconds}>0 && $a{poll_seconds}<=60 or die "wait-seconds >=0 and poll-seconds in (0,60] required\n";
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
  my %traceenv=map {$_=>$env{$_}} grep {/\A(?:CUDA_|ATTN_|DIE_|FLASH_GPU_)/} keys %env;
  my $manifest={arguments=>\%a,utc=>utc(),runner=>image($0),helper=>image("$FindBin::Bin/QsaEvidence.pm"),
   environment=>\%traceenv,trace_requested=>$a{trace}?JSON::PP::true:JSON::PP::false,
   measurement_mode=>$a{check_only}?'check-only':$a{trace}?'physical-cta-trace':'untraced-performance',
   locking=>{host=>$a{host_lock},gpu_dir=>$a{lock_dir}}};
  if(!$a{check_only}) {
   $manifest->{old_cubin}=image($a{old});$manifest->{new_cubin}=image($a{new});
   $manifest->{old_library}=image($a{old_library});$manifest->{new_library}=image($a{new_library});
   $manifest->{harness}=image($a{harness});$manifest->{compiler}=result(['cc','--version']);
   $manifest->{ptxas}=result([$ENV{PTXAS11}//'/opt/cuda-11.4/bin/ptxas','--version']);
   $manifest->{timing}='64 positions per persistent launch, four warmup pairs, balanced alternating old/new order; one-phase partial and two-phase partial+merge programs; GPU events exclude allocation/upload and CPU checks';
   $manifest->{input}='finite fixed-seed FP32 values; seed0 has zero query; padding finite zero';
  }
  put("$out/manifest.json",$manifest);
  put("$out/status.json",{status=>'WAITING',cases=>0,gpu_executed=>JSON::PP::false});
  push @locks,acquire_lock($a{host_lock},\%a);
  push @locks,acquire_lock("$a{lock_dir}/$a{gpu}.lock",\%a);
  if($a{check_only}) {
   wait_idle(\%a,$out,'check-only');
   put("$out/status.json",{status=>'IDLE',check_only=>JSON::PP::true,gpu_executed=>JSON::PP::false});
  } else {
   for my $n(@counts) {for my $seed(@seeds) {
    my $case="keys-$n-seed-$seed";my $before=wait_idle(\%a,$out,$case);
    my @args=(absolute($a{harness}),absolute($a{old}),absolute($a{new}),"$n","$seed","$a{repeats}");
    ++$launches;my $r=result(\@args,\%env,180);
    $r->{gpu_before}=$before;$r->{gpu_after}=gpu($a{gpu});put("$out/$case.json",$r);
    $r->{exit}==0 or die "GPU comparison failed: n=$n, seed=$seed, exit=$r->{exit}\n";
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
