#!/usr/bin/env perl
# Cancel an owned core's chunked prefill; compare recovery with a verified scalar run.
use strict;use warnings;use FindBin;use lib $FindBin::Bin;
use QsaEvidence qw(sha put write_text run absolute);
use SocketEvidence qw(connect_unix send_text read_line close_socket monotonic);
use Getopt::Long 'GetOptions';use JSON::PP;use File::Path 'make_path';use File::Temp 'tempdir';
use Sys::Hostname 'hostname';use Fcntl qw(LOCK_SH LOCK_NB);use POSIX qw(setpgid WNOHANG _exit);use Time::HiRes 'sleep';
$|=1;$SIG{PIPE}='IGNORE';$SIG{INT}=sub{die "interrupted\n";};$SIG{TERM}=sub{die "terminated\n";};
sub check{my($ok,$why)=@_;$ok or die "$why\n";}
sub fields{my($line)=@_;my %f;while($line=~/(\w+)=([^\s]+)/g){$f{$1}=$2;}return \%f;}
sub same{my($a,$b)=@_;my $j=JSON::PP->new->canonical;return $j->encode($a) eq $j->encode($b);}
sub load_json{my($p)=@_;open my $f,'<:raw',$p or die "open $p: $!\n";local $/;my $s=<$f>;close $f;return JSON::PP->new->decode($s);}
sub read_command{
 my($client,$command,$terminator,$records)=@_;my $entry={command=>$command,response=>[]};push @$records,$entry;send_text($client,$command."\n");my $deadline=monotonic()+90;
 for(1..10000){my $left=$deadline-monotonic();check($left>0,'response deadline exceeded');$client->{timeout}=$left;my $raw=read_line($client,1024*1024);check(defined($raw) && length($raw)<1024*1024,'EOF or oversized response');$raw=~s/[\r\n]+\z//;push @{$entry->{response}},$raw;check(index($raw,'error:')!=0,$raw);return $entry->{response} if $terminator->($raw);}die "response line limit exceeded\n";
}
sub exchange{my($path,$command,$terminator,$records)=@_;my $c=connect_unix($path,90);my $v=read_command($c,$command,$terminator,$records);close_socket($c);return $v;}
sub state{
 my($lines)=@_;my @summaries=grep {index($_,'statehash positions=')==0}@$lines;check(@summaries==1,'missing or duplicate statehash summary');my $f=fields($summaries[0]);my %r;
 for my $k(keys %$f){$f->{$k}=~/\A[+-]?\d+\z/ or die "invalid statehash integer $k\n";$r{$k}=0+$f->{$k};}check(exists($r{positions}) && exists($r{paired_positions}),'missing counters');my @body=grep {$_ ne $summaries[0] && $_ ne 'statehash done'}@$lines;check(@body>0,'empty statehash body');$r{state}=\@body;return \%r;
}
sub poll_child{
 my($child)=@_;return $child->{exit} if exists($child->{exit});my $r=waitpid($child->{pid},WNOHANG);if($r==$child->{pid}){my $s=$?;$child->{exit}=($s&127)?-($s&127):($s>>8);return $child->{exit};}die "waitpid: $!\n" if $r<0;return undef;
}
sub wait_child{my($child,$seconds)=@_;my $deadline=monotonic()+$seconds;while(1){my $rc=poll_child($child);return $rc if defined($rc);return undef if monotonic()>=$deadline;sleep(.05);}}
sub stop_child{
 my($child,$path)=@_;return unless $child;
 if(!defined(poll_child($child))){my $graceful=eval{my $c=connect_unix($path,5);send_text($c,"quit\n");close_socket($c);my $rc=wait_child($child,5);defined($rc) or die "quit timeout\n";1;};
  if(!$graceful && !defined(poll_child($child))){kill 'TERM',-$child->{pid};kill 'TERM',$child->{pid};if(!defined(wait_child($child,5))){kill 'KILL',-$child->{pid};kill 'KILL',$child->{pid};defined(wait_child($child,5)) or die "owned core did not exit after kill\n";}}
 }
}
sub spawn_core{
 my($argv,$cwd,$env,$stdout,$stderr)=@_;my $pid=fork();defined($pid) or die "fork core: $!\n";
 if(!$pid){eval{setpgid(0,0)==0 or die "setpgid: $!\n";open STDIN,'<','/dev/null' or die "stdin: $!\n";open STDOUT,'>:raw',$stdout or die "stdout $stdout: $!\n";open STDERR,'>:raw',$stderr or die "stderr $stderr: $!\n";chdir($cwd) or die "chdir $cwd: $!\n";%ENV=%$env;exec {$argv->[0]}@$argv;die "exec core: $!\n";};print STDERR $@;_exit(127);}
 setpgid($pid,$pid);return {pid=>$pid};
}
sub main{
 my($reference,$revision,$kv_host)=(undef,undef,'0');GetOptions('reference=s'=>\$reference,'sha=s'=>\$revision,'kv-host=s'=>\$kv_host) or die "invalid options\n";
 @ARGV==2 && defined($reference) && defined($revision) or die "usage: $0 ROOTDIR OUTDIR --reference results.json --sha FULL_SHA [--kv-host 0|1]\n";my($root,$out)=@ARGV;check($kv_host eq '0' || $kv_host eq '1','--kv-host must be 0 or 1');
 check(!-e $out,'output directory already exists');make_path($out);$out=absolute($out);$root=absolute($root);my $binpath=-f "$root/kern/flashcore"?"$root/kern":$root;
 my $report={hostname=>hostname(),head_sha=>$revision,pass=>JSON::PP::false,commands=>[],reference=>absolute($reference)};my($child,$sockpath,$reader);my @locks;my $error='';
 eval{
  check($revision=~/\A[0-9a-f]{40}\z/,'supply a full commit SHA');my $ref=load_json($reference);check(defined($ref->{head_sha}) && $ref->{head_sha} eq $revision && $ref->{pass},'reference revision/pass mismatch');my $scalar=$ref->{runs}{scalar};check(ref($scalar) eq 'HASH','missing scalar reference run');
  my %expected=(rev=>$revision,kv_host=>$kv_host,ple_disk=>'1',maxt=>'128',layers=>'48/48',nblk=>'4',dies=>'6');my $refcfg=fields($scalar->{config});for my $k(keys %expected){check(defined($refcfg->{$k}) && $refcfg->{$k} eq $expected{$k},"reference config mismatch: $k");}
  my $oracle=$scalar->{cases}{p17};check(ref($oracle) eq 'HASH','missing p17 scalar oracle');my %hashes=map {$_=>sha("$binpath/$_")}qw(flashcore libengine.so die.cubin);$report->{sha256}=\%hashes;check(same(\%hashes,$ref->{sha256}),'reference binaries differ');$report->{reference_sha256}=sha($reference);
  my @argv=("$binpath/flashcore",'/mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf',"$binpath/libengine.so",map {"/tmp/hold-h$_.manifest"}0..5);$report->{argv}=\@argv;
  my($gpu_text,$gpu_error,$gpu_rc)=run(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],\%ENV,15);check($gpu_rc==0,"GPU inventory failed exit=$gpu_rc: $gpu_error");my %gpus;
  for my $line(split /\n/,$gpu_text){my($i,$u)=split /,/,$line,2;check(defined($u),'invalid GPU inventory line');$i=~s/\A\s+|\s+\z//g;$u=~s/\A\s+|\s+\z//g;check($i=~/\A\d+\z/,'invalid GPU index');$gpus{0+$i}=$u if $i<6;}
  check(join(',',sort {$a<=>$b}keys %gpus) eq '0,1,2,3,4,5','need GPU indices 0 through 5');$report->{gpu_uuids}=\%gpus;
  for my $i(0..5){my $path=$ENV{HOME}.'/codex/asm-port/'.$gpus{$i}.'.lock';open my $f,'>>',$path or die "open lock $path: $!\n";flock($f,LOCK_SH|LOCK_NB) or die "GPU shared lock unavailable $path: $!\n";push @locks,$f;}
  my $private=tempdir('prefill-cancel-XXXXXX',DIR=>'/tmp',CLEANUP=>1);$sockpath="$private/core.sock";
  my %settings=(CUDA_DEVICE_ORDER=>'PCI_BUS_ID',CUDA_VISIBLE_DEVICES=>'0,1,2,3,4,5',SPLIT_SHARES=>'112,97,81,116,97,97',ATTN_MAP=>'1,1,1,1,4,4,4,4,5,5,5,5',ATTN_DIE=>'1',FLASH_MAXT=>'128',FLASH_ENGINE_MAXT=>'128',FLASH_MAXIMG=>'0',FLASH_KV_HOST=>$kv_host,FLASH_PLE_DISK=>'1',FLASH_PLE_OFF=>'192',FLASH_NBLK=>'4',FLASH_WATCHDOG_S=>'15',FLASH_ANCHOR_DIE=>'0',FLASH_LAYERS=>'48',FLASH_CUBIN=>"$binpath/die.cubin",FLASH_PREFILL_CHUNK=>'4',FLASH_SOCK=>$sockpath);
  my %env=map {$_=>$ENV{$_}}grep {!/^FLASH_/}keys %ENV;@env{keys %settings}=values %settings;$report->{environment}=\%settings;
  # Direct log files avoid blocking a core on an undrained output pipe.
  write_text("$out/stdout.log",'');write_text("$out/stderr.log",'');$child=spawn_core(\@argv,$binpath,\%env,"$out/stdout.log","$out/stderr.log");$report->{pid}=$child->{pid};open $reader,'<:raw',"$out/stdout.log" or die "open stdout log: $!\n";
  my $deadline=monotonic()+90;my $pending='';while(index($pending,"ready\n")<0 || !-e $sockpath){check(!defined(poll_child($child)),'core exited before ready');check(monotonic()<$deadline,'core startup deadline exceeded');my $n=sysread($reader,my $data,65536);if(!defined($n)){next if $!{EINTR};die "read core stdout: $!\n";}$pending.=$data if $n;sleep(.05) unless $n;}
  close $reader;$reader=undef;
  my $records=$report->{commands};my $configs=exchange($sockpath,'config',sub{index($_[0],'config ')==0},$records);my $config=$configs->[-1];$report->{config}=$config;my $actual=fields($config);$expected{two_position}='1';$expected{prefill_chunk}='4';for my $k(keys %expected){check(defined($actual->{$k}) && $actual->{$k} eq $expected{$k},"core config mismatch: $k");}
  my $client=connect_unix($sockpath,90);read_command($client,'snap 5',sub{index($_[0],'ok:')==0},$records);my $command='chat 1 '.join(' ',100..160);push @$records,{command=>$command,action=>'close_after_send'};send_text($client,$command."\n");close_socket($client);
  my $cancelled=state(exchange($sockpath,'statehash',sub{$_[0] eq 'statehash done'},$records));$report->{cancelled}=$cancelled;my($positions,$paired)=@$cancelled{qw(positions paired_positions)};check($positions>=0 && $positions<=4,"cancellation exceeded one chunk: $positions");check($paired>=0 && $paired<=$positions && $paired%2==0,'invalid paired counter');check(!defined(poll_child($child)),'core exited after cancellation');
  my $recovery='gen 1 '.join(' ',100..116);my $lines=exchange($sockpath,$recovery,sub{index($_[0],'time:')==0},$records);my @token_lines=grep {index($_,'tokens:')==0}@$lines;check(@token_lines==1,'missing or duplicate recovery tokens');my $token_text=$token_lines[0];$token_text=~s/\Atokens://;my @tokens=split ' ',$token_text;for(@tokens){/\A[+-]?\d+\z/ or die "invalid recovery token\n";$_=0+$_;}
  my $got=state(exchange($sockpath,'statehash',sub{$_[0] eq 'statehash done'},$records));$report->{recovery}={tokens=>\@tokens,time=>$lines->[-1],%$got};check(index($lines->[-1],'0 from the snapshot')>=0,'incomplete snapshot was reused');check(same(\@tokens,$oracle->{tokens}),'recovery tokens differ from scalar');check($got->{positions}==$oracle->{positions},'recovery positions differ');check(same($got->{state},$oracle->{state}),'recovery state differs from scalar');
 };$error=$@;
 if($reader){close $reader;}
 if($child){my $cleanup_error='';eval{stop_child($child,$sockpath);};$cleanup_error=$@;$error.=$cleanup_error if $cleanup_error;}
 if(!$error && $child){my $rc=poll_child($child);$error="core exit ".(defined($rc)?$rc:'still running')."\n" unless defined($rc) && $rc==0;}
 $report->{pass}=$error?JSON::PP::false:JSON::PP::true;$report->{error}=$error if $error;$report->{child_exit_code}=$child?poll_child($child):undef;
 put("$out/results.json",$report);print JSON::PP->new->canonical->encode({pass=>$report->{pass},error=>$report->{error},results=>"$out/results.json"}),"\n";return $error?1:0;
}
my $rc=eval{main()};if($@){print STDERR $@;exit 1;}exit $rc;
